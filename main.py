import os
import logging
from typing import Optional
import uuid

import torch
from PIL import Image
import numpy as np

from config.settings import SystemConfig
from core.schema import (
    DiagnosticSession,
    ImageMetadata,
    ImageStatus,
    AggregatedResult,
    MilanCategory,
)
from traceability.logger import AuditLogger
from input.receiver import InputReceiver
from validation.validator import TechnicalValidator
from quality.evaluator import QualityEvaluator
from preprocessing.normaliser import ImageNormaliser
from artifacts.detector import CytologyArtifactDetector
from models.backbone_factory import BackboneFactory
from morphology.extractor import MorphologyExtractor
from aggregation.fusion import QualityWeightedAttentionFusion
from diagnosis.classifier import HybridDiagnosticClassifier
from uncertainty.estimator import UncertaintyEstimator
from explainability.visualiser import GradCAMVisualiser
from reporting.generator import ClinicalReportGenerator

logger = logging.getLogger("NECK-CAD.Main")


class NeckCADController:
    """Central orchestrator connecting pipeline modules to the GUI layer."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.audit_logger = AuditLogger()

        # --- Pipeline Engines ---
        self.receiver = InputReceiver(logger=self.audit_logger)
        self.validator = TechnicalValidator(config=self.config, logger=self.audit_logger)
        self.evaluator = QualityEvaluator(config=self.config, logger=self.audit_logger)
        self.normaliser = ImageNormaliser(config=self.config, logger=self.audit_logger)
        self.artifact_detector = CytologyArtifactDetector(logger=self.audit_logger)

        # --- Feature Extraction & Model Engines ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.backbone = BackboneFactory.create(
            model_name=self.config.DEFAULT_BACKBONE, pretrained=True
        ).to(self.device)
        self.backbone.eval()

        self.morphology_extractor = MorphologyExtractor(
            backend="watershed",
            min_area=self.config.MIN_NUCLEUS_AREA,
            max_area=self.config.MAX_NUCLEUS_AREA,
        )

        embedding_dim = self.backbone.get_embedding_dim()
        self.fusion_engine = QualityWeightedAttentionFusion(feature_dim=embedding_dim).to(self.device)

        self.classifier = HybridDiagnosticClassifier(
            embedding_dim=embedding_dim,
            morphology_dim=9,
            num_primary=3,
            num_milan=7,
            num_entities=10,
        ).to(self.device)

        # Load Trained Checkpoint Weights if Available
        checkpoint_path = os.path.join("./checkpoints", "best_neckcad_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.fusion_engine.load_state_dict(checkpoint["fusion_state_dict"])
            self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            self.audit_logger.log(f"Loaded trained model weights from '{checkpoint_path}'.")
        else:
            logger.warning("No trained checkpoint found at load time. Running with initialised weights.")

        self.fusion_engine.eval()
        self.classifier.eval()

        self.uncertainty_estimator = UncertaintyEstimator(
            config=self.config, logger=self.audit_logger
        )
        self.visualiser = GradCAMVisualiser(self.audit_logger)
        self.reporter = ClinicalReportGenerator(self.audit_logger)

        logger.info(
            f"NeckCADController initialised successfully on device '{self.device}' "
            f"using backbone '{self.config.DEFAULT_BACKBONE}'."
        )

    def create_session(self, file_paths: list[str]) -> DiagnosticSession:
        session_id = str(uuid.uuid4())[:8]
        session = DiagnosticSession(session_id=session_id)
        
        for index, path in enumerate(file_paths):
            image_id = f"IMG_{session_id}_{index+1:02d}"
            metadata = ImageMetadata(
                image_id=image_id,
                file_path=path,
                filename=os.path.basename(path)
            )
            session.images.append(metadata)
            
        self.audit_logger.log(f"Session {session_id} created with {len(file_paths)} image(s).")
        return session

    def process_session(self, session: DiagnosticSession) -> DiagnosticSession:
        """Executes the full diagnostic analysis pipeline across session images."""
        valid_tensors = []
        valid_morphology_vectors = []
        valid_quality_scores = []
        valid_image_ids = []
        valid_raw_images = []  # kept for real Grad-CAM generation later

        for img_meta in session.images:
            # 1. Ingestion / graceful load
            img_meta, raw_img = self.receiver.load_image(img_meta)
            if raw_img is None or img_meta.status != ImageStatus.PENDING:
                continue

            # 2. Technical File Validation
            img_meta = self.validator.validate(img_meta, raw_img)
            if img_meta.status != ImageStatus.VALID:
                continue

            # 3. Quality Control, Blur Assessment & Artifact Detection
            img_meta = self.evaluator.evaluate_quality(img_meta, raw_img)
            if img_meta.status != ImageStatus.VALID:
                continue

            # 3b. Cytology Artifact Quality Assessment
            np_raw = np.array(raw_img.convert("RGB"))
            artifact_res = self.artifact_detector.assess_fov(np_raw, image_id=img_meta.image_id)
            if not artifact_res.is_usable:
                img_meta.status = ImageStatus.REJECTED_QUALITY
                img_meta.rejection_reason = " | ".join(artifact_res.flag_reasons)
                continue
            
            # 4. Preprocessing
            pil_img = raw_img.convert("RGB")
            stain_norm_pil = self.normaliser.normalise_stain(pil_img)
            img_meta = self.normaliser.process(img_meta, stain_norm_pil)

            if img_meta.status == ImageStatus.REJECTED_QUALITY:
                continue
            if img_meta.processed_tensor is not None:
                img_meta.processed_tensor = img_meta.processed_tensor.to(self.device)

            # 5. Quantitative Morphology Extraction
            np_img = np.array(stain_norm_pil)
            morph_metrics = self.morphology_extractor.extract_features(np_img)
            morph_vector = torch.from_numpy(morph_metrics.to_vector()).unsqueeze(0).to(self.device)

            # 6. Backbone Deep Feature Extraction (frozen)
            with torch.no_grad():
                deep_embedding = self.backbone(img_meta.processed_tensor)

            valid_tensors.append(deep_embedding)
            valid_morphology_vectors.append(morph_vector)
            valid_quality_scores.append(img_meta.quality_score)
            valid_image_ids.append(img_meta.image_id)
            valid_raw_images.append(raw_img)

        if not valid_tensors:
            logger.warning(f"Session {session.session_id}: No valid image FOVs passed quality gating.")
            return session

        # 7. Multi-FOV Evidence Aggregation
        stacked_embeddings = torch.cat(valid_tensors, dim=0)
        stacked_morphology = torch.cat(valid_morphology_vectors, dim=0)
        case_morphology = torch.mean(stacked_morphology, dim=0, keepdim=True)

        case_embedding, attention_weights = self.fusion_engine(
            stacked_embeddings, quality_scores=valid_quality_scores
        )

        # 8. Multi-Task Diagnostic Classification
        with torch.no_grad():
            outputs = self.classifier(case_embedding, case_morphology)

        milan_probs = outputs["milan_probs"].squeeze(0).cpu().numpy()
        primary_probs = outputs["primary_probs"].squeeze(0).cpu().numpy()

        # 9. Uncertainty & OOD Analysis
        milan_labels = [
            MilanCategory.I_NON_DIAGNOSTIC,
            MilanCategory.II_NON_NEOPLASTIC,
            MilanCategory.III_AUS,
            MilanCategory.IVA_NEOPLASM_BENIGN,
            MilanCategory.IVB_SUMP,
            MilanCategory.V_SUSPICIOUS,
            MilanCategory.VI_MALIGNANT,
        ]
        top_milan_idx = int(np.argmax(milan_probs))
        selected_milan = milan_labels[top_milan_idx]

        primary_labels = ["Non-Diagnostic", "Non-Neoplastic", "Neoplastic"]
        top_primary_idx = int(np.argmax(primary_probs))
        selected_primary = primary_labels[top_primary_idx]

        milan_probs_tensor = outputs["milan_probs"]
        milan_logits_tensor = outputs["milan_logits"]

        entropy = self.uncertainty_estimator.calculate_entropy(milan_probs_tensor)
        energy = self.uncertainty_estimator.calculate_energy(milan_logits_tensor)

        top_confidence = float(milan_probs[top_milan_idx])
        is_uncertain = top_confidence < self.uncertainty_estimator.config.CONFIDENCE_THRESHOLD or entropy > 0.75
        is_ood = energy > self.uncertainty_estimator.config.OOD_ENERGY_THRESHOLD

        session.aggregated_result = AggregatedResult(
            primary_category=selected_primary,
            milan_category=selected_milan.value,
            specific_diagnosis="Trained Multi-Task Classification",
            confidence_scores={
                "milan_confidence": top_confidence,
                "primary_confidence": float(primary_probs[top_primary_idx]),
                "shannon_entropy": float(entropy),
                "free_energy": float(energy),
            },
            differential_diagnosis=[
                (milan_labels[i].value, float(milan_probs[i])) for i in range(len(milan_labels))
            ],
            is_uncertain=is_uncertain,
            is_ood=is_ood,
        )

        # 10. Visual Explainability (real Grad-CAM overlays)
        for idx, img_id in enumerate(valid_image_ids):
            overlay = self.visualiser.generate_heatmap(
                valid_raw_images[idx], valid_tensors[idx]
            )
            session.explanations[img_id] = overlay

        self.audit_logger.log(
            message=f"Session {session.session_id} complete. Milan Category: {selected_milan.value}.",
        )
        return session