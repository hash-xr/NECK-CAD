import os
import uuid
import torch
from PIL import Image
from config.settings import config
from core.schema import AnalysisSession, ImageMetadata, ImageStatus
from traceability.logger import AuditLogger
from input.receiver import InputReceiver
from validation.validator import TechnicalValidator
from quality.evaluator import QualityEvaluator
from preprocessing.normaliser import ImageNormaliser
from diagnosis.classifier import DiagnosticClassifier
from aggregation.fusion import MultiImageFusionEngine
from uncertainty.estimator import UncertaintyEstimator
from explainability.visualiser import GradCAMVisualiser
from reporting.generator import ClinicalReportGenerator

class NeckCADController:
    def __init__(self):
        self.config = config
        self.logger = AuditLogger()
        self.receiver = InputReceiver(self.logger)
        self.validator = TechnicalValidator(self.config, self.logger)
        self.evaluator = QualityEvaluator(self.config, self.logger)
        self.normaliser = ImageNormaliser(self.config, self.logger)
        self.classifier = DiagnosticClassifier(self.config, self.logger)
        self.fusion_engine = MultiImageFusionEngine(self.logger)
        self.uncertainty_estimator = UncertaintyEstimator(self.config, self.logger)
        self.visualiser = GradCAMVisualiser(self.logger)
        self.reporter = ClinicalReportGenerator(self.logger)
        self.logger.log("NECK-CAD Controller initialised with Phase 6 Reporting Engine.")

    def create_session(self, file_paths: list[str]) -> AnalysisSession:
        session_id = str(uuid.uuid4())[:8]
        session = AnalysisSession(session_id=session_id)
        
        for index, path in enumerate(file_paths):
            image_id = f"IMG_{session_id}_{index+1:02d}"
            metadata = ImageMetadata(
                image_id=image_id,
                file_path=path,
                filename=os.path.basename(path)
            )
            session.images.append(metadata)
            
        self.logger.log(f"Session {session_id} created with {len(file_paths)} image(s).")
        return session

    def process_session(self, session: AnalysisSession) -> AnalysisSession:
        predictions_list = []
        valid_metadata_list = []

        for metadata in session.images:
            # 1. Ingestion
            metadata, raw_img = self.receiver.load_image(metadata)
            if raw_img is None or metadata.status != ImageStatus.PENDING:
                continue

            # 2. Technical Validation
            metadata = self.validator.validate(metadata, raw_img)
            if metadata.status != ImageStatus.VALID:
                continue

            # 3. Quality Control
            metadata = self.evaluator.evaluate(metadata, raw_img)
            if metadata.status != ImageStatus.VALID:
                continue

            # 4. Preprocessing
            metadata = self.normaliser.process(metadata, raw_img)
            if metadata.status != ImageStatus.VALID or metadata.processed_tensor is None:
                continue

            # 5. Diagnostic Inference
            preds = self.classifier.predict_single_image(metadata.processed_tensor)
            predictions_list.append(preds)
            valid_metadata_list.append(metadata)

            # 6. Generate Visual Explanation (Grad-CAM Overlay)
            overlay = self.visualiser.generate_heatmap(raw_img, preds["features"])
            session.explanations[metadata.image_id] = overlay

        # 7. Multi-Image Aggregation
        session.aggregated_result = self.fusion_engine.aggregate_session_predictions(
            predictions_list, valid_metadata_list
        )

        # 8. Uncertainty & OOD Safeguard Check
        if session.aggregated_result and predictions_list:
            avg_milan_probs = torch.mean(
                torch.stack([p["milan_probs"] for p in predictions_list]), dim=0
            )
            simulated_raw_logits = torch.log(avg_milan_probs + 1e-8)
            
            session.aggregated_result = self.uncertainty_estimator.evaluate_uncertainty(
                session.aggregated_result, avg_milan_probs, simulated_raw_logits
            )

        # 9. Clinical Report Generation
        self.reporter.save_report(session)

        self.logger.log(f"Session {session.session_id} execution and reporting complete.")
        return session

if __name__ == "__main__":
    controller = NeckCADController()
    
    test_img_dir = "tests"
    os.makedirs(test_img_dir, exist_ok=True)
    sample_path = os.path.join(test_img_dir, "sample_test.png")
    
    if not os.path.exists(sample_path):
        dummy_img = Image.new("RGB", (300, 300), color=(180, 120, 160))
        dummy_img.save(sample_path)

    session = controller.create_session([sample_path])
    processed_session = controller.process_session(session)
    
    print("\n" + controller.reporter.format_text_report(processed_session)) 