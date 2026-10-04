# extract_features.py
import os
import logging
import pandas as pd
import torch
import numpy as np
from PIL import Image
import uuid
from main import NeckCADController
from core.schema import ImageMetadata, ImageStatus

# Set up logging to track the feature pipeline execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NECK-CAD.FeatureExtraction")

def cache_dataset_features(data_dir: str, output_path: str):
    """
    Extracts deep learning features and quantitative morphology metrics from raw patient slide FOVs, 
    matching them dynamically with clinical labels from a ground-truth CSV file.
    """
    controller = NeckCADController()
    dataset_samples = []
    
    # 1. Dynamically load the ground-truth metadata sheets
    metadata_csv_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(
            f"Ground-truth metadata file not found at: '{metadata_csv_path}'. "
            f"Please ensure a 'metadata.csv' containing 'patient_id', 'primary', 'milan', and 'entity' "
            f"columns exists inside your raw data directory before executing feature collection."
        )
        
    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        # Build lookup table mapping patient_id string -> row dictionaries
        metadata_lookup = metadata_df.set_index("patient_id").to_dict(orient="index")
        logger.info(f"Loaded ground-truth records for {len(metadata_lookup)} patients from CSV.")
    except Exception as e:
        raise RuntimeError(f"Failed to parse ground-truth metadata spreadsheet: {str(e)}")

    # Ensure output destination subdirectory directory exists 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Iterate through patient folders
    for patient_id in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_path): 
            continue
        if patient_id == "metadata.csv":
            continue
            
        # Dynamically retrieve true labels for this exact patient
        patient_labels = metadata_lookup.get(patient_id)
        if not patient_labels:
            logger.warning(f"Skipping directory '{patient_id}': No records found in ground-truth metadata sheet.")
            continue
            
        # Parse targets directly out of the ground-truth data mapping 
        primary_class = int(patient_labels["primary"])
        milan_class = int(patient_labels["milan"])
        entity_class = int(patient_labels["entity"])
        
        embeddings, morph_vectors, quality_scores = [], [], []
        
        # Process patient slides through pipeline steps 1-6
        for img_name in os.listdir(patient_path):
            img_path = os.path.join(patient_path, img_name)
            
            try:
                raw_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error(f"Skipping unreadable or corrupt file '{img_name}': {str(e)}")
                continue

            # Create standard metadata shell tracking properties
            meta = ImageMetadata(
                image_id=str(uuid.uuid4()), 
                file_path=img_path, 
                filename=img_name, 
                status=ImageStatus.PENDING
            )
            
            # Step A: Evaluate Image Quality (Replaces hardcoded 1.0 quality scores)
            meta = controller.evaluator.evaluate_quality(meta, raw_img)
            if meta.status != ImageStatus.VALID:
                logger.warning(f"Image {img_name} rejected by Quality Engine (Score: {meta.quality_score:.2f}). Skipping.")
                continue

            # Step B: Run the safe preprocess workflow (Resizing, Stain Norm, Tensor scaling)
            meta = controller.normaliser.process(meta, raw_img)
            
            # Step C: Pull the safely structured tensor directly off the metadata payload
            if meta.status == ImageStatus.VALID and meta.processed_tensor is not None:
                img_tensor = meta.processed_tensor.to(controller.device)
            else:
                continue 
                
            # Morphology & Deep Feature Extraction
            stain_norm_pil = controller.normaliser.normalise_stain(raw_img)
            morph_metrics = controller.morphology_extractor.extract_features(np.array(stain_norm_pil))
            
            with torch.no_grad():
                deep_emb = controller.backbone(img_tensor)
                
            embeddings.append(deep_emb.cpu())
            morph_vectors.append(torch.from_numpy(morph_metrics.to_vector()).unsqueeze(0))
            
            # Append the actual calculated quality score value directly from the quality analyzer module
            quality_scores.append(float(meta.quality_score))
            
        if embeddings:
            dataset_samples.append({
                "patient_id": patient_id,
                "embeddings": torch.cat(embeddings, dim=0),       # [N, Dim]
                "morphology": torch.cat(morph_vectors, dim=0),     # [N, 9]
                "quality_scores": quality_scores,                 # Real metrics list
                "primary": primary_class,                         # Linked CSV Target Class
                "milan": milan_class,                             # Linked CSV Target Milan label
                "entity": entity_class,                           # Linked CSV Specific Diagnosis index
            })
            logger.info(f"Successfully processed and extracted features for patient: {patient_id}")
            
    torch.save(dataset_samples, output_path)
    logger.info(f"Audit Complete: Saved {len(dataset_samples)} real patient matrices to '{output_path}'.")

if __name__ == "__main__":
    cache_dataset_features("./raw_data", "data/processed_features.pt")
