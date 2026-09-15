# extract_features.py
import os, torch, numpy as np
from PIL import Image
from main import NeckCADController
import uuid

def cache_dataset_features(data_dir: str, output_path: str):
    controller = NeckCADController()
    dataset_samples = []
    
    # Iterate through patient folders
    for patient_id in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_path): continue
        
        embeddings, morph_vectors, quality_scores = [], [], []
        # Process patient slides through pipeline steps 1-6
        for img_name in os.listdir(patient_path):
            img_path = os.path.join(patient_path, img_name)
            raw_img = Image.open(img_path).convert("RGB")

            # 1. Create a minimal metadata shell for the image
            from core.schema import ImageMetadata, ImageStatus
            meta = ImageMetadata(image_id=str(uuid.uuid4()), file_path=img_path, filename=img_name, status=ImageStatus.VALID)
            
            # 2. Run the full, safe process workflow 
            meta = controller.normaliser.process(meta, raw_img)
            
            # 3. Pull the safely structured tensor directly off the metadata payload
            if meta.status == ImageStatus.VALID and meta.processed_tensor is not None:
                img_tensor = meta.processed_tensor.to(controller.device)
            else:
                continue # Skip corrupt files gracefully
            # Morphology & Deep Feature Extraction
            stain_norm_pil = controller.normaliser.normalise_stain(raw_img)
            morph_metrics = controller.morphology_extractor.extract_features(np.array(stain_norm_pil))
            with torch.no_grad():
                deep_emb = controller.backbone(img_tensor)
                
            embeddings.append(deep_emb.cpu())
            morph_vectors.append(torch.from_numpy(morph_metrics.to_vector()).unsqueeze(0))
            quality_scores.append(1.0)
            
        if embeddings:
            dataset_samples.append({
                "patient_id": patient_id,
                "embeddings": torch.cat(embeddings, dim=0),      # [N, Dim]
                "morphology": torch.cat(morph_vectors, dim=0),     # [N, 9]
                "quality_scores": quality_scores,
                "primary": 2,  # Target class label from metadata
                "milan": 3,    # Target Milan label from metadata
                "entity": 1,
            })
            
    torch.save(dataset_samples, output_path)
    print(f"Cached {len(dataset_samples)} patient cases to '{output_path}'.")

if __name__ == "__main__":
    cache_dataset_features("./raw_data", "data/processed_features.pt")