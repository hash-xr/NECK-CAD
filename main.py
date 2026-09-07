import uuid
from typing import List
from config.settings import config
from core.schema import AnalysisSession, ImageMetadata, ImageStatus

class NeckCADController:
    def __init__(self):
        self.config = config

    def create_session(self, file_paths: List[str]) -> AnalysisSession:
        session_id = str(uuid.uuid4())[:8]
        session = AnalysisSession(session_id=session_id)
        
        for index, path in enumerate(file_paths):
            image_id = f"IMG_{session_id}_{index+1:02d}"
            metadata = ImageMetadata(
                image_id=image_id,
                file_path=path,
                filename=path.split("/")[-1].split("\\")[-1]
            )
            session.images.append(metadata)
            
        session.logs.append(f"Session {session_id} created with {len(file_paths)} image(s).")
        return session

    def run_pipeline(self, session: AnalysisSession) -> AnalysisSession:
        # Pipeline orchestration steps will be called here as modules are built:
        # 1. Technical Validation
        # 2. Quality Assessment
        # 3. Preprocessing
        # 4. Diagnostic Inference & Multi-Image Aggregation
        # 5. Uncertainty & Explainability
        # 6. Report Generation
        return session

if __name__ == "__main__":
    controller = NeckCADController()
    print("NECK-CAD Controller Initialized.")