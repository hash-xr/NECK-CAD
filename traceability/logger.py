import logging
import os
from datetime import datetime

class AuditLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"neck_cad_{timestamp}.log")
        
        self.logger = logging.getLogger("NECK_CAD")
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding duplicate handlers if logger is re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, message: str, level: str = "INFO") -> None:
        if level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        else:
            self.logger.info(message)

    def get_log_path(self) -> str:
        return self.log_file