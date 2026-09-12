from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SystemConfig:
    # --- Supported File Formats & Dimensions --------------------------------
    SUPPORTED_FORMATS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    MIN_IMAGE_DIMENSIONS: Tuple[int, int] = (224, 224)
    REQUIRED_CHANNELS: int = 3

    # NEW: main.py's TechnicalValidator needs an explicit max file size.
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    # --- Image Quality Thresholds --------------------------------------------
    # Original field, kept as-is so anything already using it still works.
    SHARPNESS_THRESHOLD: float = 100.0

    # NEW: main.py's QualityEvaluator expects this name specifically.
    # Same concept as SHARPNESS_THRESHOLD (Laplacian variance blur cutoff) --
    # kept in sync with the original value. If you ever change blur
    # sensitivity, update both together (or come back and I'll wire one to
    # reference the other properly).
    LAPLACIAN_BLUR_THRESHOLD: float = 100.0

    # NEW: exposure gating (0-255 pixel intensity range), not present before.
    MIN_EXPOSURE_THRESHOLD: float = 10.0
    MAX_EXPOSURE_THRESHOLD: float = 245.0

    # --- Preprocessing ---------------------------------------------------------
    # Original field, kept as-is.
    TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)
    NORMALISE_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    NORMALISE_STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # NEW: main.py's ImageNormaliser expects this name specifically.
    # Mirrors TARGET_IMAGE_SIZE.
    IMAGE_TARGET_SIZE: Tuple[int, int] = (224, 224)

    # --- Classification & Uncertainty ------------------------------------------
    # Original fields, kept as-is.
    CONFIDENCE_THRESHOLD: float = 0.60
    OOD_ENERGY_THRESHOLD: float = -5.0

    # NEW: main.py's UncertaintyEstimator expects these names specifically.
    ENTROPY_THRESHOLD: float = 1.0
    ENERGY_THRESHOLD: float = -5.0  # mirrors OOD_ENERGY_THRESHOLD

    # --- Backbone / Model Selection (NEW) ---------------------------------------
    # One of: "resnet50", "phikon", "uni", "virchow" -- whichever your
    # BackboneFactory.create() supports. Set to whatever you've actually
    # implemented so far.
    DEFAULT_BACKBONE: str = "resnet50"

    # --- Morphology Extraction (NEW) --------------------------------------------
    # Pixel-area bounds used by MorphologyExtractor's watershed segmentation
    # to filter out noise/artifact blobs. Placeholder values -- tune against
    # real slide data.
    MIN_NUCLEUS_AREA: int = 50
    MAX_NUCLEUS_AREA: int = 5000

    # --- Logging (NEW) -----------------------------------------------------------
    LOG_FILE_PATH: str = "logs/neck_cad_audit.log"

config = SystemConfig()