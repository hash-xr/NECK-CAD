import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.segmentation import watershed

logger = logging.getLogger("NECK-CAD.Morphology")


@dataclass
class FOVMorphologyMetrics:
    """Quantitative cytological metrics aggregated across a single FOV."""

    nucleus_count: int
    mean_nuclear_area: float
    std_nuclear_area: float
    anisonucleosis_index: float  # Coefficient of Variation (std / mean)
    mean_circularity: float
    mean_solidity: float
    mean_eccentricity: float
    mean_optical_density: float  # Chromatin darkness / hyperchromasia
    mean_nc_ratio: float  # True Nuclear-to-Cytoplasmic ratio
    raw_feature_matrix: List[List[float]]

    def to_vector(self) -> np.ndarray:
        """Converts summary statistics to a 1D feature array for model fusion."""
        return np.array(
            [
                float(self.nucleus_count),
                self.mean_nuclear_area,
                self.std_nuclear_area,
                self.anisonucleosis_index,
                self.mean_circularity,
                self.mean_solidity,
                self.mean_eccentricity,
                self.mean_optical_density,
                self.mean_nc_ratio,
            ],
            dtype=np.float32,
        )


class MorphologyExtractor:
    """
    Extracts quantitative nuclear and cytoplasmic morphological features
    with Voronoi label partitioning and dynamic scanner white-point calibration.
    """

    def __init__(
        self,
        backend: str = "watershed",
        min_area: int = 50,
        max_area: int = 5000,
        max_cyto_radius: int = 15,
    ):
        self.backend = backend.lower()
        self.min_area = min_area
        self.max_area = max_area
        self.max_cyto_radius = max_cyto_radius

    def segment_nuclei(self, image_rgb: np.ndarray) -> np.ndarray:
        """Generates an integer labelled mask where each nucleus has a unique ID > 0."""
        gray = rgb2gray(image_rgb)
        inverted = 1.0 - gray
        blur = cv2.GaussianBlur((inverted * 255).astype(np.uint8), (5, 5), 0)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0) #type: ignore
        sure_fg = sure_fg.astype(np.uint8)

        return label(sure_fg) #type: ignore

    def _partition_cytoplasm_voronoi(self, nuclear_mask: np.ndarray) -> np.ndarray:
        """
        Allocates surrounding cytoplasmic pixels non-overlappingly using
        a distance-constrained Voronoi watershed boundary.
        """
        if nuclear_mask.max() == 0:
            return nuclear_mask

        # Mask of region within allowed distance from any nucleus
        binary_nuclei = (nuclear_mask > 0).astype(np.uint8)
        dilated_zone = cv2.dilate(
            binary_nuclei,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.max_cyto_radius, self.max_cyto_radius)),
        )

        # Distance map from nucleus boundaries
        dist_map = cv2.distanceTransform(1 - binary_nuclei, cv2.DIST_L2, 5)

        # Watershed allocating pixels to nearest nucleus label without crossing midlines
        cell_mask = watershed(dist_map, markers=nuclear_mask, mask=dilated_zone)
        return cell_mask

    def extract_features(self, image_rgb: np.ndarray) -> FOVMorphologyMetrics:
        """Extracts calibrated morphological features from an input FOV."""
        nuclear_mask = self.segment_nuclei(image_rgb)
        gray_image = rgb2gray(image_rgb)

        # Dynamic White Point Calibration (I_0 calculated from 99th percentile)
        i_0 = float(np.percentile(gray_image, 99))
        i_0 = max(i_0, 0.1)  # Safeguard against dark images
        epsilon = 1e-5

        # Calibrated Optical Density map
        clipped_gray = np.clip(gray_image, epsilon, i_0)
        optical_density_map = -np.log10(clipped_gray / i_0)

        # Non-overlapping Cytoplasm Partitioning
        cell_mask = self._partition_cytoplasm_voronoi(nuclear_mask)

        cell_counts = np.bincount(cell_mask.ravel())

        props = regionprops(nuclear_mask, intensity_image=optical_density_map)

        areas = []
        circularities = []
        solidities = []
        eccentricities = []
        optical_densities = []
        nc_ratios = []
        raw_rows = []

        for p in props:
            area = p.area
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = p.perimeter if p.perimeter > 0 else 1.0
            circularity = (4.0 * np.pi * area) / (perimeter**2)
            solidity = p.solidity
            eccentricity = p.eccentricity
            mean_od = p.mean_intensity

            # Calculate Cytoplasm Area with Voronoi constraint
            total_cell_area = cell_counts[p.label] if p.label < len(cell_counts) else area
            cytoplasm_area = max(float(total_cell_area - area), 1.0)  # Avoid division by zero

            # True Nuclear-to-Cytoplasmic (N:C) Ratio
            nc_ratio = float(area / cytoplasm_area)

            areas.append(area)
            circularities.append(circularity)
            solidities.append(solidity)
            eccentricities.append(eccentricity)
            optical_densities.append(mean_od)
            nc_ratios.append(nc_ratio)

            raw_rows.append([area, perimeter, circularity, solidity, eccentricity, mean_od, nc_ratio])

        nucleus_count = len(areas)

        if nucleus_count == 0:
            return FOVMorphologyMetrics(
                nucleus_count=0,
                mean_nuclear_area=0.0,
                std_nuclear_area=0.0,
                anisonucleosis_index=0.0,
                mean_circularity=0.0,
                mean_solidity=0.0,
                mean_eccentricity=0.0,
                mean_optical_density=0.0,
                mean_nc_ratio=0.0,
                raw_feature_matrix=[],
            )

        mean_area = float(np.mean(areas))
        std_area = float(np.std(areas))
        anisonucleosis = std_area / mean_area if mean_area > 0 else 0.0

        return FOVMorphologyMetrics(
            nucleus_count=nucleus_count,
            mean_nuclear_area=mean_area,
            std_nuclear_area=std_area,
            anisonucleosis_index=float(anisonucleosis),
            mean_circularity=float(np.mean(circularities)),
            mean_solidity=float(np.mean(solidities)),
            mean_eccentricity=float(np.mean(eccentricities)),
            mean_optical_density=float(np.mean(optical_densities)),
            mean_nc_ratio=float(np.mean(nc_ratios)),
            raw_feature_matrix=raw_rows,
        )