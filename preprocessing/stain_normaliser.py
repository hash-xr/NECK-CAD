import logging
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger("NECK-CAD.StainNormaliser")


class MacenkoStainNormaliser:
    """
    Normalises H&E or Pap/Giemsa cytopathological slide stain concentrations 
    using SVD-based Optical Density decomposition (Macenko et al. method).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        target_io: float = 240.0,
    ):
        """
        Args:
            alpha (float): Percentile threshold for stain vector extraction.
            beta (float): Optical density threshold for background masking.
            target_io (float): Transmitted light intensity constant.
        """
        self.alpha = alpha
        self.beta = beta
        self.target_io = target_io

        # Reference Stain Matrices for H&E / Cytology
        self.ref_stain_matrix = np.array(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float32
        )
        self.ref_max_concentrations = np.array([1.9705, 1.0308], dtype=np.float32)

    def _rgb_to_od(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts RGB image to Optical Density (OD) space."""
        img_float = img_rgb.astype(np.float32)
        img_float[img_float == 0] = 1.0  # Safeguard division by zero
        
        # Calculate Optical Density: OD = -log10(I / I_0)
        od = -np.log10(img_float / self.target_io)
        
        # OD threshold mask for non-background pixels
        od_hat = od[np.all(od >= self.beta, axis=2)]
        return od, od_hat

    def fit_transform(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Normalises stain intensities of the input RGB image tensor.

        Args:
            img_rgb (np.ndarray): Input image array (HxWx3, uint8).

        Returns:
            np.ndarray: Stain-normalised RGB image array (HxWx3, uint8).
        """
        try:
            h, w, c = img_rgb.shape
            od, od_hat = self._rgb_to_od(img_rgb)

            if len(od_hat) < 10:
                logger.warning("Insufficient tissue area for stain normalization. Returning original.")
                return img_rgb

            # Compute eigenvectors via SVD on OD space
            _, _, vh = np.linalg.svd(od_hat, full_matrices=False)
            plane = vh[:2]

            # Project OD points onto plane and calculate angles
            projected = np.dot(od_hat, plane.T)
            angles = np.arctan2(projected[:, 1], projected[:, 0])

            min_angle = np.percentile(angles, self.alpha)
            max_angle = np.percentile(angles, 100 - self.alpha)

            v_min = np.dot(plane.T, np.array([np.cos(min_angle), np.sin(min_angle)]))
            v_max = np.dot(plane.T, np.array([np.cos(max_angle), np.sin(max_angle)]))

            # Order stain vectors correctly
            if v_min[0] > v_max[0]:
                stain_matrix = np.array([v_min, v_max]).T
            else:
                stain_matrix = np.array([v_max, v_min]).T

            # Calculate individual pixel stain concentrations
            od_flat = od.reshape((-1, c)).T
            concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od_flat, rcond=None)

            # Normalise concentrations to reference target
            max_concentrations = np.percentile(concentrations, 99, axis=1)
            max_concentrations[max_concentrations == 0] = 1.0
            
            norm_concentrations = concentrations / max_concentrations[:, None]
            norm_concentrations *= self.ref_max_concentrations[:, None]

            # Reconstruct OD space and convert back to RGB
            norm_od = np.dot(self.ref_stain_matrix, norm_concentrations)
            norm_rgb = self.target_io * (10.0 ** (-norm_od))
            
            norm_rgb = np.clip(norm_rgb.T.reshape((h, w, c)), 0, 255).astype(np.uint8)
            return norm_rgb

        except Exception as e:
            logger.error(f"Stain normalization failed: {e}. Returning original image.")
            return img_rgb