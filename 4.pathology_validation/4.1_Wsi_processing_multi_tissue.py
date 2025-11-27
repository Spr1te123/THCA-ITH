import os
import numpy as np
import cv2
import joblib
from datetime import datetime
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox import logger
import matplotlib

matplotlib.use('Agg')  # 设置非交互式后端，避免X11问题
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.patches import Rectangle
import tempfile
import shutil
import time
import json
from sklearn.decomposition import DictionaryLearning
import random
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import sys
import logging
from logging.handlers import RotatingFileHandler

# Enhanced color dictionary with brighter colors for better visibility
COLOR_DICT = {
    0: ("Background", (255, 165, 0)),  # Orange
    1: ("Neoplastic Epithelial", (255, 0, 0)),  # Red
    2: ("Inflammatory", (255, 255, 0)),  # Yellow
    3: ("Connective", (0, 255, 0)),  # Green
    4: ("Dead", (128, 128, 128)),  # Gray
    5: ("Non-neoplastic Epithelial", (0, 0, 255)),  # Blue
}


# 日志设置函数（保持原样）
def setup_logging(output_dir, log_name=None):
    """设置日志系统，同时输出到控制台和文件"""

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建日志文件名
    if log_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f"wsi_processing_{timestamp}.log"

    log_path = os.path.join(output_dir, log_name)

    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除现有的处理器
    root_logger.handlers = []

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # 同时配置 tiatoolbox 的日志器
    tia_logger = logging.getLogger('tiatoolbox')
    tia_logger.setLevel(logging.INFO)

    logger.info(f"Logging initialized. Log file: {log_path}")

    return log_path


# Quick Test Configuration（保持原样）
class QuickTestConfig:
    """Configuration for quick test mode with enhanced visualization"""

    def __init__(self, enabled=False):
        self.enabled = enabled
        if enabled:
            # Quick test settings with better visualization
            self.max_tiles = 3
            self.tile_size = 2000
            self.batch_size = 64
            self.visualization_sample_rate = 5  # Show more nuclei (every 5th)
            self.max_patches = 8
            self.num_norm_examples = 2
            self.contour_thickness = 3  # Thicker contours for visibility
            logger.info("Quick test mode enabled:")
            logger.info(f"  - Max tiles: {self.max_tiles}")
            logger.info(f"  - Tile size: {self.tile_size}")
            logger.info(f"  - Batch size: {self.batch_size}")
            logger.info(f"  - Visualization sample rate: 1/{self.visualization_sample_rate}")
        else:
            # Normal settings
            self.max_tiles = None
            self.tile_size = 5000
            self.batch_size = 512  # Larger batch for your powerful GPUs
            self.visualization_sample_rate = 20  # Show every 20th nucleus
            self.max_patches = 16
            self.num_norm_examples = 5
            self.contour_thickness = 2


# ============= Stain Normalization Classes (保持原样) =============

class StainNormalizer:
    """Base class for stain normalization"""

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_image):
        """Fit normalizer to target image"""
        raise NotImplementedError

    def transform(self, source_image):
        """Transform source image to match target"""
        raise NotImplementedError


class ReinhardNormalizer(StainNormalizer):
    """Reinhard color normalization in LAB color space"""

    def __init__(self):
        super().__init__()

    def fit(self, target_image):
        """Compute target statistics"""
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float32)

        self.target_means = []
        self.target_stds = []

        for i in range(3):
            channel = target_lab[:, :, i].flatten()
            self.target_means.append(np.mean(channel))
            self.target_stds.append(np.std(channel))

        return self

    def transform(self, source_image):
        """Apply Reinhard normalization"""
        if self.target_means is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB).astype(np.float32)

        for i in range(3):
            channel = source_lab[:, :, i]
            source_mean = np.mean(channel)
            source_std = np.std(channel)

            if source_std > 0:
                channel = (channel - source_mean) / source_std
                channel = channel * self.target_stds[i] + self.target_means[i]
                source_lab[:, :, i] = channel

        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        normalized = cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)

        return normalized


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalization for H&E images"""

    def __init__(self, alpha=1, beta=0.15):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.stain_matrix_target = None
        self.maxC_target = None

    def get_stain_matrix(self, image):
        """Extract stain matrix using SVD"""
        OD = self.RGB_to_OD(image)

        OD_flat = OD.reshape(-1, 3)
        OD_thresh = OD_flat[np.any(OD_flat > self.beta, axis=1)]

        if len(OD_thresh) == 0:
            return None

        _, eigvecs = np.linalg.eigh(np.cov(OD_thresh.T))
        eigvecs = eigvecs[:, [2, 1]]

        proj = OD_thresh @ eigvecs
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        vec1 = eigvecs @ np.array([np.cos(min_angle), np.sin(min_angle)])
        vec2 = eigvecs @ np.array([np.cos(max_angle), np.sin(max_angle)])

        if vec1[0] > vec2[0]:
            HE = np.array([vec1, vec2]).T
        else:
            HE = np.array([vec2, vec1]).T

        return self.normalize_rows(HE)

    def fit(self, target_image):
        """Fit to target image"""
        self.stain_matrix_target = self.get_stain_matrix(target_image)

        if self.stain_matrix_target is not None:
            OD = self.RGB_to_OD(target_image)
            C = self.get_concentrations(OD, self.stain_matrix_target)
            self.maxC_target = np.percentile(C, 99, axis=0)

        return self

    def transform(self, source_image):
        """Transform source image"""
        if self.stain_matrix_target is None:
            logger.warning("Target stain matrix not computed, returning original image")
            return source_image

        stain_matrix_source = self.get_stain_matrix(source_image)

        if stain_matrix_source is None:
            logger.warning("Could not compute source stain matrix, returning original image")
            return source_image

        OD = self.RGB_to_OD(source_image)
        C = self.get_concentrations(OD, stain_matrix_source)

        maxC_source = np.percentile(C, 99, axis=0)
        C = C * (self.maxC_target / maxC_source)

        OD_normalized = C @ self.stain_matrix_target.T
        normalized = self.OD_to_RGB(OD_normalized.reshape(source_image.shape))

        return normalized

    @staticmethod
    def RGB_to_OD(image):
        """Convert RGB to optical density"""
        image = image.astype(np.float32) + 1
        return -np.log(image / 255.0)

    @staticmethod
    def OD_to_RGB(OD):
        """Convert optical density to RGB"""
        return (255 * np.exp(-OD)).astype(np.uint8)

    @staticmethod
    def normalize_rows(A):
        """Normalize rows of matrix"""
        return A / np.linalg.norm(A, axis=1)[:, None]

    def get_concentrations(self, OD, stain_matrix):
        """Get stain concentrations"""
        OD_flat = OD.reshape(-1, 3)
        return OD_flat @ np.linalg.pinv(stain_matrix.T)


class VahadaneNormalizer(StainNormalizer):
    """Vahadane stain normalization using sparse NMF"""

    def __init__(self, lambda_=0.1):
        super().__init__()
        self.lambda_ = lambda_
        self.stain_matrix_target = None

    def get_stain_matrix(self, image):
        """Extract stain matrix using sparse NMF"""
        try:
            OD = MacenkoNormalizer.RGB_to_OD(image)
            OD_flat = OD.reshape(-1, 3).T

            dl = DictionaryLearning(
                n_components=2,
                alpha=self.lambda_,
                transform_algorithm='lasso_lars',
                positive_dict=True,
                positive_code=True,
                max_iter=10
            )

            dl.fit(OD_flat.T)
            stain_matrix = dl.components_.T

            if stain_matrix[0, 0] < stain_matrix[0, 1]:
                stain_matrix = stain_matrix[:, [1, 0]]

            return stain_matrix

        except Exception as e:
            logger.warning(f"Vahadane normalization failed: {e}")
            return None

    def fit(self, target_image):
        """Fit to target image"""
        self.stain_matrix_target = self.get_stain_matrix(target_image)
        return self

    def transform(self, source_image):
        """Transform source image"""
        if self.stain_matrix_target is None:
            return source_image

        stain_matrix_source = self.get_stain_matrix(source_image)

        if stain_matrix_source is None:
            return source_image

        OD = MacenkoNormalizer.RGB_to_OD(source_image)
        OD_flat = OD.reshape(-1, 3)

        C = np.linalg.lstsq(stain_matrix_source.T, OD_flat.T, rcond=None)[0].T

        OD_normalized = C @ self.stain_matrix_target.T
        normalized = MacenkoNormalizer.OD_to_RGB(OD_normalized.reshape(source_image.shape))

        return normalized


# ============= Helper Functions (保持原样) =============

def normalize_tile(tile_image, normalizer, tissue_threshold=0.1):
    """Normalize a single tile if it contains enough tissue"""
    gray = cv2.cvtColor(tile_image, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < 235
    tissue_fraction = np.sum(tissue_mask) / tissue_mask.size

    if tissue_fraction < tissue_threshold:
        return tile_image

    try:
        normalized = normalizer.transform(tile_image)
        return normalized
    except Exception as e:
        logger.warning(f"Normalization failed for tile: {e}")
        return tile_image


def create_target_image_from_tiles(wsi, num_tiles=10, tile_size=1024, quick_test=False):
    """Create a representative target image by sampling tiles from WSI"""
    if quick_test:
        num_tiles = min(num_tiles, 5)

    wsi_dims = wsi.info.slide_dimensions
    tiles = []

    for _ in range(num_tiles * 2):
        if len(tiles) >= num_tiles:
            break

        x = np.random.randint(0, max(1, wsi_dims[0] - tile_size))
        y = np.random.randint(0, max(1, wsi_dims[1] - tile_size))

        tile = wsi.read_region(
            location=(x, y),
            level=0,
            size=(tile_size, tile_size)
        )

        if hasattr(tile, 'convert'):
            tile = np.array(tile.convert('RGB'))

        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < 235
        tissue_fraction = np.sum(tissue_mask) / tissue_mask.size

        if tissue_fraction > 0.5:
            tiles.append(tile)

    if not tiles:
        raise ValueError("Could not find tiles with sufficient tissue")

    n_cols = int(np.ceil(np.sqrt(len(tiles))))
    n_rows = int(np.ceil(len(tiles) / n_cols))

    composite = np.zeros((n_rows * tile_size, n_cols * tile_size, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        row = idx // n_cols
        col = idx % n_cols
        composite[row * tile_size:(row + 1) * tile_size,
        col * tile_size:(col + 1) * tile_size] = tile

    return composite


# ============= 修改的处理类：支持多组织区域 =============

class NormalizedTileProcessor:
    """Tile processor with stain normalization support and multi-tissue region detection"""

    def __init__(self, wsi_path, tile_size=5000, overlap=500, edge_threshold=0.005,
                 normalize=False, norm_method='macenko', target_image_path=None,
                 quick_test_config=None):
        self.wsi_path = wsi_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.edge_threshold = edge_threshold
        self.wsi = WSIReader.open(wsi_path)
        self.wsi_dims = self.wsi.info.slide_dimensions
        self.quick_test_config = quick_test_config or QuickTestConfig()

        if self.quick_test_config.enabled:
            self.tile_size = self.quick_test_config.tile_size

        self.normalize = normalize
        self.normalizer = None

        if self.normalize:
            self.setup_normalizer(norm_method, target_image_path)

    def setup_normalizer(self, norm_method, target_image_path):
        """Setup stain normalizer"""
        logger.info(f"Setting up {norm_method} normalizer...")

        if norm_method == 'reinhard':
            self.normalizer = ReinhardNormalizer()
        elif norm_method == 'macenko':
            self.normalizer = MacenkoNormalizer()
        elif norm_method == 'vahadane':
            self.normalizer = VahadaneNormalizer()
        else:
            raise ValueError(f"Unknown normalizer: {norm_method}")

        if target_image_path and os.path.exists(target_image_path):
            logger.info(f"Using target image: {target_image_path}")
            target_image = cv2.imread(target_image_path)
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        else:
            logger.info("Creating target image from WSI tiles...")
            target_image = create_target_image_from_tiles(
                self.wsi,
                num_tiles=20,
                quick_test=self.quick_test_config.enabled
            )

            target_save_path = os.path.join(
                os.path.dirname(self.wsi_path),
                f"normalization_target_{norm_method}.png"
            )
            cv2.imwrite(target_save_path, cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Target image saved to: {target_save_path}")

        self.normalizer.fit(target_image)
        logger.info("Normalizer fitted successfully")

    def detect_all_tissue_regions(self):
        """Detect ALL tissue regions, not just the largest"""
        logger.info("Detecting ALL tissue regions...")

        try:
            thumbnail = self.wsi.slide_thumbnail()
            gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)

            threshold = 240 if self.quick_test_config.enabled else 235
            _, tissue_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

            kernel = np.ones((10, 10), np.uint8)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("No tissue regions found")
                return []

            # Scale factors
            thumb_h, thumb_w = thumbnail.shape[:2]
            scale_x = self.wsi_dims[0] / thumb_w
            scale_y = self.wsi_dims[1] / thumb_h

            # Process all significant tissue regions
            tissue_regions = []
            min_area_threshold = 100000 / (scale_x * scale_y)  # Minimum area in thumbnail pixels

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_area_threshold:
                    logger.debug(f"Skipping small contour with area {area}")
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Convert to full resolution coordinates
                margin = 200 if self.quick_test_config.enabled else 500
                x_start = max(0, int(x * scale_x) - margin)
                y_start = max(0, int(y * scale_y) - margin)
                x_end = min(self.wsi_dims[0], int((x + w) * scale_x) + margin)
                y_end = min(self.wsi_dims[1], int((y + h) * scale_y) + margin)

                tissue_regions.append({
                    'index': i,
                    'bounds': (x_start, y_start, x_end, y_end),
                    'area': area * scale_x * scale_y
                })

                logger.info(f"  Tissue region {i}: bounds=({x_start},{y_start},{x_end},{y_end}), "
                            f"size={x_end - x_start}x{y_end - y_start}")

            # Sort by area (largest first)
            tissue_regions.sort(key=lambda r: r['area'], reverse=True)

            logger.info(f"Found {len(tissue_regions)} tissue regions")
            return tissue_regions

        except Exception as e:
            logger.error(f"Failed to detect tissue regions: {e}")
            # Fallback to processing entire slide
            return [{'index': 0, 'bounds': (0, 0, self.wsi_dims[0], self.wsi_dims[1])}]

    def detect_tissue_bounds(self):
        """Legacy method for compatibility - returns largest region bounds"""
        regions = self.detect_all_tissue_regions()
        if regions:
            return regions[0]['bounds']
        return None

    def generate_tiles_with_edge_handling(self):
        """Generate tiles ensuring coverage of all tissue regions"""
        tiles = []
        step = self.tile_size - self.overlap

        logger.info("Detecting tissue boundaries...")
        tissue_regions = self.detect_all_tissue_regions()

        if not tissue_regions:
            # Fallback to full image
            logger.warning("No tissue regions found, processing entire image")
            tissue_regions = [{'index': 0, 'bounds': (0, 0, self.wsi_dims[0], self.wsi_dims[1])}]

        # Generate tiles for each tissue region
        for region in tissue_regions:
            x_start, y_start, x_end, y_end = region['bounds']
            logger.info(f"Generating tiles for region {region['index']}: "
                        f"X({x_start}-{x_end}), Y({y_start}-{y_end})")

            region_tiles = 0
            for y in range(y_start, y_end, step):
                for x in range(x_start, x_end, step):
                    w = min(self.tile_size, self.wsi_dims[0] - x, x_end - x)
                    h = min(self.tile_size, self.wsi_dims[1] - y, y_end - y)

                    if w > 100 and h > 100:
                        tiles.append({
                            'id': len(tiles),
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'bounds': (x, y, x + w, y + h),
                            'region': region['index']
                        })
                        region_tiles += 1

            logger.info(f"  Generated {region_tiles} tiles for region {region['index']}")

        return tiles

    def check_tile_tissue(self, tile):
        """Check if tile contains tissue"""
        downsample = 32
        thumb_w = max(1, tile['w'] // downsample)
        thumb_h = max(1, tile['h'] // downsample)

        try:
            thumb = self.wsi.read_region(
                location=(tile['x'], tile['y']),
                level=0,
                size=(thumb_w, thumb_h)
            )

            if hasattr(thumb, 'convert'):
                thumb = np.array(thumb.convert('RGB'))

            if thumb.size == 0:
                return False

            gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
            tissue_mask = gray < 235
            tissue_ratio = np.sum(tissue_mask) / tissue_mask.size

            threshold = 0.001 if self.quick_test_config.enabled else self.edge_threshold
            return tissue_ratio > threshold

        except Exception as e:
            logger.debug(f"Error checking tile {tile['id']}: {e}")
            return False


# ============= 可视化函数（保持原样） =============

def create_comprehensive_visualization(wsi, predictions, tiles, output_dir, quick_test_config, wsi_name):
    """Create enhanced comprehensive visualization with better visibility"""

    logger.info("Creating enhanced comprehensive visualization...")

    # Get thumbnail
    thumbnail = wsi.slide_thumbnail()
    thumb_h, thumb_w = thumbnail.shape[:2]
    wsi_dims = wsi.info.slide_dimensions

    # Scale factors
    scale_x = thumb_w / wsi_dims[0]
    scale_y = thumb_h / wsi_dims[1]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1. Original WSI thumbnail
    axes[0, 0].imshow(thumbnail)
    axes[0, 0].set_title("Original WSI", fontsize=14)
    axes[0, 0].axis('off')

    # 2. Enhanced nucleus contour overlay
    overlay = thumbnail.copy()

    # Create a separate contour layer for better visibility
    contour_layer = np.zeros_like(thumbnail)

    # Draw nucleus contours with enhanced visibility
    sample_rate = quick_test_config.visualization_sample_rate
    thickness = quick_test_config.contour_thickness

    logger.info(f"Drawing {len(predictions)} nuclei (showing every {sample_rate}th)")

    for idx, (nuc_id, nuc_info) in enumerate(predictions.items()):
        if idx % sample_rate == 0:
            if idx % (sample_rate * 100) == 0:
                logger.info(f"Drawing progress: {idx}/{len(predictions)}")

            nuc_type = nuc_info.get('type', 0)
            color = COLOR_DICT[nuc_type][1]

            # Scale contour to thumbnail coordinates
            contour = nuc_info['contour']
            scaled_contour = contour.copy()
            scaled_contour[:, 0] = contour[:, 0] * scale_x
            scaled_contour[:, 1] = contour[:, 1] * scale_y
            scaled_contour = scaled_contour.astype(np.int32)

            # Draw filled contour for better visibility
            cv2.drawContours(contour_layer, [scaled_contour], -1, color[::-1], -1)

            # Draw outline
            cv2.drawContours(overlay, [scaled_contour], -1, color[::-1], thickness)

    # Blend contour layer with original for better visibility
    overlay = cv2.addWeighted(thumbnail, 0.7, contour_layer, 0.3, 0)

    # Add nucleus outlines
    for idx, (nuc_id, nuc_info) in enumerate(predictions.items()):
        if idx % sample_rate == 0:
            nuc_type = nuc_info.get('type', 0)
            color = COLOR_DICT[nuc_type][1]

            contour = nuc_info['contour']
            scaled_contour = contour.copy()
            scaled_contour[:, 0] = contour[:, 0] * scale_x
            scaled_contour[:, 1] = contour[:, 1] * scale_y
            scaled_contour = scaled_contour.astype(np.int32)

            cv2.drawContours(overlay, [scaled_contour], -1, color[::-1], thickness)

    axes[0, 1].imshow(overlay)
    mode_text = " (Quick Mode)" if quick_test_config.enabled else ""
    axes[0, 1].set_title(f"Nucleus Contour Overlay ({len(predictions)} nuclei){mode_text}", fontsize=14)
    axes[0, 1].axis('off')

    # Save high-resolution overlay
    overlay_path = os.path.join(output_dir, f"nuclei_overlay_full_{wsi_name}.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"High-resolution overlay saved to: {overlay_path}")

    # 3. Tile grid display
    grid_overlay = thumbnail.copy()
    for tile in tiles:
        x1 = int(tile['x'] * scale_x)
        y1 = int(tile['y'] * scale_y)
        x2 = int((tile['x'] + tile['w']) * scale_x)
        y2 = int((tile['y'] + tile['h']) * scale_y)

        cv2.rectangle(grid_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    axes[1, 0].imshow(grid_overlay)
    axes[1, 0].set_title(f"Tile Grid ({len(tiles)} tiles)", fontsize=14)
    axes[1, 0].axis('off')

    # 4. Enhanced nucleus density heatmap
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    for nuc_info in predictions.values():
        cent = nuc_info['centroid']
        x = int(cent[0] * scale_x)
        y = int(cent[1] * scale_y)

        if 0 <= x < thumb_w and 0 <= y < thumb_h:
            cv2.circle(heatmap, (x, y), 12, 1, -1)  # Larger circles for better visibility

    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

    axes[1, 1].imshow(thumbnail)
    if heatmap.max() > 0:
        im = axes[1, 1].imshow(heatmap, cmap='hot', alpha=0.6)
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_title("Nucleus Density Heatmap", fontsize=14)
    axes[1, 1].axis('off')

    # Add legend for nucleus types
    legend_elements = []
    for type_id, (type_name, color) in COLOR_DICT.items():
        count = sum(1 for n in predictions.values() if n.get('type', 0) == type_id)
        if count > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=np.array(color) / 255,
                                              markersize=10, label=f'{type_name}: {count}'))

    if legend_elements:
        axes[0, 1].legend(handles=legend_elements, loc='upper right',
                          bbox_to_anchor=(1.1, 1.0), fontsize=10)

    plt.suptitle(f"WSI Analysis Results - {wsi_name}{mode_text}", fontsize=16)
    plt.tight_layout()

    # Save as both PNG and return figure for PDF
    png_path = os.path.join(output_dir, f"comprehensive_visualization_{wsi_name}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    return fig


def extract_nucleus_patches(wsi, predictions, output_dir, quick_test_config, wsi_name):
    """Extract nucleus patches with enhanced visualization"""

    logger.info("Extracting nucleus patches...")

    num_samples = quick_test_config.max_patches

    # Group nuclei by type
    nuclei_by_type = {}
    for nuc_id, nuc_info in predictions.items():
        nuc_type = nuc_info.get('type', 0)
        if nuc_type not in nuclei_by_type:
            nuclei_by_type[nuc_type] = []
        nuclei_by_type[nuc_type].append((nuc_id, nuc_info))

    # Create patches directory
    patches_dir = os.path.join(output_dir, f"nucleus_patches_{wsi_name}")
    os.makedirs(patches_dir, exist_ok=True)

    patch_size = 256
    half_size = patch_size // 2

    samples_collected = []

    # Select samples from each type
    for nuc_type, nuclei_list in nuclei_by_type.items():
        if not nuclei_list:
            continue

        num_type_samples = 1 if quick_test_config.enabled else min(3, len(nuclei_list))

        if len(nuclei_list) > num_type_samples:
            selected_indices = random.sample(range(len(nuclei_list)), num_type_samples)
        else:
            selected_indices = range(len(nuclei_list))

        for idx in selected_indices:
            if len(samples_collected) >= num_samples:
                break

            nuc_id, nuc_info = nuclei_list[idx]
            cent = nuc_info['centroid'].astype(int)

            try:
                # Extract original patch
                patch = wsi.read_region(
                    location=(cent[0] - half_size, cent[1] - half_size),
                    level=0,
                    size=(patch_size, patch_size)
                )

                if hasattr(patch, 'convert'):
                    patch = np.array(patch.convert('RGB'))

                # Check tissue content
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                tissue_percent = np.sum(gray < 235) / (patch_size * patch_size) * 100

                if tissue_percent < 10:
                    continue

                # Create overlay with enhanced contours
                overlay = patch.copy()

                # Adjust contour coordinates to patch space
                contour = nuc_info['contour'] - np.array([cent[0] - half_size, cent[1] - half_size])
                contour = contour.astype(np.int32)

                # Draw contour with enhanced visibility
                color = COLOR_DICT[nuc_type][1]
                cv2.drawContours(overlay, [contour], -1, color[::-1], 4)  # Thicker contour

                # Add center cross
                cv2.line(overlay, (half_size - 15, half_size), (half_size + 15, half_size), color[::-1], 3)
                cv2.line(overlay, (half_size, half_size - 15), (half_size, half_size + 15), color[::-1], 3)

                # Add to samples
                samples_collected.append({
                    'original': patch,
                    'overlay': overlay,
                    'type': nuc_type,
                    'type_name': COLOR_DICT[nuc_type][0],
                    'nuc_id': nuc_id
                })

                # Save individual patch
                patch_filename = f"nucleus_{nuc_id}_type{nuc_type}_{COLOR_DICT[nuc_type][0].replace(' ', '_')}.png"
                cv2.imwrite(
                    os.path.join(patches_dir, patch_filename),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

            except Exception as e:
                logger.debug(f"Failed to extract patch (nucleus {nuc_id}): {e}")
                continue

    # Create grid visualization
    if samples_collected:
        n_samples = len(samples_collected)
        n_rows = min(4, (n_samples + 3) // 4)
        n_cols = min(4, n_samples)

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 3))

        # Ensure axes is 2D array
        if n_rows == 1 and n_cols * 2 > 1:
            axes = axes.reshape(1, -1)
        elif n_rows > 1 and n_cols * 2 == 1:
            axes = axes.reshape(-1, 1)
        elif n_rows == 1 and n_cols * 2 == 1:
            axes = np.array([[axes]])

        # Fill grid
        for i, sample in enumerate(samples_collected):
            if i >= n_rows * n_cols:
                break

            row = i // n_cols
            col = i % n_cols

            # Original patch
            ax1 = axes[row, col * 2]
            ax1.imshow(sample['original'])
            ax1.set_title("Original", fontsize=10)
            ax1.axis('off')

            # Overlay patch
            ax2 = axes[row, col * 2 + 1]
            ax2.imshow(sample['overlay'])
            ax2.set_title(f"{sample['type_name']}", fontsize=10,
                          color=np.array(COLOR_DICT[sample['type']][1]) / 255)
            ax2.axis('off')

        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col * 2 < axes.shape[1]:
                axes[row, col * 2].axis('off')
                if col * 2 + 1 < axes.shape[1]:
                    axes[row, col * 2 + 1].axis('off')

        mode_text = " (Quick Mode)" if quick_test_config.enabled else ""
        plt.suptitle(f"Nucleus Patches Sample - {wsi_name}{mode_text}", fontsize=16)
        plt.tight_layout()

        # Save and return figure
        png_path = os.path.join(output_dir, f"nucleus_patches_grid_{wsi_name}.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight')

        logger.info(f"Successfully extracted {len(samples_collected)} nucleus patches")

        return fig
    else:
        logger.warning("No valid nucleus patches could be extracted")
        plt.close()
        return None


def create_pdf_report(output_dir, wsi_name, figures):
    """Create a comprehensive PDF report with all visualizations"""

    pdf_path = os.path.join(output_dir, f"analysis_report_{wsi_name}.pdf")

    with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
        # Add each figure to PDF
        for fig in figures:
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Add metadata
        d = pdf.infodict()
        d['Title'] = f'WSI Analysis Report - {wsi_name}'
        d['Author'] = 'TIAToolbox Analysis Pipeline'
        d['Subject'] = 'Nucleus Instance Segmentation Results'
        d['Keywords'] = 'Pathology, H&E, Nucleus Segmentation'
        d['CreationDate'] = datetime.now()

    logger.info(f"PDF report saved to: {pdf_path}")


# ============= 主处理函数（仅修改了相关部分） =============

def process_single_wsi(wsi_path, output_base_dir, tile_size, overlap, batch_size,
                       gpu_id, normalize, norm_method, target_image_path, quick_test):
    """Process a single WSI file"""

    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    output_dir = os.path.join(output_base_dir, wsi_name)

    # 记录处理开始
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting to process WSI: {wsi_name}")
    logger.info(f"Full path: {wsi_path}")
    logger.info(f"File size: {os.path.getsize(wsi_path) / 1024 / 1024 / 1024:.2f} GB")
    logger.info(f"Output will be saved to: {output_dir}")
    logger.info(f"{'=' * 60}")

    # 设置GPU
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 获取实际使用的GPU
    actual_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing WSI: {wsi_name}")
    logger.info(f"Using GPU: {actual_gpu}")
    logger.info(f"{'=' * 60}")

    try:
        # Create quick test config
        quick_test_config = QuickTestConfig(enabled=quick_test)

        # Override parameters if quick test
        if quick_test:
            tile_size = quick_test_config.tile_size
            batch_size = quick_test_config.batch_size

        os.makedirs(output_dir, exist_ok=True)
        segmentation_dir = os.path.join(output_dir, "segmentation")
        os.makedirs(segmentation_dir, exist_ok=True)

        # Initialize processor
        processor = NormalizedTileProcessor(
            wsi_path, tile_size, overlap,
            normalize=normalize,
            norm_method=norm_method,
            target_image_path=target_image_path,
            quick_test_config=quick_test_config
        )

        # Generate tiles（这里会使用新的多区域检测）
        logger.info("\nGenerating tiles...")
        tiles = processor.generate_tiles_with_edge_handling()
        logger.info(f"Total tiles generated: {len(tiles)}")

        # 显示每个区域的tile数量
        region_counts = {}
        for tile in tiles:
            region = tile.get('region', 0)
            region_counts[region] = region_counts.get(region, 0) + 1

        for region, count in sorted(region_counts.items()):
            logger.info(f"  Region {region}: {count} tiles")

        # Filter tiles with tissue
        logger.info("\nFiltering tiles with tissue...")
        valid_tiles = []
        for i, tile in enumerate(tiles):
            if i % 10 == 0:
                logger.info(f"Checking progress: {i}/{len(tiles)}")

            if processor.check_tile_tissue(tile):
                valid_tiles.append(tile)

        # Limit tiles in quick test mode
        if quick_test_config.enabled and quick_test_config.max_tiles:
            if len(valid_tiles) > quick_test_config.max_tiles:
                logger.info(f"Quick test mode: limiting to {quick_test_config.max_tiles} tiles")
                # 从每个区域选择一些tiles
                region_tiles = {}
                for tile in valid_tiles:
                    region = tile.get('region', 0)
                    if region not in region_tiles:
                        region_tiles[region] = []
                    region_tiles[region].append(tile)

                valid_tiles = []
                tiles_per_region = max(1, quick_test_config.max_tiles // len(region_tiles))
                for region, tiles_list in region_tiles.items():
                    valid_tiles.extend(tiles_list[:tiles_per_region])

                valid_tiles = valid_tiles[:quick_test_config.max_tiles]

        logger.info(f"Tiles to process: {len(valid_tiles)}")

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        # Initialize segmentor
        logger.info("\nInitializing segmentor...")
        segmentor = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-pannuke",
            num_loader_workers=2,
            num_postproc_workers=2,
            batch_size=batch_size,
            verbose=False,
        )

        # Process tiles
        all_predictions = {}
        nucleus_counter = 0
        file_map = {}

        if normalize:
            norm_examples_dir = os.path.join(output_dir, "normalization_examples")
            os.makedirs(norm_examples_dir, exist_ok=True)
            num_norm_examples = quick_test_config.num_norm_examples

        logger.info(f"\nProcessing {len(valid_tiles)} tiles...")
        start_time = time.time()

        for tile_idx, tile in enumerate(valid_tiles):
            logger.info(f"\nProcessing tile {tile_idx + 1}/{len(valid_tiles)} "
                        f"(ID: {tile['id']}, Region: {tile.get('region', 0)})...")

            try:
                # Extract tile
                tile_img = processor.wsi.read_region(
                    location=(tile['x'], tile['y']),
                    level=0,
                    size=(tile['w'], tile['h'])
                )

                if hasattr(tile_img, 'convert'):
                    tile_img = np.array(tile_img.convert('RGB'))

                # Save normalization examples
                if normalize and tile_idx < num_norm_examples:
                    orig_path = os.path.join(norm_examples_dir, f"tile_{tile['id']:04d}_original.png")
                    cv2.imwrite(orig_path, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))

                # Apply normalization if enabled
                if processor.normalize and processor.normalizer:
                    tile_img_normalized = normalize_tile(tile_img, processor.normalizer)

                    if tile_idx < num_norm_examples:
                        norm_path = os.path.join(norm_examples_dir, f"tile_{tile['id']:04d}_normalized.png")
                        cv2.imwrite(norm_path, cv2.cvtColor(tile_img_normalized, cv2.COLOR_RGB2BGR))

                    tile_img = tile_img_normalized

                # Save tile for processing
                tile_path = os.path.join(temp_dir, f"tile_{tile['id']:06d}.png")
                cv2.imwrite(tile_path, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))

                # Process tile
                tile_output_dir = os.path.join(temp_dir, f"output_{tile['id']:06d}")

                try:
                    output = segmentor.predict(
                        [tile_path],
                        save_dir=tile_output_dir,
                        mode="tile",
                        device="cuda",
                    )

                    # Load results
                    result_file = os.path.join(tile_output_dir, "0.dat")
                    if os.path.exists(result_file):
                        predictions = joblib.load(result_file)

                        # Process predictions
                        for nuc_id, nuc_info in predictions.items():
                            cent = nuc_info['centroid']

                            # Simple deduplication
                            margin = 50
                            if (cent[0] < margin and tile['x'] > 0) or \
                                    (cent[1] < margin and tile['y'] > 0) or \
                                    (cent[0] > tile['w'] - margin and tile['x'] + tile['w'] < processor.wsi_dims[0]) or \
                                    (cent[1] > tile['h'] - margin and tile['y'] + tile['h'] < processor.wsi_dims[1]):
                                continue

                            # Adjust coordinates
                            new_id = nucleus_counter
                            adj_info = nuc_info.copy()
                            adj_info['centroid'] = nuc_info['centroid'] + np.array([tile['x'], tile['y']])
                            adj_info['contour'] = nuc_info['contour'] + np.array([tile['x'], tile['y']])
                            adj_info['box'] = [
                                nuc_info['box'][0] + tile['x'],
                                nuc_info['box'][1] + tile['y'],
                                nuc_info['box'][2] + tile['x'],
                                nuc_info['box'][3] + tile['y']
                            ]

                            all_predictions[new_id] = adj_info
                            nucleus_counter += 1

                        logger.info(f"  Tile {tile['id']}: detected {len(predictions)} nuclei")

                except Exception as e:
                    logger.error(f"  Error processing tile: {e}")

                # Cleanup
                if os.path.exists(tile_path):
                    os.remove(tile_path)
                if os.path.exists(tile_output_dir):
                    shutil.rmtree(tile_output_dir)

            except Exception as e:
                logger.error(f"Failed to process tile {tile['id']}: {e}")
                continue

            # Show ETA
            if tile_idx > 0:
                elapsed = time.time() - start_time
                avg_time_per_tile = elapsed / (tile_idx + 1)
                remaining_tiles = len(valid_tiles) - tile_idx - 1
                eta_seconds = remaining_tiles * avg_time_per_tile
                eta_minutes = eta_seconds / 60
                logger.info(f"  ETA: {eta_minutes:.1f} minutes")

        logger.info(f"\nTotal nuclei detected: {len(all_predictions)}")

        # Save results
        output_file = os.path.join(segmentation_dir, "0.dat")
        joblib.dump(all_predictions, output_file)

        file_map[0] = os.path.basename(wsi_path)
        file_map_path = os.path.join(segmentation_dir, "file_map.dat")
        joblib.dump(file_map, file_map_path)

        # Create visualizations and collect figures for PDF
        figures = []

        logger.info("\nCreating visualizations...")
        fig1 = create_comprehensive_visualization(processor.wsi, all_predictions, valid_tiles,
                                                  output_dir, quick_test_config, wsi_name)
        figures.append(fig1)

        logger.info("\nExtracting nucleus patches...")
        fig2 = extract_nucleus_patches(processor.wsi, all_predictions, output_dir,
                                       quick_test_config, wsi_name)
        if fig2 is not None:
            figures.append(fig2)

        # Generate analysis report
        generate_analysis_report(processor.wsi, all_predictions, output_dir, wsi_path,
                                 normalize, norm_method, quick_test, wsi_name)

        # Create PDF report
        create_pdf_report(output_dir, wsi_name, figures)

        # Cleanup
        shutil.rmtree(temp_dir)

        total_time = time.time() - start_time
        logger.info(f"\nProcessing complete! Total time: {total_time / 60:.1f} minutes")

        if quick_test:
            logger.info("\n*** Quick test completed ***")
            logger.info("If results look good, run without --quick-test for full processing")

        return wsi_name, True, len(all_predictions)

    except Exception as e:
        logger.error(f"Failed to process {wsi_name}: {e}")
        return wsi_name, False, 0


def generate_analysis_report(wsi, predictions, output_dir, wsi_path,
                             normalize, norm_method, quick_test, wsi_name):
    """Generate analysis report in English"""

    if not predictions:
        return

    all_x = [p['centroid'][0] for p in predictions.values()]
    all_y = [p['centroid'][1] for p in predictions.values()]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    wsi_dims = wsi.info.slide_dimensions
    x_coverage = (x_max - x_min) / wsi_dims[0] * 100
    y_coverage = (y_max - y_min) / wsi_dims[1] * 100

    type_counts = {}
    for nuc_info in predictions.values():
        nuc_type = nuc_info.get('type', 0)
        type_name = COLOR_DICT[nuc_type][0]
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    report_path = os.path.join(output_dir, f"analysis_report_{wsi_name}.txt")
    with open(report_path, 'w') as f:
        f.write("WSI Processing Report\n")
        f.write("=" * 50 + "\n")
        if quick_test:
            f.write("*** QUICK TEST MODE ***\n")
        f.write(f"WSI File: {wsi_path}\n")
        f.write(f"WSI Dimensions: {wsi_dims[0]} x {wsi_dims[1]}\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nStain Normalization:\n")
        f.write(f"  Enabled: {'Yes' if normalize else 'No'}\n")
        if normalize:
            f.write(f"  Method: {norm_method}\n")
        f.write(f"\nDetection Results:\n")
        f.write(f"Total Nuclei: {len(predictions)}\n")
        if quick_test:
            f.write(f"Coverage: X={x_coverage:.1f}%, Y={y_coverage:.1f}% (partial tiles only)\n")
        else:
            f.write(f"Coverage: X={x_coverage:.1f}%, Y={y_coverage:.1f}%\n")
        f.write(f"\nNucleus Type Distribution:\n")

        for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(predictions) * 100
            f.write(f"  {type_name}: {count} ({percentage:.1f}%)\n")


# 批处理函数（保持原样）
def process_batch_wsi(wsi_files, output_base_dir, **kwargs):
    """Process multiple WSI files using multiple GPUs"""

    num_gpus = 4  # You have 4×4090 GPUs
    results = []

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Batch Processing {len(wsi_files)} WSI files")
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(f"{'=' * 60}")

    # Process WSIs in parallel using different GPUs
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}

        for idx, wsi_path in enumerate(wsi_files):
            gpu_id = idx % num_gpus
            future = executor.submit(
                process_single_wsi,
                wsi_path,
                output_base_dir,
                kwargs.get('tile_size', 5000),
                kwargs.get('overlap', 500),
                kwargs.get('batch_size', 512),
                gpu_id,
                kwargs.get('normalize', False),
                kwargs.get('norm_method', 'macenko'),
                kwargs.get('target_image_path', None),
                kwargs.get('quick_test', False)
            )
            futures[future] = wsi_path

        # Collect results
        for future in as_completed(futures):
            wsi_path = futures[future]
            try:
                wsi_name, success, num_nuclei = future.result()
                results.append({
                    'wsi': wsi_name,
                    'success': success,
                    'nuclei_count': num_nuclei,
                    'path': wsi_path
                })
                logger.info(f"Completed: {wsi_name} - {'Success' if success else 'Failed'} - {num_nuclei} nuclei")
            except Exception as e:
                logger.error(f"Failed to process {wsi_path}: {e}")
                results.append({
                    'wsi': os.path.basename(wsi_path),
                    'success': False,
                    'nuclei_count': 0,
                    'path': wsi_path
                })

    # Generate summary report
    generate_batch_summary(results, output_base_dir)

    return results


def generate_batch_summary(results, output_base_dir):
    """Generate summary report for batch processing"""

    summary_path = os.path.join(output_base_dir, "batch_processing_summary.txt")

    with open(summary_path, 'w') as f:
        f.write("Batch Processing Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total WSI Files: {len(results)}\n")
        f.write(f"Successful: {sum(1 for r in results if r['success'])}\n")
        f.write(f"Failed: {sum(1 for r in results if not r['success'])}\n")
        f.write("\nDetailed Results:\n")
        f.write("-" * 60 + "\n")

        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            f.write(f"{result['wsi']:<30} {status:<10} {result['nuclei_count']:>10} nuclei\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total Nuclei Detected: {sum(r['nuclei_count'] for r in results)}\n")

    logger.info(f"Batch summary saved to: {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch WSI Processing with Multi-Tissue Support')
    parser.add_argument('--input-dir', required=True, help='Directory containing WSI files')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--tile-size', type=int, default=5000, help='Tile size')
    parser.add_argument('--overlap', type=int, default=500, help='Tile overlap')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')

    parser.add_argument('--normalize', action='store_true', help='Enable stain normalization')
    parser.add_argument('--norm-method', choices=['reinhard', 'macenko', 'vahadane'],
                        default='macenko', help='Normalization method')
    parser.add_argument('--target-image', help='Target image path (optional)')

    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode: process fewer tiles for testing')
    parser.add_argument('--single-wsi', help='Process only a single WSI file')

    args = parser.parse_args()

    # 初始化日志系统
    log_output_dir = "/results"
    if not os.path.exists(log_output_dir):
        log_output_dir = args.output

    setup_logging(log_output_dir)

    # 记录运行参数
    logger.info("=" * 80)
    logger.info("WSI Batch Processing Started (Multi-Tissue Support)")
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info("=" * 80)
    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Find all WSI files
    if args.single_wsi:
        if not os.path.exists(args.single_wsi):
            logger.error(f"WSI file not found: {args.single_wsi}")
            return
        wsi_files = [args.single_wsi]
    else:
        wsi_patterns = ['*.svs', '*.ndpi', '*.tiff', '*.tif']
        wsi_files = []
        for pattern in wsi_patterns:
            wsi_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

        if not wsi_files:
            logger.error(f"No WSI files found in: {args.input_dir}")
            return

    logger.info(f"Found {len(wsi_files)} WSI files")

    # Process WSIs
    if len(wsi_files) == 1:
        # Single WSI - process directly
        process_single_wsi(
            wsi_files[0],
            args.output,
            args.tile_size,
            args.overlap,
            args.batch_size,
            0,  # Use GPU 0
            args.normalize,
            args.norm_method,
            args.target_image,
            args.quick_test
        )
    else:
        # Multiple WSIs - batch process
        process_batch_wsi(
            wsi_files,
            args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            normalize=args.normalize,
            norm_method=args.norm_method,
            target_image_path=args.target_image,
            quick_test=args.quick_test
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("All processing complete!")
    logger.info(f"Results saved in: {args.output}")

    # 添加运行总结
    logger.info(f"Total processing time: {datetime.now()}")
    logger.info(f"Log files saved in: {log_output_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
