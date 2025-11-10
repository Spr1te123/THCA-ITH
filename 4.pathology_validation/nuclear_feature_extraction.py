import shutil  # 用于删除目录
import os
import numpy as np
import cv2
import joblib
from skimage import measure, morphology
from scipy import ndimage, stats
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tiatoolbox.wsicore.wsireader import WSIReader
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell type definitions (from wsi_processing_multi_tissue.py)
CELL_TYPES = {
    1: "Neoplastic_Epithelial",  # 肿瘤上皮细胞
    2: "Inflammatory",  # 炎症细胞
    3: "Connective"  # 结缔组织/间质细胞
}


class StainNormalizer:
    """Base class for stain normalization (from original script)"""

    def __init__(self):
        self.target_means = None
        self.target_stds = None


class ReinhardNormalizer(StainNormalizer):
    """Reinhard color normalization in LAB color space"""

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


class NuclearFeatureExtractor:
    """Extract morphological and textural features from segmented nuclei"""

    def __init__(self, normalizer=None):
        self.normalizer = normalizer
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        """Get all feature names"""
        morph_features = [
            'area', 'area_bbox', 'eccentricity', 'circularity',
            'elongation', 'extent', 'major_axis_length', 'minor_axis_length',
            'perimeter', 'solidity', 'curve_mean', 'curve_max',
            'curve_min', 'curve_std'
        ]

        intensity_features = [
            'intensity_mean', 'intensity_std', 'intensity_max', 'intensity_min',
            'intensity_skewness', 'intensity_kurtosis', 'intensity_entropy'
        ]

        texture_features = [
            'contrast', 'homogeneity', 'energy', 'correlation'
        ]

        return morph_features + intensity_features + texture_features

    def extract_morphological_features(self, contour, mask):
        """Extract morphological features from a nucleus"""
        features = {}

        try:
            # Basic shape properties
            props = measure.regionprops(mask.astype(int))[0]

            # Area features
            features['area'] = props.area
            features['area_bbox'] = props.bbox_area

            # Shape features
            features['eccentricity'] = props.eccentricity
            features['perimeter'] = props.perimeter

            # Circularity
            if features['perimeter'] > 0:
                features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
            else:
                features['circularity'] = 0

            # Elongation
            features['major_axis_length'] = props.major_axis_length
            features['minor_axis_length'] = props.minor_axis_length
            if props.minor_axis_length > 0:
                features['elongation'] = props.major_axis_length / props.minor_axis_length
            else:
                features['elongation'] = 1

            # Extent and Solidity
            features['extent'] = props.extent
            features['solidity'] = props.solidity

            # Contour curvature features
            curvatures = self._compute_curvature(contour)
            features['curve_mean'] = np.mean(curvatures)
            features['curve_max'] = np.max(curvatures)
            features['curve_min'] = np.min(curvatures)
            features['curve_std'] = np.std(curvatures)

        except Exception as e:
            logger.warning(f"Error in morphological feature extraction: {e}")
            # Return default values
            for name in ['area', 'area_bbox', 'eccentricity', 'circularity',
                         'elongation', 'extent', 'major_axis_length', 'minor_axis_length',
                         'perimeter', 'solidity', 'curve_mean', 'curve_max',
                         'curve_min', 'curve_std']:
                features[name] = 0

        return features

    def _compute_curvature(self, contour, k=5):
        """Compute curvature along the contour"""
        n_points = len(contour)
        if n_points < 2 * k + 1:
            return np.zeros(n_points)

        curvatures = []
        for i in range(n_points):
            p_prev = contour[(i - k) % n_points]
            p_curr = contour[i]
            p_next = contour[(i + k) % n_points]

            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]

            cross = dx1 * dy2 - dy1 * dx2
            norm = np.sqrt((dx1 ** 2 + dy1 ** 2) * (dx2 ** 2 + dy2 ** 2))

            if norm > 0:
                curvatures.append(cross / norm)
            else:
                curvatures.append(0)

        return np.array(curvatures)

    def extract_intensity_features(self, nucleus_patch, mask):
        """Extract intensity-based features from nucleus patch"""
        features = {}

        try:
            # Convert to grayscale
            if len(nucleus_patch.shape) == 3:
                gray_patch = cv2.cvtColor(nucleus_patch, cv2.COLOR_RGB2GRAY)
            else:
                gray_patch = nucleus_patch

            # Get only nucleus pixels
            nucleus_pixels = gray_patch[mask > 0]

            if len(nucleus_pixels) > 0:
                # Basic intensity statistics
                features['intensity_mean'] = np.mean(nucleus_pixels)
                features['intensity_std'] = np.std(nucleus_pixels)
                features['intensity_max'] = np.max(nucleus_pixels)
                features['intensity_min'] = np.min(nucleus_pixels)

                # Additional statistics
                features['intensity_skewness'] = stats.skew(nucleus_pixels)
                features['intensity_kurtosis'] = stats.kurtosis(nucleus_pixels)

                # Entropy
                hist, _ = np.histogram(nucleus_pixels, bins=256, range=(0, 256), density=True)
                hist = hist[hist > 0]
                features['intensity_entropy'] = -np.sum(hist * np.log2(hist))
            else:
                raise ValueError("No nucleus pixels found")

        except Exception as e:
            logger.warning(f"Error in intensity feature extraction: {e}")
            # Return default values
            for name in ['intensity_mean', 'intensity_std', 'intensity_max',
                         'intensity_min', 'intensity_skewness', 'intensity_kurtosis',
                         'intensity_entropy']:
                features[name] = 0

        return features

    def extract_texture_features(self, nucleus_patch, mask):
        """Extract GLCM texture features"""
        features = {}

        try:
            from skimage.feature import graycomatrix, graycoprops

            # Convert to grayscale
            if len(nucleus_patch.shape) == 3:
                gray_patch = cv2.cvtColor(nucleus_patch, cv2.COLOR_RGB2GRAY)
            else:
                gray_patch = nucleus_patch

            # Apply mask and normalize
            masked_patch = gray_patch.copy()
            masked_patch[mask == 0] = 0

            # Normalize to 8 levels
            gray_norm = (masked_patch / 32).astype(np.uint8)

            # Compute GLCM
            glcm = graycomatrix(gray_norm, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=8, symmetric=True, normed=True)

            # Extract properties
            features['contrast'] = graycoprops(glcm, 'contrast')[0, :].mean()
            features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, :].mean()
            features['energy'] = graycoprops(glcm, 'energy')[0, :].mean()
            features['correlation'] = graycoprops(glcm, 'correlation')[0, :].mean()

        except Exception as e:
            logger.warning(f"Error in texture feature extraction: {e}")
            for name in ['contrast', 'homogeneity', 'energy', 'correlation']:
                features[name] = 0

        return features


def process_wsi_with_normalization(wsi_path, nuclei_data_path, output_dir,
                                   norm_method='macenko', target_image_path=None):
    """
    Process WSI with stain normalization and extract features for specific cell types
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load nuclei predictions
    logger.info(f"Loading nuclei data from {nuclei_data_path}")
    nuclei_predictions = joblib.load(nuclei_data_path)

    # Initialize WSI reader
    logger.info(f"Opening WSI: {wsi_path}")
    wsi = WSIReader.open(wsi_path)

    # Setup normalizer
    logger.info(f"Setting up {norm_method} normalizer...")
    if norm_method == 'reinhard':
        normalizer = ReinhardNormalizer()
    elif norm_method == 'macenko':
        normalizer = MacenkoNormalizer()
    else:
        raise ValueError(f"Unknown normalizer: {norm_method}")

    # Create or load target image for normalization
    if target_image_path and os.path.exists(target_image_path):
        logger.info(f"Using target image: {target_image_path}")
        target_image = cv2.imread(target_image_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    else:
        logger.info("Creating target image from WSI...")
        target_image = create_target_image_from_wsi(wsi)
        # Save target image
        target_save_path = os.path.join(output_dir, f"normalization_target_{norm_method}.png")
        cv2.imwrite(target_save_path, cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Target image saved to: {target_save_path}")

    normalizer.fit(target_image)

    # Initialize feature extractor
    extractor = NuclearFeatureExtractor(normalizer)

    # Filter nuclei by cell type
    filtered_nuclei = {nuc_id: nuc_info for nuc_id, nuc_info in nuclei_predictions.items()
                       if nuc_info.get('type', -1) in CELL_TYPES.keys()}

    logger.info(f"Total nuclei: {len(nuclei_predictions)}")
    logger.info(f"Filtered nuclei (types 1,2,3): {len(filtered_nuclei)}")

    # Cell type distribution
    cell_type_counts = {}
    for nuc_info in filtered_nuclei.values():
        cell_type = nuc_info.get('type', -1)
        cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1

    for cell_type, count in sorted(cell_type_counts.items()):
        logger.info(f"  {CELL_TYPES[cell_type]}: {count} cells")

    # Extract features
    all_features = []
    patch_size = 64  # Size of patch around nucleus

    logger.info("Extracting features from nuclei...")
    for nuc_id, nuc_info in tqdm(filtered_nuclei.items(), desc="Processing nuclei"):
        try:
            # Get nucleus bounding box
            bbox = nuc_info['box']
            x_min, y_min, x_max, y_max = bbox

            # Expand bbox for context
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Read patch from WSI
            patch_x = max(0, center_x - patch_size // 2)
            patch_y = max(0, center_y - patch_size // 2)

            patch = wsi.read_region(
                location=(patch_x, patch_y),
                level=0,
                size=(patch_size, patch_size)
            )

            # Convert to numpy array
            if hasattr(patch, 'convert'):
                patch = np.array(patch.convert('RGB'))

            # Apply stain normalization
            if normalizer is not None:
                patch_normalized = normalizer.transform(patch)
            else:
                patch_normalized = patch

            # Adjust coordinates to patch space
            contour = nuc_info['contour'] - np.array([patch_x, patch_y])

            # Create mask
            mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            cv2.drawContours(mask, [contour.astype(np.int32)], -1, 1, -1)

            # Extract all features
            features = {}

            # Morphological features
            morph_features = extractor.extract_morphological_features(contour, mask)
            features.update(morph_features)

            # Intensity features
            intensity_features = extractor.extract_intensity_features(patch_normalized, mask)
            features.update(intensity_features)

            # Texture features
            texture_features = extractor.extract_texture_features(patch_normalized, mask)
            features.update(texture_features)

            # Add metadata
            features['nucleus_id'] = nuc_id
            features['nucleus_type'] = nuc_info.get('type', -1)
            features['cell_type_name'] = CELL_TYPES[nuc_info.get('type', -1)]
            features['centroid_x'] = nuc_info['centroid'][0]
            features['centroid_y'] = nuc_info['centroid'][1]

            all_features.append(features)

        except Exception as e:
            logger.warning(f"Failed to process nucleus {nuc_id}: {e}")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Save features
    output_path = os.path.join(output_dir, 'nuclear_features.csv')
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    # Calculate ITH scores
    ith_scores = calculate_cellular_ith(features_df)

    # Save ITH scores
    ith_df = pd.DataFrame([ith_scores])
    ith_path = os.path.join(output_dir, 'cellular_ith_scores.csv')
    ith_df.to_csv(ith_path, index=False)

    # Generate analysis report
    generate_analysis_report(features_df, ith_scores, output_dir)

    # Visualizations
    create_visualizations(features_df, output_dir)

    return features_df, ith_scores


def create_target_image_from_wsi(wsi, num_tiles=10, tile_size=1024):
    """Create target image for stain normalization"""
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

        # Check tissue content
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < 235
        tissue_fraction = np.sum(tissue_mask) / tissue_mask.size

        if tissue_fraction > 0.5:
            tiles.append(tile)

    if not tiles:
        raise ValueError("Could not find tiles with sufficient tissue")

    # Create composite image
    n_cols = int(np.ceil(np.sqrt(len(tiles))))
    n_rows = int(np.ceil(len(tiles) / n_cols))

    composite = np.zeros((n_rows * tile_size, n_cols * tile_size, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        row = idx // n_cols
        col = idx % n_cols
        composite[row * tile_size:(row + 1) * tile_size,
        col * tile_size:(col + 1) * tile_size] = tile

    return composite


def calculate_cellular_ith(features_df):
    """Calculate cellular morphological ITH for the three cell types"""
    ith_scores = {}

    feature_cols = [col for col in features_df.columns
                    if col not in ['nucleus_id', 'nucleus_type', 'cell_type_name',
                                   'centroid_x', 'centroid_y']]

    # Calculate ITH for each feature across all three cell types
    for feature in feature_cols:
        feature_ith_values = []

        for cell_type in CELL_TYPES.keys():
            cell_features = features_df[features_df['nucleus_type'] == cell_type][feature]

            if len(cell_features) > 1:
                ith = np.std(cell_features)
                feature_ith_values.append(ith)

        if feature_ith_values:
            ith_scores[f'{feature}_ith'] = np.mean(feature_ith_values)

    return ith_scores


def generate_analysis_report(features_df, ith_scores, output_dir):
    """Generate comprehensive analysis report"""
    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("Nuclear Feature Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Cell Type Distribution:\n")
        for cell_type, cell_name in CELL_TYPES.items():
            count = len(features_df[features_df['nucleus_type'] == cell_type])
            percentage = count / len(features_df) * 100
            f.write(f"  {cell_name}: {count} ({percentage:.1f}%)\n")

        f.write(f"\nTotal nuclei analyzed: {len(features_df)}\n")

        f.write("\nTop 10 Features with Highest ITH:\n")
        ith_sorted = sorted(ith_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in ith_sorted:
            f.write(f"  {feature}: {score:.4f}\n")

        # Key features from the paper
        f.write("\nKey Features (from research):\n")
        key_features = ['intensity_max_ith', 'curve_std_ith', 'curve_mean_ith']
        for feature in key_features:
            if feature in ith_scores:
                f.write(f"  {feature}: {ith_scores[feature]:.4f}\n")

def create_visualizations(features_df, output_dir):
    """Create comprehensive visualizations"""

    # 1. Feature distributions by cell type (similar to Figure 4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    key_features = ['intensity_max', 'curve_std', 'curve_mean', 'area']

    for idx, feature in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]

        # Create violin plots for each cell type
        data_for_plot = []
        labels = []
        colors = []

        for cell_type, cell_name in CELL_TYPES.items():
            cell_data = features_df[features_df['nucleus_type'] == cell_type][feature].values
            if len(cell_data) > 0:
                data_for_plot.append(cell_data)
                labels.append(cell_name)
                # Colors matching the paper
                color_map = {1: 'red', 2: 'yellow', 3: 'green'}
                colors.append(color_map[cell_type])

        parts = ax.violinplot(data_for_plot, positions=range(len(labels)),
                              widths=0.7, showmeans=True, showextrema=True)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} Distribution by Cell Type')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Morphological Features by Cell Type', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation heatmap
    feature_cols = [col for col in features_df.columns
                    if col not in ['nucleus_id', 'nucleus_type', 'cell_type_name',
                                   'centroid_x', 'centroid_y']]

    corr_matrix = features_df[feature_cols].corr()

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=False)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=300)
    plt.close()

    # 3. ITH scores visualization
    ith_df = pd.read_csv(os.path.join(output_dir, 'cellular_ith_scores.csv'))

    # Select top features
    ith_values = ith_df.iloc[0].sort_values(ascending=False)[:20]

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(ith_values)), ith_values.values)
    plt.xticks(range(len(ith_values)), ith_values.index, rotation=90, ha='right')
    plt.ylabel('ITH Score (Standard Deviation)')
    plt.title('Top 20 Features by Intratumoral Heterogeneity')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ith_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
# create_visualizations 函数在这里完整结束


def check_output_complete(output_dir):
    """Check if output directory contains all required files"""
    required_files = [
        'nuclear_features.csv',
        'cellular_ith_scores.csv',
        'analysis_report.txt',
        'feature_distributions.png',
        'feature_correlations.png',
        'ith_scores.png'
    ]

    if not os.path.exists(output_dir):
        return False

    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            return False

    # Optional: Check if files are not empty
    csv_files = ['nuclear_features.csv', 'cellular_ith_scores.csv']
    for csv_file in csv_files:
        file_path = os.path.join(output_dir, csv_file)
        if os.path.getsize(file_path) < 100:  # Less than 100 bytes likely means empty/header only
            return False

    return True


def process_batch_wsi(wsi_dir, results_dir, output_base_dir, norm_method, target_image, skip_existing=True):
    """Process multiple WSI files in batch mode"""

    # Find all WSI files
    wsi_files = []
    for ext in ['*.svs', '*.tif']:
        wsi_files.extend(glob.glob(os.path.join(wsi_dir, ext)))

    if not wsi_files:
        logger.error(f"No WSI files found in {wsi_dir}")
        return

    logger.info(f"Found {len(wsi_files)} WSI files to process")

    # Process each WSI
    successful = 0
    failed = 0
    skipped = 0

    for wsi_path in wsi_files:
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        nuclei_data_path = os.path.join(results_dir, wsi_name, 'segmentation', '0.dat')

        if not os.path.exists(nuclei_data_path):
            logger.warning(f"Nuclei data not found for {wsi_name}: {nuclei_data_path}")
            failed += 1
            continue

        # Create output directory for this WSI
        output_dir = os.path.join(output_base_dir, wsi_name)

        # Check if already processed
        if skip_existing and check_output_complete(output_dir):
            logger.info(f"Skipping {wsi_name} - already processed (found complete output files)")
            skipped += 1
            continue

        logger.info(f"\nProcessing WSI: {wsi_name}")
        logger.info(f"WSI path: {wsi_path}")
        logger.info(f"Nuclei data: {nuclei_data_path}")
        logger.info(f"Output directory: {output_dir}")

        try:
            features_df, ith_scores = process_wsi_with_normalization(
                wsi_path,
                nuclei_data_path,
                output_dir,
                norm_method,
                target_image
            )
            successful += 1
            logger.info(f"Successfully processed {wsi_name}")
        except Exception as e:
            logger.error(f"Failed to process {wsi_name}: {str(e)}")
            failed += 1

            # Optional: Remove incomplete output directory
            if os.path.exists(output_dir) and not check_output_complete(output_dir):
                logger.info(f"Removing incomplete output directory: {output_dir}")
                shutil.rmtree(output_dir)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Batch processing complete!")
    logger.info(f"Total WSI files: {len(wsi_files)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Skipped (already complete): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results saved to: {output_base_dir}")
    logger.info("=" * 50)

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract nuclear features with stain normalization'
    )

    # Add mutually exclusive group for single vs batch processing
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--wsi-path',
                            help='Path to single WSI file (.svs, .tif, etc.)')
    mode_group.add_argument('--batch', action='store_true',
                            help='Process all WSI files in batch mode')

    # Single file mode arguments
    parser.add_argument('--nuclei-data',
                        help='Path to nuclei predictions (0.dat file) - required for single file mode')

    # Batch mode arguments
    parser.add_argument('--wsi-dir', default='/data/tiatoolbox/data',
                        help='Directory containing WSI files (for batch mode)')
    parser.add_argument('--results-dir', default='/data/tiatoolbox/results1',
                        help='Directory containing segmentation results (for batch mode)')
    # 修复：去掉 default=True，因为 action='store_true' 默认值就是 False
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip WSI files that already have complete output (default: False)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of all files, even if output exists')

    # Common arguments
    parser.add_argument('--output-dir', default='nuclear_features',
                        help='Output directory')
    parser.add_argument('--norm-method', choices=['reinhard', 'macenko'],
                        default='macenko', help='Normalization method')
    parser.add_argument('--target-image', default='/data/tiatoolbox/reference_all_wsi/reference_macenko.png',
                        help='Target image for normalization')

    args = parser.parse_args()

    if args.batch:
        # Determine skip_existing based on arguments
        skip_existing = not args.force_reprocess

        # Batch processing mode
        process_batch_wsi(
            args.wsi_dir,
            args.results_dir,
            args.output_dir,
            args.norm_method,
            args.target_image,
            skip_existing=skip_existing
        )
    else:
        # Single file processing mode
        if not args.nuclei_data:
            parser.error("--nuclei-data is required when processing a single file")

        # Check files exist
        if not os.path.exists(args.wsi_path):
            logger.error(f"WSI file not found: {args.wsi_path}")
            return

        if not os.path.exists(args.nuclei_data):
            logger.error(f"Nuclei data file not found: {args.nuclei_data}")
            return

        # Process WSI
        features_df, ith_scores = process_wsi_with_normalization(
            args.wsi_path,
            args.nuclei_data,
            args.output_dir,
            args.norm_method,
            args.target_image
        )

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("Feature extraction complete!")
        logger.info(f"Total nuclei processed: {len(features_df)}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Generated files:")
        logger.info("  - nuclear_features.csv: All extracted features")
        logger.info("  - cellular_ith_scores.csv: ITH scores for each feature")
        logger.info("  - analysis_report.txt: Detailed analysis report")
        logger.info("  - Visualization plots")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
