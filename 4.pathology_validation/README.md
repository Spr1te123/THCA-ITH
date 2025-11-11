# Pathology Validation Module

## Overview

This module provides a comprehensive whole slide image (WSI) analysis pipeline for validating machine learning model predictions through pathology-based biomarkers. The workflow includes WSI processing, nucleus instance segmentation, nuclear feature extraction, and statistical comparison between risk groups predicted by the best-performing model. This module bridges the gap between radiomics-based predictions and histopathological ground truth.

---

## Scripts

### 1. wsi_processing_multi_tissue.py

**Purpose**: Automated whole slide image processing with multi-tissue region support and nucleus instance segmentation

**Key Features**:

#### Tissue Detection & Segmentation
- **Multi-tissue region detection**: Automatically identifies and processes all significant tissue regions in WSI
- **Adaptive tissue boundary detection**: Handles slides with multiple disconnected tissue areas
- **Edge-aware tiling**: Ensures complete coverage of all tissue regions with overlapping tiles

#### Stain Normalization (Optional)
- **Reinhard normalization**: LAB color space normalization
- **Macenko normalization**: H&E stain separation and normalization
- **Vahadane normalization**: Sparse non-negative matrix factorization
- **Automatic target image generation**: Creates normalization reference from WSI tiles

#### Nucleus Instance Segmentation
- **TIAToolbox integration**: Uses pre-trained HoVerNet model
- **Cell type classification**: Identifies 6 cell types:
  - Background (0)
  - Neoplastic Epithelial (1) - Tumor cells
  - Inflammatory (2) - Immune cells
  - Connective (3) - Stromal/connective tissue
  - Dead (4) - Necrotic cells
  - Non-neoplastic Epithelial (5) - Normal epithelial cells

#### Batch Processing & GPU Optimization
- **GPU memory management**: Automatic CUDA memory optimization
- **Batch processing**: Configurable batch size for efficient processing
- **Multi-GPU support**: Can be parallelized across multiple GPUs
- **Quick test mode**: Reduced processing for validation and debugging

#### Visualization & Quality Control
- **Multi-layer PDF output**:
  - WSI overview with tissue regions highlighted
  - Segmentation overlay (nucleus boundaries and types)
  - Individual nucleus visualization (sampled)
  - Cell type distribution statistics
- **Processing logs**: Detailed logs for each WSI
- **Quality metrics**: Tissue coverage, nucleus counts, processing time

**Main Workflow**:
```
WSI Input
    ↓
[Tissue Region Detection] ← Identifies all tissue areas
    ↓
[Tile Generation] ← 5000×5000 tiles with overlap
    ↓
[Stain Normalization] ← Optional (Reinhard/Macenko/Vahadane)
    ↓
[Nucleus Segmentation] ← HoVerNet (batch processing)
    ↓
[Cell Type Classification] ← 6 cell types
    ↓
[Result Aggregation] ← Combine all tiles
    ↓
Output: .dat files + PDF visualization
```

**Output Files**:
- `{wsi_name}_nuclei.dat`: Segmentation results (nucleus coordinates, types, contours)
- `{wsi_name}_visualization.pdf`: Multi-page PDF with overlays and statistics
- `log_{wsi_name}.txt`: Processing log

**Usage**:
```bash
# Single WSI processing
python wsi_processing_multi_tissue.py \
    --input-dir /path/to/wsi/data \
    --output /path/to/results \
    --single-wsi /path/to/slide.svs \
    --batch-size 512

# With stain normalization
python wsi_processing_multi_tissue.py \
    --single-wsi slide.svs \
    --normalize \
    --norm-method macenko \
    --target-image reference.png

# Quick test mode
python wsi_processing_multi_tissue.py \
    --single-wsi slide.svs \
    --quick-test
```

**Key Parameters**:
```python
--tile-size 5000          # Tile size for processing (default: 5000)
--batch-size 512          # Batch size for GPU inference
--overlap 500             # Tile overlap (default: 500)
--normalize               # Enable stain normalization
--norm-method macenko     # Normalization method (reinhard/macenko/vahadane)
--quick-test              # Enable quick test mode (3 tiles only)
```

---

### 2. batch_process_multi_tissue.sh

**Purpose**: Bash script for parallelized batch processing of multiple WSIs across GPUs

**Key Features**:
- **Multi-GPU distribution**: Distributes workload across multiple GPUs
- **File list management**: Reads file lists from GPU-specific text files
- **Progress tracking**: Shows real-time progress and file sizes
- **Error handling**: Captures and logs errors for each slide
- **Automatic environment setup**: Activates conda environment and configures CUDA

**Usage**:
```bash
# Prepare file lists for each GPU
# Create: files_gpu_0.txt, files_gpu_1.txt, etc.
echo "slide1.svs" > files_gpu_0.txt
echo "slide2.svs" >> files_gpu_0.txt
echo "slide3.svs" > files_gpu_1.txt

# Run on GPU 0
bash batch_process_multi_tissue.sh 0

# Run on GPU 1 (in another terminal)
bash batch_process_multi_tissue.sh 1
```

**File List Format**:
```
# files_gpu_0.txt
slide001.svs
slide002.svs
slide003.svs
```

**Features**:
- Automatic GPU allocation via `CUDA_VISIBLE_DEVICES`
- Per-file logging: `log_{basename}_gpu{ID}.txt`
- Progress counter: `[current/total] Processing: filename`
- File size display for monitoring
- Success/failure indicators with emoji (✓/✗)

---

### 3. nuclear_feature_extraction.py

**Purpose**: Extract comprehensive morphological, intensity, and textural features from segmented nuclei

**Key Features**:

#### Feature Extraction (25+ features per nucleus)

**Morphological Features** (14):
- Area, eccentricity, circularity, elongation
- Extent, major/minor axis lengths, perimeter
- Solidity, curvature statistics (mean, max, min, std)

**Intensity Features** (7):
- Mean, std, max, min intensity
- Skewness, kurtosis, entropy

**Textural Features** (4):
- GLCM contrast, homogeneity, energy, correlation

#### Statistical Analysis

**Cellular Intratumoral Heterogeneity (ITH) Metrics**:
- Standard deviation of features across nuclei
- Coefficient of variation (CV)
- Interquartile range (IQR)
- Spatial heterogeneity measures

**Cell Type-Specific Analysis**:
- Features computed separately for each cell type
- Tumor cell (Type 1) features prioritized
- Stromal-tumor interface characterization

#### Spatial Analysis

**Neighborhood Analysis**:
- k-nearest neighbor distances (k=5, 10, 20)
- Local density estimation
- Spatial clustering metrics

**Regional Heterogeneity**:
- Tessellation-based analysis
- Grid-based feature aggregation
- Spatial autocorrelation

**Output Files**:
- `nuclear_features_{patient_id}.csv`: Per-nucleus features
- `cellular_ith_scores_{patient_id}.csv`: Patient-level ITH metrics
- `spatial_metrics_{patient_id}.csv`: Spatial heterogeneity measures

**Usage**:
```python
from nuclear_feature_extraction import NuclearFeatureExtractor

# Initialize extractor
extractor = NuclearFeatureExtractor(normalizer=None)

# Load segmentation results
seg_data = joblib.load('slide_nuclei.dat')

# Extract features
features = extractor.extract_all_features(seg_data, wsi_path)

# Calculate ITH scores
ith_scores = extractor.calculate_ith_metrics(features)
```

---

### 4. generate_reference_all.py

**Purpose**: Generate high-quality reference images for stain normalization from a cohort of WSIs

**Key Features**:

#### Intelligent Tile Sampling
- **Multi-WSI sampling**: Extracts tiles from all available WSIs
- **Quality-based selection**: Scores tiles by staining quality, tissue density
- **Diverse sampling**: Ensures representation across different slides

#### Quality Assessment
- **Stain quality score**: H&E stain intensity and distribution
- **Tissue density**: Nucleus density and tissue coverage
- **Color histogram analysis**: Balanced color distribution
- **Artifact detection**: Filters out folds, bubbles, pen marks

#### Reference Image Generation
- **Tile clustering**: K-means clustering to find representative tiles
- **Consensus building**: Selects tiles close to cluster centroids
- **Multiple references**: Generates 3-5 reference images per cohort
- **Visual validation**: Outputs composite images for manual review

**Usage**:
```bash
python generate_reference_all.py \
    --wsi-dir /path/to/wsi/folder \
    --output-dir /path/to/output \
    --tiles-per-wsi 3 \
    --n-references 5
```

**Output Files**:
- `reference_image_1.png` to `reference_image_N.png`: Final reference images
- `tile_quality_scores.csv`: Quality metrics for all extracted tiles
- `reference_generation_report.txt`: Summary of generation process

---

### 5. thyroid_cancer_pathology_analysis.py

**Purpose**: Comprehensive statistical analysis comparing pathology features between model-predicted risk groups

**Key Features**:

#### Risk Group Definition
- **Low risk group**: Patients predicted as low metastasis risk
- **High risk group**: Patients predicted as high metastasis risk
- Based on best-performing model's screening-optimized threshold

#### Statistical Analyses (10+ analyses)

**1. Intratumoral Heterogeneity (ITH) Analysis**:
- Compare ITH metrics between risk groups
- Mann-Whitney U tests for all ITH features
- Effect size calculation (Cohen's d)
- Identifies ITH signatures associated with metastasis risk

**2. Cell Composition Analysis**:
- Tumor cell proportion (Type 1)
- Inflammatory cell proportion (Type 2)
- Stromal cell proportion (Type 3)
- Immune infiltration patterns

**3. Nuclear Morphology Analysis**:
- Tumor nucleus size, shape, eccentricity
- Nuclear pleomorphism quantification
- Chromatin texture differences

**4. Spatial Architecture Analysis**:
- Tumor-stroma interface complexity
- Cell clustering patterns
- Nearest neighbor distances
- Spatial autocorrelation

**5. Microenvironment Analysis**:
- Tumor-infiltrating lymphocyte (TIL) density
- Stromal density and organization
- Vascular density estimation

**6. Correlation with Radiomics**:
- Correlation between ITH and iTED features
- Correlation between nuclear features and 3D_ITHscore
- Multi-modal validation

**7. Feature Importance Ranking**:
- Discriminative power of each pathology feature
- AUC for risk group classification
- Feature selection for biomarker identification

**8. Survival Correlation** (if outcome data available):
- Association with actual metastasis status
- Hazard ratios for key features
- Kaplan-Meier curves

**9. Subgroup Analysis**:
- Analysis by tumor T-stage
- Analysis by lymph node status
- Stratified analysis by clinical variables

**10. Comprehensive Scoring System**:
- Integrated pathology risk score
- Combines multiple ITH and morphology metrics
- Validates against model predictions

#### Visualization Outputs (20+ plots)

**Publication-Quality Figures** (300 DPI):
- Box plots comparing ITH features
- Violin plots for cell composition
- Heatmaps for correlation matrices
- Scatter plots for feature relationships
- Forest plots for effect sizes
- Radar charts for multi-feature comparison
- PCA plots for dimensionality reduction
- t-SNE for patient clustering

**Usage**:
```python
from thyroid_cancer_pathology_analysis import ThyroidCancerPathologyAnalyzer

# Initialize analyzer
analyzer = ThyroidCancerPathologyAnalyzer(
    data_dir='/path/to/nuclear/features',
    output_dir='/path/to/results'
)

# Load patient data
analyzer.load_patient_data()

# Run all analyses
analyzer.run_comprehensive_analysis()

# Generate report
analyzer.generate_final_report()
```

**Output Files**:
- `ith_analysis_results.csv`: ITH comparison statistics
- `cell_composition_analysis.csv`: Cell type proportions
- `nuclear_morphology_analysis.csv`: Morphological feature statistics
- `spatial_analysis_results.csv`: Spatial metrics comparison
- `correlation_with_radiomics.csv`: Radiomics-pathology correlations
- `feature_importance_ranking.csv`: Discriminative features
- `comprehensive_pathology_report.pdf`: Final report with all figures
- Individual PNG files for each figure (300 DPI)

---

## Complete Workflow

```
1. WSI Processing
   ↓
[wsi_processing_multi_tissue.py] ← Process all patient WSIs
   ↓
   Outputs: {patient_id}_nuclei.dat files

2. Reference Generation (Optional)
   ↓
[generate_reference_all.py] ← Create normalization reference
   ↓
   Outputs: reference_image.png

3. Feature Extraction
   ↓
[nuclear_feature_extraction.py] ← Extract features from .dat files
   ↓
   Outputs: nuclear_features_{patient_id}.csv
           cellular_ith_scores_{patient_id}.csv

4. Statistical Analysis
   ↓
[thyroid_cancer_pathology_analysis.py] ← Compare risk groups
   ↓
   Outputs: Analysis results + Visualizations + PDF report

5. Model Interpretation
   ↓
   Validate model predictions with pathology evidence
```

---

## Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 11GB+ VRAM (RTX 2080 Ti or better)
- **RAM**: 32GB+ recommended for large WSIs
- **Storage**: ~1GB per WSI for processed outputs
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Python Dependencies

```
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-image>=0.19.0
scipy>=1.7.0

# TIAToolbox for WSI processing
tiatoolbox>=1.4.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Statistical analysis
scikit-learn>=1.0.0
statsmodels>=0.13.0

# File I/O
joblib>=1.1.0

# GPU support
torch>=1.10.0  # For TIAToolbox backend
```

### External Dependencies
- **CUDA**: 11.3+ (for GPU support)
- **cuDNN**: 8.2+ (for GPU support)
- **OpenSlide**: For reading proprietary WSI formats (optional but recommended)

---

## Installation

```bash
# Create conda environment
conda create -n tiatoolbox-dev python=3.9
conda activate tiatoolbox-dev

# Install PyTorch with CUDA support
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install TIAToolbox
pip install tiatoolbox

# Install other dependencies
pip install opencv-python scikit-image scipy pandas \
            matplotlib seaborn scikit-learn statsmodels joblib

# Install OpenSlide (optional, for better format support)
conda install openslide -c conda-forge
```

---

## Usage Examples

### Example 1: Process Single WSI

```bash
# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Process WSI
python wsi_processing_multi_tissue.py \
    --input-dir /data/wsi \
    --output /data/results \
    --single-wsi /data/wsi/patient_001.svs \
    --batch-size 512
```

### Example 2: Batch Processing with Multiple GPUs

```bash
# Prepare file lists
ls /data/wsi/*.svs | head -50 > files_gpu_0.txt
ls /data/wsi/*.svs | tail -50 > files_gpu_1.txt

# Launch on GPU 0
bash batch_process_multi_tissue.sh 0 &

# Launch on GPU 1
bash batch_process_multi_tissue.sh 1 &

# Monitor progress
watch -n 5 'tail files_gpu_*.txt'
```

### Example 3: Complete Pipeline

```python
# Step 1: Process WSIs (via bash script)
# bash batch_process_multi_tissue.sh 0

# Step 2: Extract features
from nuclear_feature_extraction import NuclearFeatureExtractor
import joblib

extractor = NuclearFeatureExtractor()

for patient_id in [47, 365, 491, ...]:
    seg_file = f'/data/results/patient_{patient_id}_nuclei.dat'
    wsi_file = f'/data/wsi/patient_{patient_id}.svs'
    
    # Load segmentation
    seg_data = joblib.load(seg_file)
    
    # Extract features
    features = extractor.extract_all_features(seg_data, wsi_file)
    features.to_csv(f'nuclear_features_{patient_id}.csv', index=False)
    
    # Calculate ITH
    ith_scores = extractor.calculate_ith_metrics(features)
    ith_scores.to_csv(f'cellular_ith_scores_{patient_id}.csv', index=False)

# Step 3: Statistical analysis
from thyroid_cancer_pathology_analysis import ThyroidCancerPathologyAnalyzer

analyzer = ThyroidCancerPathologyAnalyzer(
    data_dir='/data/nuclear_features',
    output_dir='/data/pathology_analysis'
)

analyzer.load_patient_data()
analyzer.run_comprehensive_analysis()
analyzer.generate_final_report()
```

---

## Output Interpretation

### WSI Processing Outputs

**{patient_id}_nuclei.dat**:
- Binary file containing segmentation results
- Structure: List of dictionaries, one per nucleus
  ```python
  {
      'centroid': (x, y),           # Nucleus center coordinates
      'contour': [(x1,y1), ...],    # Boundary points
      'bbox': (x, y, w, h),         # Bounding box
      'type': int,                  # Cell type (0-5)
      'prob': float                 # Classification confidence
  }
  ```

### Feature Extraction Outputs

**nuclear_features_{patient_id}.csv**:
- One row per nucleus
- Columns: nucleus_id, centroid_x, centroid_y, nucleus_type, [25+ feature columns]

**cellular_ith_scores_{patient_id}.csv**:
- One row (patient-level aggregation)
- Columns: Features with suffixes:
  - `_ith_std`: Standard deviation across nuclei
  - `_ith_cv`: Coefficient of variation
  - `_ith_iqr`: Interquartile range
  - `_ith_range`: Max - Min

### Statistical Analysis Outputs

**ith_analysis_results.csv**:
- Columns: feature, high_mean, high_std, low_mean, low_std, p_value, effect_size
- Interpretation:
  - p_value < 0.05: Significant difference
  - effect_size > 0.5: Medium effect
  - effect_size > 0.8: Large effect

---

## Performance Benchmarks

### Processing Speed (RTX 3090, 24GB VRAM)

**Single WSI**:
- Small slide (5000×5000 μm²): ~5-10 minutes
- Medium slide (15000×15000 μm²): ~20-30 minutes
- Large slide (30000×30000 μm²): ~45-60 minutes

**Batch Processing** (50 slides):
- 1 GPU: ~30 hours
- 2 GPUs: ~15 hours
- 4 GPUs: ~8 hours

**Memory Usage**:
- WSI loading: ~2-4GB RAM per slide
- Segmentation: ~8-10GB GPU memory
- Feature extraction: ~1-2GB RAM per patient

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch-size` (try 256, 128, or 64)
   - Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - Close other GPU processes

2. **Segmentation Quality Poor**
   - Try enabling stain normalization: `--normalize --norm-method macenko`
   - Check tissue detection: review PDF visualization
   - Adjust tissue threshold in code (line ~473)

3. **Missing Tissue Regions**
   - Script now detects ALL tissue regions by default
   - Check log for "Found X tissue regions"
   - Manually verify tissue_mask visualization

4. **Slow Processing**
   - Increase `--batch-size` if GPU memory allows
   - Use `--quick-test` for validation first
   - Ensure NVMe SSD for WSI storage (I/O bottleneck)

5. **OpenSlide Errors**
   - Install OpenSlide: `conda install openslide -c conda-forge`
   - Check file format: Convert to `.svs` if needed
   - Verify file integrity: `openslide-show-properties slide.svs`

---

## Best Practices

### WSI Processing
- Always run `--quick-test` on one slide first to verify setup
- Use stain normalization for slides from different scanners
- Monitor GPU temperature during long batch jobs
- Save intermediate results frequently

### Feature Extraction
- Extract features immediately after segmentation (while data is fresh)
- Validate feature distributions before analysis
- Check for outliers (artifacts can affect features)
- Normalize features before statistical tests

### Statistical Analysis
- Use non-parametric tests (Mann-Whitney U) for non-normal distributions
- Always check effect sizes, not just p-values
- Perform multiple testing correction (Bonferroni, FDR)
- Visualize data before running tests

### Reproducibility
- Document all parameters used
- Save random seeds
- Version control analysis scripts
- Archive raw segmentation data

---

## Citation

If you use this pipeline in your research, please cite:

```
[Your citation information]
TIAToolbox: [TIAToolbox citation]
HoVerNet: [HoVerNet citation if using that model]
```

---

## License

[Specify your license]

---

## Contact

For questions or issues:
- **Email**: [your-email@example.com]
- **GitHub Issues**: [Repository URL]/issues

---

## Acknowledgments

- **TIAToolbox**: Computational pathology toolbox
- **HoVerNet**: Nucleus instance segmentation model
- **OpenSlide**: Whole slide image I/O library

---

## Version History

- **v1.0** (2024): Initial release with multi-tissue support, comprehensive feature extraction, and statistical analysis
