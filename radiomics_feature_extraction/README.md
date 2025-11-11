# Thyroid Radiomics Analysis Pipeline

## Overview

This repository contains two Python scripts for comprehensive radiomics analysis of thyroid imaging data, developed for medical image feature extraction and tumor heterogeneity assessment.

## Scripts

### 1. Thyroid_Radiomics_Standard.py

**Purpose**: Standard radiomics feature extraction pipeline

**Key Features**:
- Extracts traditional radiomics features from medical images (NIfTI format)
- Multi-class feature extraction: shape, first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM
- Integrated quality control system (SNR, contrast, tumor volume validation)
- Parallel processing support for batch analysis
- Automatic visualization generation
- Comprehensive logging and error handling

**Main Functions**:
- Image quality control and validation
- PyRadiomics-based feature extraction
- Statistical summary generation
- Result visualization (tumor contours, feature distributions)
- Incremental result saving to prevent data loss

**Output**:
- `1_Radiomics_Features/`: Extracted radiomics features (CSV)
- `2_Summary_Stats/`: Processing summary and statistics
- `3_Visualizations/`: Generated plots and images
- `4_Code_And_Settings/`: Parameter settings and script backup
- `5_Quality_Control/`: QC reports and metrics

---

### 2. TITAN_Pipeline.py

**Purpose**: Tumor habitat analysis and intratumoral heterogeneity quantification (TITAN: Tumor Imaging-based Tumor Analysis and kNowledge)

**Key Features**:
- 3D SLIC superpixel segmentation for tumor subregion identification
- iTED (intratumor Euclidean Distance) calculation using Gaussian Mixture Models
- 3D ITHscore (Intratumoral Heterogeneity score) computation with K-means clustering
- Adaptive mask expansion with thyroid boundary constraints
- SLIC stability assessment and bootstrap confidence intervals
- Statistical power analysis for sample size adequacy

**Main Functions**:
- Tumor subregion segmentation (3D SLIC algorithm)
- Subregion-level radiomics feature extraction
- iTED feature computation (measuring heterogeneity via GMM)
- 3D ITHscore calculation (quantifying spatial heterogeneity)
- Mask expansion for small tumors (adaptive algorithm)
- SLIC reproducibility validation
- Comprehensive quality control

**Output**:
- `1_iTED_Features/`: iTED-based heterogeneity features
- `2_ITHscore_3D/`: 3D ITHscore metrics with confidence intervals
- `3_Summary_Stats/`: Processing logs and statistics
- `4_Intermediate_Subregion_Segmentations/`: Subregion mask files
- `5_Intermediate_Subregion_Features/`: Subregion-level features
- `6_Visualizations/`: Subregion visualization and clustering results
- `7_Code_And_Settings/`: Parameter configurations
- `8_Quality_Control/`: Detailed QC reports

---

## Requirements

### Python Dependencies
```
numpy
pandas
SimpleITK
pyradiomics
matplotlib
scikit-image
scikit-learn
scipy
```

### System Requirements
- Python 3.7+
- Multi-core CPU (recommended for parallel processing)
- Sufficient RAM for medical image processing (8GB+ recommended)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Spr1te123/THCA-ITH.git
cd radiomics_feature_extraction

# Install dependencies
pip install numpy pandas SimpleITK pyradiomics matplotlib scikit-image scikit-learn scipy
```

---

## Usage

### Standard Radiomics Feature Extraction

```bash
python Thyroid_Radiomics_Standard.py
```

**Before running**:
1. Update input folder paths in the script:
   ```python
   image_folder = "/path/to/your/images"
   mask_folder = "/path/to/your/masks"
   main_output_folder = "/path/to/output"
   ```

2. Configure parameters in `PARAMS` dictionary as needed

3. Ensure image and mask files have matching filenames (e.g., `001.nii.gz`)

### Tumor Habitat Analysis (TITAN)

```bash
python TITAN_Pipeline.py
```

**Before running**:
1. Update input/output paths in the script
2. Adjust SLIC and clustering parameters in `PARAMS` if needed
3. Set visualization patient list in `VISUALIZATION_PATIENT_LIST`

---

## Input Data Format

- **Image Format**: NIfTI (`.nii.gz`)
- **File Naming**: Images and masks must have identical filenames
- **Mask Requirements**: Binary mask (0 = background, non-zero = tumor)
- **Coordinate System**: Image and mask must share the same spatial reference

---

## Parameter Configuration

### Key Parameters (Thyroid_Radiomics_Standard.py)

```python
PARAMS = {
    'binCount': 32,                    # Histogram bins for discretization
    'normalize': True,                  # Image normalization
    'NUM_PROCESSES': cpu_count() - 2,   # Parallel processing cores
    'MIN_SNR': 5.0,                     # Minimum signal-to-noise ratio
    'MIN_TUMOR_VOLUME_MM3': 50.0,       # Minimum tumor volume (mm³)
}
```

### Key Parameters (TITAN_Pipeline.py)

```python
PARAMS = {
    'TARGET_VOXELS_PER_SUBREGION': 100,      # SLIC target voxels per subregion
    'MIN_VALID_SUBREGIONS_FOR_ITED': 5,      # Minimum subregions for iTED
    'MAX_CLUSTERS_FOR_ITHSCORE': 8,          # Maximum K-means clusters
    'EXPANSION_DISTANCE_MM': 3.0,            # Mask expansion distance
    'ENABLE_MASK_EXPANSION': True,           # Enable adaptive expansion
    'ENABLE_3D_ITHSCORE_CI': True,           # Bootstrap confidence intervals
}
```

---

## Output Description

### Radiomics Features (CSV)
- Row = Patient
- Columns = PatientID + extracted features
- Features named as `original_[featureclass]_[featurename]`

### iTED Features (CSV)
- iTED-based heterogeneity metrics (mean, std, entropy, etc.)
- Derived from GMM modeling of subregion feature distributions

### 3D ITHscore (CSV)
- Spatial heterogeneity score (0-1 scale, higher = more heterogeneous)
- Number of distinct tumor habitats
- Bootstrap confidence intervals (if enabled)

---

## Quality Control

Both scripts include built-in QC checks:
- **SNR**: Signal-to-noise ratio validation
- **Contrast**: Tumor-background contrast assessment
- **Volume**: Minimum tumor size verification
- **Intensity Distribution**: Pixel value consistency checks
- **SLIC Stability**: Segmentation reproducibility (TITAN only)

Results flagged as failing QC should be reviewed manually.

---

## Parallel Processing

Both scripts support multi-core processing:
- Default: `cpu_count() - 2` processes
- Adjust via `NUM_PROCESSES` parameter
- Memory-intensive: monitor RAM usage for large datasets

---
