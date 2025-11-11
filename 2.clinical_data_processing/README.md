# Clinical Data Processing Module

## Overview

This module contains three Python scripts for clinical data preprocessing, baseline characteristics comparison, and univariate/multivariate statistical analysis in thyroid cancer distant metastasis prediction studies. These scripts are designed to prepare clinical data and perform comprehensive statistical analyses integrating radiomics features (iTED, 3D ITHscore) with clinical variables.

---

## Scripts

### 1. clinical_data_preprocessing.py

**Purpose**: Comprehensive clinical data preprocessing pipeline

**Key Features**:
- Missing value analysis and visualization
- Outlier detection and treatment (IQR method + Winsorization)
- Variable type identification (continuous vs categorical)
- Multiple imputation for missing data (MICE for continuous, mode for categorical)
- Quality control visualizations in publication-ready format
- Detailed preprocessing report generation

**Main Processing Steps**:
1. Data loading and exploratory analysis
2. Missing value pattern identification and visualization
3. Removal of high missing rate variables (threshold: 30%)
4. Outlier detection using IQR method and treatment via Winsorization
5. Multiple imputation using MICE algorithm
6. Before/after comparison visualizations
7. Cleaned data export with comprehensive report

**Output Files**:
- `clinical_features_processed.csv`: Cleaned and imputed clinical data
- `preprocessing_report.txt`: Detailed processing report
- `missing_percentage.pdf`: Missing value percentage bar chart
- `missing_pattern_heatmap.pdf`: Missing value pattern heatmap
- `outlier_treatment_comparison.pdf`: Before/after outlier treatment comparison
- `imputation_comparison.pdf`: Before/after imputation comparison

**Key Parameters**:
```python
# Missing value threshold
threshold = 0.3  # Remove variables with >30% missing

# Outlier detection method
method = 'IQR'  # Interquartile range method

# Imputation strategy
MICE: IterativeImputer for continuous variables
Mode: most_frequent for categorical variables
```

---

### 2. Comparison_of_clinical_baseline_data.py

**Purpose**: Statistical comparison of clinical baseline characteristics between training and external validation cohorts

**Key Features**:
- Automatic variable type recognition (continuous vs categorical)
- Appropriate statistical test selection (Mann-Whitney U for continuous, Chi-square for categorical)
- Publication-ready summary statistics formatting
- Missing value analysis for both cohorts
- Excel and CSV output with proper formatting and annotations

**Main Functions**:
- Calculate summary statistics (median + IQR for continuous, count + percentage for categorical)
- Perform statistical tests to compare distributions between cohorts
- Generate formatted comparison tables with p-values
- Identify significant differences in baseline characteristics

**Output Files**:
- `clinical_comparison_table.xlsx`: Formatted comparison table (Excel)
- `clinical_comparison_table.csv`: Comparison table (CSV)

**Statistical Tests**:
- **Continuous variables**: Mann-Whitney U test (non-parametric)
- **Categorical variables**: Chi-square test
- **Significance level**: P < 0.05

**Output Format**:
- Continuous variables: Median (IQR) with asterisk notation
- Categorical variables: Count (%) for each category
- Reference category clearly indicated
- P-values formatted according to journal standards

---

### 3. Univariate_and_multivariate_analysis.py

**Purpose**: Comprehensive univariate and multivariate logistic regression analysis integrating radiomics features with clinical variables

**Key Features**:
- Risk stratification using optimal thresholds from iTED, Radiomics, and 3D ITHscore predictions
- Comprehensive univariate analysis with proper reference category handling
- Multivariate analysis including significant variables (P < 0.1)
- Publication-quality forest plots (NEJM/JAMA style)
- Combined result tables with OR, 95% CI, and P-values
- Separate analysis for training and validation cohorts

**Analysis Workflow**:
1. Load clinical data, iTED features, radiomics features, and 3D ITHscore
2. Create risk groups (Low/High) using optimal thresholds from prediction models
3. Perform univariate logistic regression for all variables
4. Identify significant predictors (P < 0.1 for multivariate entry)
5. Build multivariate logistic regression model
6. Generate forest plots and summary tables
7. Create cross-cohort comparison summary

**Main Analyses**:

#### Univariate Analysis
- All clinical variables (continuous and categorical)
- iTED risk group (Low vs High)
- Radiomics risk group (Low vs High)
- 3D ITHscore risk group (Low vs High)
- Proper handling of reference categories
- Odds ratios with 95% confidence intervals

#### Multivariate Analysis
- All three risk scores (iTED, Radiomics, 3D_ITH)
- Clinically significant variables (P < 0.1 in univariate)
- Adjusted odds ratios accounting for confounders
- Model performance metrics

**Output Files**:

*Tables*:
- `Combined_Analysis_Training_Comprehensive.csv`: Training set results
- `Combined_Analysis_Validation_Comprehensive.csv`: Validation set results
- `Summary_Three_Scores.csv`: Cross-cohort comparison of key findings

*Forest Plots* (PNG format, 300 DPI):
- `Forest_Univariate_Training.png`: Training univariate results
- `Forest_Univariate_Validation.png`: Validation univariate results
- `Forest_Multivariate_Training.png`: Training multivariate results
- `Forest_Multivariate_Validation.png`: Validation multivariate results

**Statistical Methods**:
- **Model**: Logistic regression (L2 regularization)
- **Significance**: P < 0.05 (marked in red on forest plots)
- **Entry criterion for multivariate**: P < 0.1
- **Effect size**: Odds ratio with 95% confidence interval
- **Optimal threshold**: Youden's index from ROC analysis

**Visualization Style**:
- NEJM/JAMA standard forest plots
- Log scale for odds ratios
- Red markers for significant results (P < 0.05)
- Gray markers for non-significant results
- Reference line at OR = 1

---

## Requirements

### Python Dependencies
```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
openpyxl (for Excel output)
```

### Input Data Requirements

**For clinical_data_preprocessing.py**:
- Clinical features CSV file with columns:
  - `PatientID`: Patient identifier
  - `M_stage`: Outcome variable (M0/M1 or 0/1)
  - Clinical variables (numerical and categorical)

**For Comparison_of_clinical_baseline_data.py**:
- Training set: `clinical_features_zlyy.csv`
- Validation set: `clinical_features_ydyy.csv`
- Both must have identical column structure

**For Univariate_and_multivariate_analysis.py**:
- Processed clinical features: `clinical_features_processed_{dataset}.csv`
- iTED features: `iTED_features_{dataset}.csv`
- Radiomics features: `radiomics_features_{dataset}.csv`
- 3D ITHscore: `3D_ITHscore_{dataset}.csv`
- Prediction files (for risk stratification):
  - `iTED_predictions_all_patients_{dataset}.csv`
  - `Radiomics_predictions_all_patients_{dataset}.csv`

---

## Installation

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
```

---

## Usage

### Step 1: Preprocess Clinical Data

```bash
python clinical_data_preprocessing.py
```

**Before running**:
1. Update input file path in script:
   ```python
   df = pd.read_csv('F:/Habitat_radiomics/clinical_features_zlyy.csv')
   ```
2. Update output directory:
   ```python
   results_dir = 'F:/Habitat_radiomics/preprocess_result'
   ```

**Output**: Cleaned clinical data ready for analysis

---

### Step 2: Compare Baseline Characteristics

```bash
python Comparison_of_clinical_baseline_data.py
```

**Before running**:
1. Update paths for training and validation datasets
2. Ensure datasets have identical column structure

**Output**: Statistical comparison table for manuscript

---

### Step 3: Univariate and Multivariate Analysis

```bash
python Univariate_and_multivariate_analysis.py
```

**Before running**:
1. Ensure all required input files are present:
   - Processed clinical data
   - iTED features
   - Radiomics features
   - 3D ITHscore
   - Prediction files
2. Update file paths in the script
3. Set output directory:
   ```python
   OUTPUT_DIR = 'F:/Habitat_radiomics/Publication_Results'
   ```

**Output**: Comprehensive analysis tables and forest plots

---

## Typical Workflow

```
Raw Clinical Data
       ↓
[1. clinical_data_preprocessing.py]
       ↓
Cleaned Clinical Data + QC Visualizations
       ↓
[2. Comparison_of_clinical_baseline_data.py]
       ↓
Baseline Characteristics Table
       ↓
[Merge with Radiomics Features]
       ↓
[3. Univariate_and_multivariate_analysis.py]
       ↓
Forest Plots + Statistical Tables
       ↓
Publication-Ready Results
```

---

## Key Notes

### Data Quality
- Variables with >30% missing values are automatically removed
- Outliers are handled using Winsorization (not deletion)
- MICE algorithm preserves relationships between variables during imputation

### Statistical Considerations
- Non-parametric tests used for non-normal continuous data
- Proper reference category handling for categorical variables
- Multivariate models include clinically relevant confounders
- P < 0.1 threshold for multivariate entry (common in medical literature)

### Visualization Standards
- All plots are publication-ready (300 DPI)
- Forest plots follow NEJM/JAMA style guidelines
- Clear distinction between significant and non-significant results
- Consistent color schemes and formatting

### Reproducibility
- Random seed set for imputation (random_state=42)
- All processing steps documented in reports
- Parameter settings saved with outputs

---

## Troubleshooting

### Common Issues

1. **Missing Required Files**
   - Verify all input files exist at specified paths
   - Check file naming conventions match script expectations

2. **Column Name Mismatch**
   - Ensure column names are consistent across datasets
   - Update categorical variable configurations if needed

3. **Insufficient Sample Size**
   - Minimum 20 cases per variable recommended
   - At least 5 events per predictor variable

4. **Convergence Issues in Logistic Regression**
   - Check for perfect separation in categorical variables
   - Consider combining rare categories
   - Verify no extreme multicollinearity

---

## Citation

If you use these scripts in your research, please cite:

```
[Your citation information]
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

## Version History

- **v1.0** (2024): Initial release with preprocessing, comparison, and regression analysis

---

## Acknowledgments

- Statistical methods based on STROBE guidelines for observational studies
- Visualization standards following NEJM and JAMA style requirements
