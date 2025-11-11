# Model Training Module

## Overview

This module contains a comprehensive LightGBM-based machine learning pipeline for predicting distant metastasis in thyroid cancer patients. The script implements multi-modal feature fusion (clinical + radiomics + iTED + 3D ITHscore), advanced class imbalance handling, Optuna-based hyperparameter optimization, and extensive model evaluation with publication-ready visualizations.

---

## Script

### lightgbm_training.py

**Purpose**: End-to-end machine learning pipeline for thyroid cancer metastasis prediction

**Key Features**:

#### 1. Multi-Modal Feature Integration
- Clinical features
- Traditional radiomics features
- iTED (Intratumor Euclidean Distance) features
- 3D Intratumoral Heterogeneity Score (ITHscore)
- Multiple feature combination strategies (13 combinations)

#### 2. Advanced Class Imbalance Handling
- Multiple resampling techniques:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - BorderlineSMOTE
  - SMOTEENN (SMOTE + Edited Nearest Neighbors)
  - SMOTETomek (SMOTE + Tomek Links)
  - SVMSMOTE (Support Vector Machine SMOTE)
- Automatic best resampling method selection
- Handling extreme imbalance ratios (1:6.4 M1:M0)

#### 3. Hyperparameter Optimization
- Optuna framework for Bayesian optimization
- 100 trials per model
- Multi-objective optimization
- Cross-validation during optimization
- Automatic early stopping

#### 4. Model Training & Validation
- Stratified 5-fold cross-validation
- Internal validation set (20% holdout)
- External validation cohort
- Probability calibration (Isotonic Regression)
- Multiple threshold optimization strategies

#### 5. Comprehensive Model Evaluation

**Performance Metrics** (30+ metrics):
- **Discrimination**: AUC-ROC, AUC-PR, Sensitivity, Specificity
- **Calibration**: Brier Score, Calibration curves
- **Classification**: Accuracy, Balanced Accuracy, F1, F2, F3 scores
- **Clinical Utility**: NPV, PPV, Likelihood Ratios, DOR (Diagnostic Odds Ratio)
- **Imbalanced Data**: G-Mean, MCC (Matthews Correlation Coefficient), Cohen's Kappa
- **Screening Tool**: Detection Rate, False Negative Rate, Youden Index

**Model Selection Criteria**:
- **Screening-Optimized Scoring System** (12 components):
  1. Sensitivity (20% weight)
  2. Negative Predictive Value (20%)
  3. F3-Score (15%)
  4. AUC-PR (10%)
  5. Detection Rate (10%)
  6. Matthews Correlation Coefficient (7%)
  7. G-Mean (5%)
  8. Brier Score (5%)
  9. Specificity (3%)
  10. Positive Predictive Value (3%)
  11. Negative Likelihood Ratio (1%)
  12. Stability Score (1%)

#### 6. Risk Stratification
- **Optimal threshold identification** using:
  - Screening-optimized threshold (high sensitivity)
  - Youden's index
  - F2-Score maximization
- **Risk group classification**:
  - Tertile-based (Low/Intermediate/High)
  - Fixed threshold (0.3, 0.7)
  - Clinical threshold (optimized for screening)

#### 7. Model Interpretability
- **SHAP (SHapley Additive exPlanations)** analysis:
  - Global feature importance
  - Summary plots
  - Dependence plots
  - Force plots for individual predictions
  - Waterfall plots
- **Feature contribution analysis**
- **Decision path visualization**

#### 8. Publication-Ready Visualizations
All plots generated at 300 DPI in publication quality:
- ROC curves (with 95% CI)
- Precision-Recall curves
- Calibration plots
- Decision curve analysis (DCA)
- Confusion matrices
- Feature importance plots
- SHAP visualizations
- Box plots for risk groups
- Radar charts for model comparison

---

## Main Workflow

```
Input Data
    ↓
[Feature Loading & Preprocessing]
    ↓
[Class Imbalance Handling] ← Multiple resampling techniques tested
    ↓
[Hyperparameter Optimization] ← Optuna (100 trials)
    ↓
[Model Training] ← 5-Fold CV + Holdout Validation
    ↓
[Probability Calibration] ← Isotonic Regression
    ↓
[Threshold Optimization] ← Screening-optimized
    ↓
[Comprehensive Evaluation] ← 30+ metrics
    ↓
[Best Model Selection] ← Weighted scoring system
    ↓
[External Validation] ← Independent test cohort
    ↓
[Risk Stratification] ← Patient-level predictions
    ↓
[Model Interpretability] ← SHAP analysis
    ↓
Publication-Ready Results
```

---

## Feature Combinations

The script evaluates 13 different feature combinations:

1. **Clinical_Only**: Clinical variables alone
2. **iTED**: iTED features only
3. **Radiomics**: Traditional radiomics features only
4. **3D_ITH**: 3D ITHscore only
5. **Clinical_iTED**: Clinical + iTED
6. **Clinical_Radiomics**: Clinical + Radiomics
7. **Clinical_3D_ITH**: Clinical + 3D ITHscore
8. **iTED_Radiomics**: iTED + Radiomics
9. **iTED_3D_ITH**: iTED + 3D ITHscore
10. **Radiomics_3D_ITH**: Radiomics + 3D ITHscore
11. **Clinical_iTED_Radiomics**: Clinical + iTED + Radiomics
12. **Clinical_iTED_3D_ITH**: Clinical + iTED + 3D ITHscore
13. **All_Features**: All features combined

---

## Requirements

### Python Dependencies

```
# Core
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0

# Imbalanced data handling
imbalanced-learn>=0.9.0

# Optimization
optuna>=3.0.0

# Interpretability
shap>=0.41.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
joblib>=1.1.0
```

### Input Data Requirements

**Required CSV files**:
1. **Training set**:
   - `clinical_features_processed_zlyy.csv`
   - `iTED_features_zlyy.csv`
   - `radiomics_features_zlyy.csv`
   - `3D_ITHscore_zlyy.csv`

2. **External validation set** (optional):
   - `clinical_features_processed_ydyy.csv`
   - `iTED_features_ydyy.csv`
   - `radiomics_features_ydyy.csv`
   - `3D_ITHscore_ydyy.csv`

**Data format**:
- All files must contain `PatientID` column
- Outcome variable: `M_stage` (0 = M0, 1 = M1)
- Feature columns: numerical values (preprocessed, imputed, no missing values)
- Consistent PatientID across all feature files

---

## Installation

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm imbalanced-learn optuna shap matplotlib seaborn joblib

# Or use requirements file
pip install -r requirements.txt
```

---

## Usage

### Basic Usage

```bash
python lightgbm_training.py
```

### Configuration

**Before running**, update these paths in the script:

```python
# Output directory
results_dir = 'F:/Habitat_radiomics/results'

# Input file base paths
base_path_train = 'F:/Habitat_radiomics/'
base_path_external = 'F:/Habitat_radiomics/'
```

### Key Parameters

```python
# Random seed for reproducibility
RANDOM_STATE = 42

# Cross-validation folds
N_FOLDS = 5

# Optuna trials
N_TRIALS = 100

# Screening optimization constraints
min_sensitivity = 0.90  # Minimum sensitivity target
min_npv = 0.95         # Minimum NPV target

# Class imbalance handling
resampling_methods = [
    'SMOTE', 'ADASYN', 'BorderlineSMOTE',
    'SMOTEENN', 'SMOTETomek', 'SVMSMOTE'
]
```

---

## Output Files

### 1. Model Performance Tables

**CSV Files**:
- `model_comparison_metrics.csv`: All models with 30+ metrics
- `best_model_summary.csv`: Detailed best model statistics
- `model_ranking_by_score.csv`: Models ranked by comprehensive score
- `cv_results_all_models.csv`: Cross-validation results
- `external_validation_results.csv`: External cohort performance

### 2. Risk Stratification

**CSV Files**:
- `all_patients_predictions_{model}_zlyy.csv`: Training set predictions
- `all_patients_predictions_{model}_ydyy.csv`: Validation set predictions (if available)
- `patients_risk_groups_for_protein_analysis_{model}.csv`: Risk groups for downstream analysis
- `risk_stratification_report.txt`: Detailed risk stratification analysis

**Columns in prediction files**:
- `PatientID`: Patient identifier
- `True_M_stage`: Actual outcome
- `Predicted_Probability`: Calibrated prediction probability
- `Risk_Group_Tertile`: Tertile-based risk group (Low/Intermediate/High)
- `Risk_Group_Fixed`: Fixed threshold groups
- `Risk_Group_Clinical`: Clinical threshold groups (optimized for screening)

### 3. Visualizations (PNG, 300 DPI)

**ROC & Performance Curves**:
- `roc_curve_{model}.png`: ROC curve with 95% CI
- `pr_curve_{model}.png`: Precision-Recall curve
- `calibration_curve_{model}.png`: Calibration plot
- `dca_curve_{model}.png`: Decision curve analysis

**Model Comparison**:
- `model_comparison_radar.png`: Radar chart comparing all models
- `model_comparison_heatmap.png`: Heatmap of metrics
- `feature_importance_comparison.png`: Feature importance across models

**Best Model Analysis**:
- `confusion_matrix_{best_model}.png`: Confusion matrix
- `threshold_optimization.png`: Optimal threshold selection
- `risk_group_distribution.png`: Risk group box plots

**SHAP Interpretability**:
- `shap_summary_{model}.png`: Global feature importance
- `shap_beeswarm_{model}.png`: SHAP beeswarm plot
- `shap_dependence_top5_{model}.png`: Top 5 feature dependence plots
- `shap_waterfall_examples_{model}.png`: Individual predictions explained

### 4. Model Files

- `best_model_{model_name}.pkl`: Saved best model (joblib)
- `calibrator_{model_name}.pkl`: Probability calibrator
- `optuna_study_{model_name}.pkl`: Optimization history

### 5. Reports

**Text Files**:
- `comprehensive_model_report.txt`: Full analysis report
- `screening_tool_analysis.txt`: High-sensitivity model analysis
- `stability_analysis_report.txt`: Internal vs external validation comparison
- `risk_stratification_report.txt`: Risk grouping details
- `threshold_optimization_report.txt`: Threshold selection rationale

---

## Key Algorithms

### 1. Optuna Optimization

**Hyperparameters optimized**:
```python
{
    'num_leaves': [15, 150],
    'max_depth': [3, 15],
    'learning_rate': [0.001, 0.3],
    'n_estimators': [50, 500],
    'min_child_samples': [5, 100],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'reg_alpha': [0.0, 1.0],
    'reg_lambda': [0.0, 1.0],
    'class_weight': ['balanced', None]
}
```

### 2. Probability Calibration

**Method**: Isotonic Regression
- Non-parametric calibration
- Preserves ranking
- Handles non-linear miscalibration
- Trained on holdout calibration set

### 3. Threshold Optimization

**Screening-Optimized Threshold**:
```python
# Maximize: F2-Score + Sensitivity + NPV
# Subject to:
#   - Sensitivity >= 0.90
#   - NPV >= 0.95
#   - Detection Rate >= 0.85
```

### 4. Model Scoring System

**Formula**:
```
Total Score = Σ (Metric_Score × Weight)

Where:
- Metric_Score = MinMaxScaled(Metric_Value)
- Weights optimized for screening tools
- Higher weight on sensitivity & NPV
```

---

## Model Selection Strategy

### Screening Tool Criteria

**Primary objectives**:
1. **High Sensitivity** (>90%): Minimize false negatives
2. **High NPV** (>95%): Rule out metastasis with confidence
3. **High Detection Rate**: Identify majority of M1 cases
4. **Good Calibration**: Accurate probability estimates

**Secondary considerations**:
- Specificity (avoid excessive false positives)
- External validation stability
- Model complexity (interpretability)

### Ranking System

Models are ranked by:
1. **Total Comprehensive Score** (primary)
2. **Sensitivity** (tie-breaker)
3. **NPV** (second tie-breaker)
4. **External validation AUC** (final tie-breaker)

---

## Interpretability

### SHAP Analysis

**Global Explanations**:
- Feature importance ranking
- Average impact on predictions
- Feature interaction detection

**Local Explanations**:
- Individual prediction breakdown
- Feature contribution for each patient
- Decision path visualization

**Usage for Clinical Translation**:
```python
# Top features identified:
# 1. 3D_ITHscore (if included)
# 2. iTED_mean (heterogeneity metric)
# 3. Clinical T_stage
# 4. Selected radiomics features
```

---

## Performance Expectations

### Typical Results (Class Imbalance ~1:6)

**Internal Validation** (5-Fold CV):
- AUC-ROC: 0.85-0.95
- AUC-PR: 0.60-0.80
- Sensitivity: 0.88-0.96
- Specificity: 0.75-0.90
- NPV: 0.93-0.98

**External Validation**:
- AUC-ROC: 0.80-0.90 (slight drop expected)
- Sensitivity: 0.85-0.93
- NPV: 0.90-0.96

**Best Model Combination**:
- Often: `Clinical_iTED_3D_ITH` or `All_Features`
- Reason: Combines complementary information sources

---
