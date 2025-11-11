# Protein Expression Validation Module

## Overview

This module provides comprehensive statistical analysis and visualization of protein expression patterns to validate machine learning model predictions. The analysis compares protein expression levels between risk groups (Low vs High) predicted by the best-performing model, using a panel of proteins previously identified as associated with thyroid cancer distant metastasis. This module bridges molecular biomarkers with AI-based risk predictions.

---

## Script

### protein_expression_analysis.py

**Purpose**: Statistical analysis and visualization of differential protein expression between model-predicted risk groups

**Key Features**:

#### Protein Panel
Analysis of **8 key proteins** associated with thyroid cancer metastasis:
- **SUCLG1**: Succinate-CoA ligase GDP-forming subunit alpha
- **DLAT**: Dihydrolipoyllysine-residue acetyltransferase
- **IDH3B**: Isocitrate dehydrogenase 3 (NAD+) beta
- **ACSF2**: Acyl-CoA synthetase family member 2
- **SUCLG2**: Succinate-CoA ligase GDP/ADP-forming subunit beta
- **ACO2**: Aconitase 2, mitochondrial
- **CYCS**: Cytochrome c, somatic
- **VDAC2**: Voltage-dependent anion channel 2

*Note: These proteins are involved in mitochondrial metabolism, oxidative phosphorylation, and apoptosis pathways*

#### Expression Data Format
- **Expression levels**: Categorical (4 levels)
  - Negative (0)
  - Weak Positive (1)
  - Moderate Positive (2)
  - Strong Positive (3)
- **Groups**: Low Risk vs High Risk (from model predictions)
- **Outcome**: M_stage (actual metastasis status for validation)

#### Statistical Analyses

**1. Chi-Square Test of Independence**:
- Tests association between protein expression and risk groups
- Null hypothesis: Protein expression is independent of risk group
- Reports: χ² statistic, p-value, degrees of freedom

**2. Effect Size Calculation**:
- **Cramer's V**: Measures strength of association
  - 0 = No association
  - 1 = Perfect association
- **Interpretation**:
  - Small effect: V < 0.3
  - Medium effect: 0.3 ≤ V < 0.5
  - Large effect: V ≥ 0.5

**3. Correlation Analysis**:
- Inter-protein expression correlations
- Separate analysis for each risk group
- Identifies co-expression patterns

**4. Expression Pattern Analysis**:
- Dominant expression level per risk group
- Mean expression level comparison
- Distribution analysis

#### Visualization Outputs (Publication Quality, 300 DPI)

**1. Protein Expression by Risk Group** (`protein_expression_by_risk_group.pdf`):
- 8-panel stacked bar chart
- Shows expression level distribution (%)
- Annotated with statistical results:
  - χ² statistic
  - P-value with significance stars (*, **, ***)
  - Cramer's V effect size
- Significant proteins highlighted with light red background

**2. Statistical Significance Summary** (`protein_significance_summary.pdf`):
- **Left panel**: -log10(P-value) horizontal bar chart
  - Red bars: Significant (p < 0.05)
  - Gray bars: Non-significant
  - Dashed line: p = 0.05 threshold
  - Higher bars = More significant
- **Right panel**: Cramer's V effect size
  - Shows strength of association
  - Reference lines for small/medium/large effects
- Includes interpretation guides on both panels

**3. Correlation Heatmap** (`protein_correlation_by_risk.pdf`):
- Side-by-side correlation matrices
- Separate for Low Risk and High Risk groups
- Color scale: Red (positive correlation) to Blue (negative correlation)
- Annotated with correlation coefficients

**4. Mean Expression Heatmap Matrix** (`protein_expression_heatmap_matrix.pdf`):
- 2D heatmap: Risk Groups × Proteins
- Color intensity represents mean expression level
- Scale: 0 (Negative) to 3 (Strong Positive)
- Annotated with exact mean values

#### Data Outputs

**CSV Files**:
- `protein_statistical_results.csv`: Chi-square test results for all proteins
  - Columns: Protein, Chi-square, P-value, Cramer's V, Significance
- `protein_analysis_summary.csv`: Detailed summary with effect size interpretation
  - Additional columns: Effect_Size (categorical), Significant (boolean)
- `protein_mean_expression_by_risk.csv`: Mean expression levels matrix
- `protein_crosstabs.xlsx`: Contingency tables for each protein (multi-sheet Excel)

**Text Report**:
- `protein_analysis_report.txt`: Comprehensive analysis summary
  - Sample distribution
  - Significant proteins list
  - Non-significant proteins list
  - Dominant expression patterns
  - Clinical significance interpretation

---

## Workflow

```
Input Data
    ↓
[Load & Preprocess] ← CSV with protein expression + risk groups
    ↓
[Encode Expression] ← Convert categorical to numerical (0-3)
    ↓
[Statistical Tests] ← Chi-square test + Cramer's V
    ↓
[Correlation Analysis] ← Inter-protein correlations
    ↓
[Visualizations] ← Generate 4 publication-quality PDFs
    ↓
[Summary Report] ← Text report + CSV outputs
    ↓
Results: Proteins validating model predictions
```

---

## Requirements

### Python Dependencies

```
# Core data analysis
pandas>=1.3.0
numpy>=1.21.0

# Statistical analysis
scipy>=1.7.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Warning suppression
warnings (standard library)
```

### Input Data Requirements

**CSV file structure**:
```
PatientID, Risk_Group_Clinical, M_stage, SUCLG1, DLAT, IDH3B, ACSF2, SUCLG2, ACO2, CYCS, VDAC2
001,       Low,                 0,       Negative, Weak Positive, ...
002,       High,                1,       Strong Positive, Moderate Positive, ...
...
```

**Required columns**:
- `PatientID`: Patient identifier
- `Risk_Group_Clinical`: Model-predicted risk group (Low/High)
- `M_stage`: Actual metastasis status (0=M0, 1=M1) - for validation
- **Protein columns**: Expression levels (Negative/Weak Positive/Moderate Positive/Strong Positive)

---

## Installation

```bash
# Install dependencies
pip install pandas numpy scipy matplotlib seaborn

# Or use requirements file
pip install -r requirements.txt
```

---

## Usage

### Basic Usage

```bash
python protein_expression_analysis.py
```

### Configuration

**Before running**, update these settings in the script:

```python
# Input file path
filepath = 'F:/Habitat_radiomics/模型解释蛋白组学.csv'

# Output directory
OUTPUT_DIR = 'F:/Habitat_radiomics/protein_expression_analysis'

# Protein list (can be customized)
protein_cols = ['SUCLG1', 'DLAT', 'IDH3B', 'ACSF2', 
                'SUCLG2', 'ACO2', 'CYCS', 'VDAC2']
```

### Python API Usage

```python
from protein_expression_analysis import *

# Load and preprocess data
df = load_and_preprocess_data('your_data.csv')

# Encode protein expression
df, encoded_cols = encode_protein_expression(df, protein_cols)

# Run statistical analysis
results = analyze_all_proteins(df, protein_cols)

# Generate visualizations
plot_protein_expression_heatmap(df, protein_cols)
plot_significance_summary(results)
plot_correlation_heatmap(df, protein_cols, encoded_cols)

# Create summary report
pattern_matrix = create_expression_pattern_matrix(df, protein_cols)
report = generate_summary_report(df, results, pattern_matrix)
```

---

## Output Interpretation

### Statistical Results

**Chi-Square Test**:
- **χ² statistic**: Measure of deviation from independence
  - Higher values = Greater association
- **P-value**: Probability of observing results if null hypothesis is true
  - p < 0.05: Statistically significant
  - p < 0.01: Highly significant
  - p < 0.001: Very highly significant

**Cramer's V**:
- **Effect size**: Standardized measure of association strength
- Independent of sample size (unlike χ²)
- **Interpretation guide**:
  - V < 0.1: Negligible effect
  - 0.1 ≤ V < 0.3: Small effect
  - 0.3 ≤ V < 0.5: Medium effect
  - V ≥ 0.5: Large effect

### Significance Symbols

- `***`: p < 0.001 (very highly significant)
- `**`: p < 0.01 (highly significant)
- `*`: p < 0.05 (significant)
- `ns`: p ≥ 0.05 (not significant)

### Clinical Interpretation

**Significant protein (example)**:
```
SUCLG1: χ²=15.32, p=0.0018 **, V=0.42
```
**Interpretation**: 
- SUCLG1 expression is significantly different between risk groups (p=0.0018)
- The association is of medium strength (V=0.42)
- This protein may serve as a molecular biomarker validating the model's risk stratification

**Non-significant protein (example)**:
```
CYCS: χ²=3.21, p=0.36 ns, V=0.19
```
**Interpretation**:
- CYCS expression does not differ significantly between risk groups
- The model's predictions are not validated by this protein
- May indicate protein is not a good biomarker for this specific risk stratification

---

## Typical Results

### Expected Patterns

**Scenario 1: Model validated by protein biomarkers**
- Multiple proteins show significant differences (p < 0.05)
- Medium to large effect sizes (V > 0.3)
- High-risk group shows higher expression of metastasis-associated proteins
- **Interpretation**: Model predictions are supported by molecular evidence

**Scenario 2: Partial validation**
- Some proteins significant, others not
- Mixed effect sizes
- **Interpretation**: Model captures some but not all biological signals

**Scenario 3: No validation**
- Most proteins non-significant
- Small effect sizes (V < 0.3)
- **Interpretation**: Model predictions may be driven by imaging features not reflected in this protein panel

### Sample Statistics

For a cohort of ~30 patients (15 Low Risk, 15 High Risk):
- **Power to detect large effects (V=0.5)**: >80%
- **Power to detect medium effects (V=0.3)**: ~60%
- **Significant proteins expected**: 2-5 out of 8 (if true differences exist)

---

## Best Practices

### Data Quality
- Ensure consistent protein expression assessment across all samples
- Use the same antibodies and scoring criteria
- Blind assessors to risk group assignment
- Include quality control samples

### Statistical Considerations
- **Multiple testing**: Consider Bonferroni correction for p-values
  - Adjusted threshold: p < 0.05/8 = 0.00625
- **Sample size**: Minimum 10-15 patients per risk group recommended
- **Effect size**: Always report alongside p-values
- **Non-parametric tests**: Chi-square is robust for categorical data

### Visualization
- Use colorblind-friendly palettes (already implemented)
- Save all plots at 300 DPI for publication
- Include error bars or confidence intervals when possible
- Always annotate with statistical values

### Reproducibility
- Document protein assessment methodology
- Record antibody sources and dilutions
- Archive raw expression scoring data
- Version control analysis scripts

---

## Advanced Usage

### Adding New Proteins

```python
# Modify protein list
protein_cols = [
    'SUCLG1', 'DLAT', 'IDH3B', 'ACSF2', 
    'SUCLG2', 'ACO2', 'CYCS', 'VDAC2',
    'NEW_PROTEIN_1', 'NEW_PROTEIN_2'  # Add new proteins
]

# Ensure new proteins are in your CSV file
# Run analysis as normal
```

### Stratified Analysis

```python
# Analyze by subgroups (e.g., T-stage)
for t_stage in df['T_stage'].unique():
    subset = df[df['T_stage'] == t_stage]
    results = analyze_all_proteins(subset, protein_cols)
    # Generate subgroup-specific outputs
```

### Continuous Expression Data

If expression is quantitative (e.g., Western blot intensities):

```python
from scipy.stats import mannwhitneyu

def analyze_continuous_protein(df, protein_col):
    """Analyze continuous protein expression"""
    high_risk = df[df['Risk_Group_Clinical'] == 'High'][protein_col]
    low_risk = df[df['Risk_Group_Clinical'] == 'Low'][protein_col]
    
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = mannwhitneyu(high_risk, low_risk, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_risk.var() + low_risk.var()) / 2)
    cohens_d = (high_risk.mean() - low_risk.mean()) / pooled_std
    
    return {
        'protein': protein_col,
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d
    }
```

---

## Integration with Other Modules

### With Radiomics Features

```python
# Correlate protein expression with radiomics features
import pandas as pd
from scipy.stats import spearmanr

# Load radiomics features
radiomics = pd.read_csv('iTED_features.csv')

# Merge with protein data
merged = pd.merge(protein_df, radiomics, on='PatientID')

# Correlate proteins with iTED features
for protein in protein_cols:
    for radiomics_feature in ['iTED_mean', 'iTED_std', '3D_ITHscore']:
        corr, pval = spearmanr(
            merged[f'{protein}_encoded'], 
            merged[radiomics_feature]
        )
        print(f"{protein} vs {radiomics_feature}: r={corr:.3f}, p={pval:.3f}")
```

### With Pathology Features

```python
# Validate against pathology ITH metrics
pathology = pd.read_csv('cellular_ith_scores.csv')

merged = pd.merge(protein_df, pathology, on='PatientID')

# Test if proteins correlate with pathological heterogeneity
for protein in protein_cols:
    corr, pval = spearmanr(
        merged[f'{protein}_encoded'],
        merged['nuclear_size_ith_std']  # Example pathology ITH metric
    )
    print(f"{protein} vs nuclear_size_ITH: r={corr:.3f}, p={pval:.3f}")
```

---

## Troubleshooting

### Common Issues

1. **"KeyError: Risk_Group_Clinical"**
   - Check column names in CSV (case-sensitive)
   - Ensure 'Risk_Group_Clinical' column exists
   - Verify column contains 'Low' and 'High' values

2. **"ValueError: All input arrays must have same number of dimensions"**
   - Check for missing protein expression data
   - Verify all protein columns have valid categorical values
   - Remove rows with NaN values in protein columns

3. **Chi-square test warning: "Expected frequency is less than 5"**
   - Small sample size or sparse contingency table
   - Consider Fisher's exact test for 2×2 tables
   - Combine expression levels (e.g., Negative+Weak vs Moderate+Strong)

4. **All proteins show p > 0.05**
   - May indicate insufficient sample size
   - Check if protein panel is relevant to this risk stratification
   - Consider that absence of significance is also a valid result

5. **Cramer's V is low despite significant p-value**
   - Large sample size can produce significant p-values for small effects
   - Report both p-value and effect size
   - Clinical significance may differ from statistical significance

---

## Limitations

### Statistical Limitations
- Chi-square test requires adequate sample size (typically n ≥ 30)
- Expected frequencies should be ≥ 5 in most cells
- Cannot establish causality (only association)
- Multiple testing increases false positive rate

### Biological Limitations
- Protein expression is a snapshot (not dynamic)
- Post-translational modifications not captured
- Protein levels may not reflect activity
- Tumor heterogeneity within samples not captured

### Interpretation Considerations
- Correlation ≠ causation
- Expression differences don't prove mechanistic link
- Other unmeasured proteins may be more important
- Need validation in independent cohorts

---

## Citation

If you use this analysis pipeline in your research, please cite:

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

- **v1.0** (2024): Initial release with chi-square analysis, effect size calculation, and publication-quality visualizations

---

## References

**Statistical Methods**:
- Chi-square test: Pearson, K. (1900). "On the criterion that a given system of deviations..."
- Cramer's V: Cramér, H. (1946). "Mathematical Methods of Statistics"

**Protein Functions**:
- SUCLG1/SUCLG2: Krebs cycle intermediates
- IDH3B: Isocitrate dehydrogenase family
- ACO2: Aconitase, mitochondrial metabolism
- CYCS: Cytochrome c, apoptosis regulator
- VDAC2: Mitochondrial voltage-dependent anion channel

---

## Acknowledgments

- Proteomic data acquisition team
- Biostatistics consultation
- Pathology department for sample preparation
