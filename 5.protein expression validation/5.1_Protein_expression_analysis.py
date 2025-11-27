import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import warnings
import os
warnings.filterwarnings('ignore')

# Set font for English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set style
sns.set_style("whitegrid")

# Create output directory if it doesn't exist
OUTPUT_DIR = '/results/protein_expression_analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_preprocess_data(filepath):
    """Load and preprocess data"""
    # Read data
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Ensure column names are correct
    print("Basic Data Information:")
    print(f"Total samples: {len(df)}")
    print(f"Risk group distribution:")
    print(df['Risk_Group_Clinical'].value_counts().sort_index())
    print("\nM_stage distribution:")
    print(df.groupby('Risk_Group_Clinical')['M_stage'].value_counts())

    return df

def encode_protein_expression(df, protein_cols):
    """Encode protein expression to numerical values"""
    # Define encoding mapping
    expression_map = {
        'Negative': 0,
        'Weak Positive': 1,
        'Moderate Positive': 2,
        'Strong Positive': 3
    }

    # Create encoded columns
    encoded_cols = {}
    for col in protein_cols:
        encoded_col = f"{col}_encoded"
        df[encoded_col] = df[col].map(expression_map)
        encoded_cols[col] = encoded_col

    return df, encoded_cols

def perform_chi_square_test(df, protein_col, risk_col='Risk_Group_Clinical'):
    """Perform chi-square test for a single protein"""
    # Create crosstab
    crosstab = pd.crosstab(df[risk_col], df[protein_col])

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(crosstab)

    # Calculate effect size (Cramer's V)
    n = crosstab.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))

    return {
        'protein': protein_col,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'crosstab': crosstab
    }

def analyze_all_proteins(df, protein_cols):
    """Analyze expression differences for all proteins"""
    results = []
    print("\n=== Protein Expression Differential Statistical Analysis ===\n")

    for protein in protein_cols:
        result = perform_chi_square_test(df, protein)
        results.append(result)

        print(f"{protein}:")
        print(f"  Chi-square: {result['chi2']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Cramer's V: {result['cramers_v']:.4f}")
        print(f"  Significance: {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print()

    return results

def plot_protein_expression_heatmap(df, protein_cols):
    """Plot protein expression heatmap with statistical annotations"""
    # First, perform chi-square tests to get p-values
    test_results = {}
    for protein in protein_cols:
        result = perform_chi_square_test(df, protein)
        test_results[protein] = result

    # Calculate the proportion of protein expression in each risk group
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, protein in enumerate(protein_cols):
        ax = axes[idx]

        # Create proportion table
        prop_table = pd.crosstab(df['Risk_Group_Clinical'], df[protein], normalize='index') * 100

        # Plot stacked bar chart
        prop_table.plot(kind='bar', stacked=True, ax=ax,
                        colormap='RdYlBu_r', width=0.8)

        # Get statistical values
        chi2 = test_results[protein]['chi2']
        p_value = test_results[protein]['p_value']
        cramers_v = test_results[protein]['cramers_v']

        # Determine significance
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        # Add title with statistical info
        ax.set_title(f'{protein} Expression Distribution\nχ²={chi2:.2f}, p={p_value:.4f} {sig_text}, V={cramers_v:.3f}',
                     fontsize=12, pad=10)
        ax.set_xlabel('Risk Group', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(title='Expression Level', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Add a colored background based on significance
        if p_value < 0.05:
            ax.patch.set_facecolor('#ffeeee')  # Light red background for significant
            ax.patch.set_alpha(0.3)

    plt.suptitle('Protein Expression Patterns Across Risk Groups\n(χ²: Chi-square statistic, p: p-value, V: Cramer\'s V effect size)',
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'protein_expression_by_risk_group.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_significance_summary(results):
    """Plot significance summary with detailed explanations"""
    # Prepare data
    proteins = [r['protein'] for r in results]
    p_values = [r['p_value'] for r in results]
    cramers_v = [r['cramers_v'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # P-value bar chart
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    bars1 = ax1.barh(proteins, -np.log10(p_values), color=colors)
    ax1.axvline(x=-np.log10(0.05), color='black', linestyle='--', label='P=0.05 threshold', linewidth=2)
    ax1.set_xlabel('-log10(P-value)\n(Higher values = More significant)', fontsize=12)
    ax1.set_title('Statistical Significance of Protein Expression Differences\n(Red bars: p<0.05, Gray bars: p≥0.05)', fontsize=14)
    ax1.legend(loc='lower right')

    # Add P-value labels
    for i, (bar, p) in enumerate(zip(bars1, p_values)):
        if p < 0.001:
            label = '***'
        elif p < 0.01:
            label = '**'
        elif p < 0.05:
            label = '*'
        else:
            label = 'ns'
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{label} (p={p:.4f})', va='center', fontsize=9)

    # Add interpretation note
    ax1.text(0.02, 0.02,
             'Interpretation:\n- Bars crossing the black line are significant (p<0.05)\n- Longer bars indicate stronger evidence against null hypothesis',
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Cramer's V effect size plot
    bars2 = ax2.barh(proteins, cramers_v, color='steelblue')
    ax2.set_xlabel("Cramer's V (Effect Size)\n(0-1 scale: Higher values = Stronger association)", fontsize=12)
    ax2.set_title('Effect Size of Protein Expression Differences\n(Strength of association with risk groups)', fontsize=14)

    # Add effect size interpretation
    ax2.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Small Effect (V≥0.1)')
    ax2.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='Medium Effect (V≥0.3)')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Large Effect (V≥0.5)')
    ax2.legend(loc='lower right')

    # Add values on bars
    for i, (bar, v) in enumerate(zip(bars2, cramers_v)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=9)

    # Add interpretation guide
    ax2.text(0.02, 0.02,
             'Effect Size Guide:\n- Small: V<0.3 (weak association)\n- Medium: 0.3≤V<0.5 (moderate association)\n- Large: V≥0.5 (strong association)',
             transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Protein Expression Statistical Analysis Summary\n' +
                 'Left: Statistical significance (p-values) | Right: Effect sizes (strength of association)',
                 fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'protein_significance_summary.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print interpretation guide
    print("\n=== How to Read the Significance Summary Plots ===")
    print("\nLeft Panel (-log10 P-value):")
    print("- This shows the statistical significance of differences in protein expression across risk groups")
    print("- The x-axis shows -log10(p-value), so higher values mean MORE significant")
    print("- Red bars: Statistically significant (p<0.05)")
    print("- Gray bars: Not statistically significant (p≥0.05)")
    print("- Black dashed line: p=0.05 threshold")
    print("- Significance levels: *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant)")
    print("\nRight Panel (Cramer's V):")
    print("- This shows the STRENGTH of association between protein expression and risk groups")
    print("- Values range from 0 (no association) to 1 (perfect association)")
    print("- Even if statistically significant, the actual effect might be small")
    print("- Effect size interpretation:")
    print("  • Small effect: V < 0.3")
    print("  • Medium effect: 0.3 ≤ V < 0.5")
    print("  • Large effect: V ≥ 0.5")
    print("\nKey Point: A protein can be statistically significant but have a small effect size!")
    print("="*60)

def create_expression_heatmap_matrix(df, protein_cols, encoded_cols):
    """Create a comprehensive expression heatmap matrix"""
    fig, ax = plt.subplots(figsize=(12, 6))  # Reduced height for 2 groups

    # Calculate mean expression levels for each protein in each risk group
    heatmap_data = pd.DataFrame(index=['Low', 'High'],  # Only Low and High
                                columns=protein_cols)

    for risk in heatmap_data.index:
        risk_data = df[df['Risk_Group_Clinical'] == risk]
        for protein in protein_cols:
            # Use encoded values to calculate mean
            encoded_col = encoded_cols[protein]
            mean_val = risk_data[encoded_col].mean()
            heatmap_data.loc[risk, protein] = mean_val

    # Convert to numeric
    heatmap_data = heatmap_data.astype(float)

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Mean Expression Level\n(0=Negative, 1=Weak, 2=Moderate, 3=Strong)'},
                vmin=0, vmax=3, ax=ax)

    ax.set_title('Mean Protein Expression Levels by Risk Group', fontsize=16, pad=20)
    ax.set_xlabel('Proteins', fontsize=12)
    ax.set_ylabel('Risk Groups', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'protein_expression_heatmap_matrix.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    return heatmap_data

def create_expression_pattern_matrix(df, protein_cols):
    """Create expression pattern matrix"""
    # Calculate dominant expression pattern for each risk group
    pattern_matrix = pd.DataFrame(index=['Low', 'High'],  # Only Low and High
                                  columns=protein_cols)

    for risk in pattern_matrix.index:
        risk_data = df[df['Risk_Group_Clinical'] == risk]
        for protein in protein_cols:
            # Find the most common expression level
            mode_expression = risk_data[protein].mode()[0] if len(risk_data[protein].mode()) > 0 else 'Unknown'
            pattern_matrix.loc[risk, protein] = mode_expression

    return pattern_matrix

def plot_correlation_heatmap(df, protein_cols, encoded_cols):
    """Plot protein expression correlation heatmap"""
    # Calculate correlations for different risk groups
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Only 2 plots for Low and High
    risk_groups = ['Low', 'High']  # Only Low and High

    for idx, risk in enumerate(risk_groups):
        risk_data = df[df['Risk_Group_Clinical'] == risk]

        # Use encoded data to calculate correlation
        encoded_protein_cols = [encoded_cols[p] for p in protein_cols]
        corr_matrix = risk_data[encoded_protein_cols].corr()

        # Rename columns and index
        corr_matrix.columns = protein_cols
        corr_matrix.index = protein_cols

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=axes[idx],
                    cbar_kws={"shrink": 0.8})
        axes[idx].set_title(f'{risk} Risk Group', fontsize=14)

    plt.suptitle('Protein Expression Correlations by Risk Group', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'protein_correlation_by_risk.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df, results, pattern_matrix):
    """Generate summary report"""
    report = []
    report.append("="*60)
    report.append("Proteomic Expression and Risk Stratification Association Analysis Report")
    report.append("="*60)
    report.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Samples: {len(df)}")

    report.append("\nRisk Group Distribution:")
    for risk, count in df['Risk_Group_Clinical'].value_counts().sort_index().items():
        report.append(f"  {risk}: {count} ({count/len(df)*100:.1f}%)")

    report.append("\n" + "="*60)
    report.append("Statistical Test Results Summary")
    report.append("="*60)

    # Significant proteins
    sig_proteins = [r for r in results if r['p_value'] < 0.05]
    report.append(f"\nNumber of Significantly Different Proteins: {len(sig_proteins)}/{len(results)}")

    if sig_proteins:
        report.append("\nSignificantly Different Proteins:")
        for r in sorted(sig_proteins, key=lambda x: x['p_value']):
            report.append(f"  - {r['protein']}: P={r['p_value']:.4f}, Cramer's V={r['cramers_v']:.3f}")

    # Non-significant proteins
    non_sig_proteins = [r for r in results if r['p_value'] >= 0.05]
    if non_sig_proteins:
        report.append("\nNon-significantly Different Proteins:")
        for r in non_sig_proteins:
            report.append(f"  - {r['protein']}: P={r['p_value']:.4f}")

    report.append("\n" + "="*60)
    report.append("Dominant Expression Patterns by Risk Group")
    report.append("="*60)
    report.append("\n" + pattern_matrix.to_string())

    report.append("\n" + "="*60)
    report.append("Clinical Significance Interpretation")
    report.append("="*60)

    if sig_proteins:
        report.append("\nThe following proteins show significant expression differences across risk groups:")
        for r in sig_proteins:
            if r['cramers_v'] > 0.5:
                effect = "strong"
            elif r['cramers_v'] > 0.3:
                effect = "moderate"
            else:
                effect = "weak"
            report.append(f"- {r['protein']} shows {effect} association")

    # Save report
    report_text = '\n'.join(report)
    with open(os.path.join(OUTPUT_DIR, 'protein_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text

def main():
    """Main function"""
    # File path
    filepath = '/data/Metastasis_associated_protein_expression_data.csv'

    # Define protein column names
    protein_cols = ['SUCLG1', 'DLAT', 'IDH3B', 'ACSF2', 'SUCLG2', 'ACO2', 'CYCS', 'VDAC2']

    print("Starting proteomic expression data analysis...")

    # 1. Load data
    df = load_and_preprocess_data(filepath)

    # 2. Encode protein expression
    df, encoded_cols = encode_protein_expression(df, protein_cols)

    # 3. Statistical analysis
    results = analyze_all_proteins(df, protein_cols)

    # 4. Visualization analysis
    print("\nGenerating visualization charts...")

    # Protein expression distribution plot
    plot_protein_expression_heatmap(df, protein_cols)

    # Significance summary plot
    plot_significance_summary(results)

    # Correlation analysis
    plot_correlation_heatmap(df, protein_cols, encoded_cols)

    # Expression heatmap matrix
    heatmap_data = create_expression_heatmap_matrix(df, protein_cols, encoded_cols)

    # Save heatmap data
    heatmap_data.to_csv(os.path.join(OUTPUT_DIR, 'protein_mean_expression_by_risk.csv'))

    # 5. Create expression pattern matrix
    pattern_matrix = create_expression_pattern_matrix(df, protein_cols)

    # 6. Generate summary report
    report = generate_summary_report(df, results, pattern_matrix)
    print("\n" + report)

    # 7. Save detailed results
    # Save statistical results
    results_df = pd.DataFrame([{
        'Protein': r['protein'],
        'Chi-square': r['chi2'],
        'P-value': r['p_value'],
        'Cramer\'s V': r['cramers_v'],
        'Significance': '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
    } for r in results])
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'protein_statistical_results.csv'), index=False)

    # Create a detailed summary table
    summary_df = results_df.copy()
    summary_df['Effect_Size'] = summary_df['Cramer\'s V'].apply(
        lambda x: 'Large' if x >= 0.5 else 'Medium' if x >= 0.3 else 'Small' if x >= 0.1 else 'Negligible'
    )
    summary_df['Significant'] = summary_df['P-value'] < 0.05
    summary_df = summary_df.sort_values('P-value')
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'protein_analysis_summary.csv'), index=False)

    # Save crosstabs
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'protein_crosstabs.xlsx')) as writer:
        for r in results:
            r['crosstab'].to_excel(writer, sheet_name=r['protein'])

    print("\nAnalysis completed!")
    print(f"Generated files in {OUTPUT_DIR}:")
    print("- protein_expression_by_risk_group.pdf (with statistical annotations)")
    print("- protein_significance_summary.pdf (with interpretation guide)")
    print("- protein_correlation_by_risk.pdf")
    print("- protein_expression_heatmap_matrix.pdf (mean expression levels)")
    print("- protein_analysis_report.txt")
    print("- protein_statistical_results.csv")
    print("- protein_analysis_summary.csv (detailed summary)")
    print("- protein_mean_expression_by_risk.csv")
    print("- protein_crosstabs.xlsx")

if __name__ == "__main__":
    main()
