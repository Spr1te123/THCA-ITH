import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """加载训练集和外部验证集数据"""
    print("Loading datasets...")

    # 加载数据
    train_data = pd.read_csv('F:/Habitat_radiomics/clinical_features_zlyy.csv')
    external_data = pd.read_csv('F:/Habitat_radiomics/clinical_features_ydyy.csv')

    print(f"Training set size: {len(train_data)}")
    print(f"External validation set size: {len(external_data)}")

    return train_data, external_data


def categorize_variables(data):
    """识别连续变量和分类变量"""
    continuous_vars = []
    categorical_vars = []

    # 预定义的分类变量
    known_categorical = ['Sex', 'Benign_thyroid_lesions', 'Multifocal',
                         'Infiltrated_the_adjacent_tissue', 'T_stage', 'N_stage',
                         'M_stage', 'True_M_stage']

    for col in data.columns:
        if col == 'PatientID':
            continue

        if col in known_categorical:
            categorical_vars.append(col)
        elif data[col].nunique() <= 5:  # 少于等于5个唯一值视为分类变量
            categorical_vars.append(col)
        else:
            continuous_vars.append(col)

    return continuous_vars, categorical_vars


def calculate_continuous_stats(data, var):
    """计算连续变量的统计信息"""
    values = data[var].dropna()

    if len(values) == 0:
        return "N/A"

    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    # 根据数据范围决定显示格式
    if median > 100:
        return f"{median:.0f} ({q1:.0f}-{q3:.0f})"
    elif median > 10:
        return f"{median:.1f} ({q1:.1f}-{q3:.1f})"
    else:
        return f"{median:.2f} ({q1:.2f}-{q3:.2f})"


def calculate_categorical_stats(data, var):
    """计算分类变量的统计信息"""
    value_counts = data[var].value_counts()
    total = len(data[var].dropna())

    if total == 0:
        return {"N/A": "0 (0)"}

    stats_dict = {}
    for value, count in value_counts.items():
        percentage = (count / total) * 100
        stats_dict[str(value)] = f"{count} ({percentage:.0f})"

    return stats_dict


def perform_statistical_test(train_data, external_data, var, var_type):
    """执行统计检验"""
    train_values = train_data[var].dropna()
    external_values = external_data[var].dropna()

    if len(train_values) == 0 or len(external_values) == 0:
        return np.nan

    try:
        if var_type == 'continuous':
            # Mann-Whitney U检验（非参数检验）
            _, p_value = mannwhitneyu(train_values, external_values)
        else:
            # 卡方检验
            # 创建列联表
            all_categories = pd.concat([train_values, external_values]).unique()

            # 创建计数表
            train_counts = train_values.value_counts()
            external_counts = external_values.value_counts()

            # 构建列联表
            contingency_table = []
            for cat in all_categories:
                train_count = train_counts.get(cat, 0)
                external_count = external_counts.get(cat, 0)
                contingency_table.append([train_count, external_count])

            contingency_table = np.array(contingency_table).T

            # 执行卡方检验
            _, p_value, _, _ = chi2_contingency(contingency_table)

    except Exception as e:
        print(f"Error calculating p-value for {var}: {e}")
        p_value = np.nan

    return p_value


def format_p_value(p_value):
    """格式化p值"""
    if pd.isna(p_value):
        return "N/A"
    elif p_value < 0.001:
        return "<.001"
    elif p_value < 0.01:
        return f"{p_value:.3f}"
    elif p_value < 0.05:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.2f}"


def create_comparison_table(train_data, external_data):
    """创建比较表格"""
    print("\nCreating comparison table...")

    # 识别变量类型
    continuous_vars, categorical_vars = categorize_variables(train_data)

    # 存储结果
    results = []

    # 添加样本量
    results.append({
        'Characteristic': 'Sample size, n',
        'Training Data Set': str(len(train_data)),
        'External Validation Set': str(len(external_data)),
        'P Value': ''
    })

    # 处理连续变量
    print("\nProcessing continuous variables...")
    for var in continuous_vars:
        if var in train_data.columns and var in external_data.columns:
            train_stats = calculate_continuous_stats(train_data, var)
            external_stats = calculate_continuous_stats(external_data, var)
            p_value = perform_statistical_test(train_data, external_data, var, 'continuous')

            results.append({
                'Characteristic': f"{var}*",
                'Training Data Set': train_stats,
                'External Validation Set': external_stats,
                'P Value': format_p_value(p_value)
            })

    # 处理分类变量
    print("\nProcessing categorical variables...")
    for var in categorical_vars:
        if var in train_data.columns and var in external_data.columns:
            train_stats = calculate_categorical_stats(train_data, var)
            external_stats = calculate_categorical_stats(external_data, var)
            p_value = perform_statistical_test(train_data, external_data, var, 'categorical')

            # 首先添加变量名和p值
            results.append({
                'Characteristic': var,
                'Training Data Set': '',
                'External Validation Set': '',
                'P Value': format_p_value(p_value)
            })

            # 然后添加每个类别的统计
            all_categories = set(list(train_stats.keys()) + list(external_stats.keys()))
            for category in sorted(all_categories):
                results.append({
                    'Characteristic': f"  {category}",
                    'Training Data Set': train_stats.get(category, "0 (0)"),
                    'External Validation Set': external_stats.get(category, "0 (0)"),
                    'P Value': ''
                })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def add_table_notes(df):
    """添加表格注释"""
    notes = []
    notes.append("Note.—Data are numbers of patients, with percentages in parentheses, unless otherwise indicated.")
    notes.append("* Data are medians, with IQRs in parentheses.")
    notes.append("P values represent the comparison between training and external validation data sets.")
    notes.append("Mann-Whitney U test was used for continuous variables.")
    notes.append("Chi-square test was used for categorical variables.")

    return notes


def save_results(results_df, notes):
    """保存结果"""
    output_file = 'F:/Habitat_radiomics/clinical_comparison_table.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 写入主表
        results_df.to_excel(writer, sheet_name='Comparison Table', index=False)

        # 获取工作表
        worksheet = writer.sheets['Comparison Table']

        # 调整列宽
        worksheet.column_dimensions['A'].width = 35
        worksheet.column_dimensions['B'].width = 20
        worksheet.column_dimensions['C'].width = 25
        worksheet.column_dimensions['D'].width = 10

        # 添加注释
        start_row = len(results_df) + 3
        for i, note in enumerate(notes):
            worksheet.cell(row=start_row + i, column=1, value=note)

    # 同时保存CSV版本
    csv_file = 'F:/Habitat_radiomics/clinical_comparison_table.csv'
    results_df.to_csv(csv_file, index=False)

    print(f"\nResults saved to:")
    print(f"  - Excel: {output_file}")
    print(f"  - CSV: {csv_file}")

    return output_file, csv_file


def generate_summary_statistics(train_data, external_data):
    """生成汇总统计信息"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # M分期分布（如果存在）
    if 'True_M_stage' in train_data.columns:
        train_m1_rate = (train_data['True_M_stage'] == 1).mean() * 100
        external_m1_rate = (external_data['True_M_stage'] == 1).mean() * 100

        print(f"\nM1 Rate:")
        print(f"  Training set: {train_m1_rate:.1f}%")
        print(f"  External validation set: {external_m1_rate:.1f}%")

    # 缺失值统计
    print("\nMissing Values Summary:")
    print("\nTraining set:")
    train_missing = train_data.isnull().sum()
    train_missing = train_missing[train_missing > 0]
    if len(train_missing) > 0:
        for col, count in train_missing.items():
            print(f"  {col}: {count} ({count / len(train_data) * 100:.1f}%)")
    else:
        print("  No missing values")

    print("\nExternal validation set:")
    external_missing = external_data.isnull().sum()
    external_missing = external_missing[external_missing > 0]
    if len(external_missing) > 0:
        for col, count in external_missing.items():
            print(f"  {col}: {count} ({count / len(external_data) * 100:.1f}%)")
    else:
        print("  No missing values")


def main():
    """主函数"""
    print("Clinical Variables Comparison Analysis")
    print("=" * 60)

    # 加载数据
    train_data, external_data = load_and_prepare_data()

    # 创建比较表格
    results_df = create_comparison_table(train_data, external_data)

    # 添加注释
    notes = add_table_notes(results_df)

    # 保存结果
    excel_file, csv_file = save_results(results_df, notes)

    # 生成汇总统计
    generate_summary_statistics(train_data, external_data)

    # 显示部分结果
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (First 20 rows)")
    print("=" * 60)
    print(results_df.head(20).to_string(index=False))

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
