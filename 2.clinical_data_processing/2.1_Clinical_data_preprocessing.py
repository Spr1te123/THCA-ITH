import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
import os

warnings.filterwarnings('ignore')

# 创建结果目录
results_dir = '/results/clinical_data_preprocess_result'
os.makedirs(results_dir, exist_ok=True)

# 设置绘图风格和中文显示
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据加载和初步探索 =====================
print("=" * 60)
print("1. 数据加载和初步探索")
print("=" * 60)

# 加载数据
df = pd.read_csv('/data/clinical_features_ydyy.csv')
print(f"数据维度: {df.shape}")

# 分离特征和标签
patient_id = df['PatientID']
y = df['M_stage'].map({'M0': 0, 'M1': 1})
X_clinical = df.iloc[:, 2:26]

# 结局变量分布
print(f"\n结局变量分布:")
print(f"M0 (无转移): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"M1 (有转移): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")


# ===================== 2. 识别变量类型 =====================
def identify_variable_types(df, unique_threshold=10):
    """识别数值型和分类型变量"""
    numeric_vars = []
    categorical_vars = []

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            unique_values = df[col].nunique()
            if unique_values < unique_threshold:
                categorical_vars.append(col)
            else:
                numeric_vars.append(col)
        else:
            categorical_vars.append(col)

    return numeric_vars, categorical_vars


numeric_vars, categorical_vars = identify_variable_types(X_clinical)
print(f"\n数值型变量 ({len(numeric_vars)}): {numeric_vars}")
print(f"分类型变量 ({len(categorical_vars)}): {categorical_vars}")

# ===================== 3. 缺失值分析和可视化 =====================
print("\n" + "=" * 60)
print("2. 缺失值分析")
print("=" * 60)

# 缺失值统计
missing_stats = pd.DataFrame({
    'Variable': X_clinical.columns,
    'Missing_Count': X_clinical.isnull().sum(),
    'Missing_Percentage': (X_clinical.isnull().sum() / len(X_clinical) * 100).round(2)
}).sort_values('Missing_Percentage', ascending=False)

print("\n缺失值统计:")
print(missing_stats[missing_stats['Missing_Count'] > 0])

# 图1: 缺失值百分比条形图（SCI论文常用）
fig, ax = plt.subplots(figsize=(10, 6))

missing_pct = (X_clinical.isnull().sum() / len(X_clinical) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

if len(missing_pct) > 0:
    # 创建颜色渐变
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(missing_pct)))

    # 绘制条形图
    bars = ax.bar(range(len(missing_pct)), missing_pct.values, color=colors, edgecolor='black', linewidth=0.5)

    # 添加阈值线
    ax.axhline(y=30, color='red', linestyle='--', alpha=0.7, linewidth=2, label='30% threshold')
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='20% threshold')

    # 设置标签
    ax.set_xticks(range(len(missing_pct)))
    ax.set_xticklabels(missing_pct.index, rotation=45, ha='right')
    ax.set_ylabel('Missing Percentage (%)', fontsize=14)
    ax.set_xlabel('Variables', fontsize=14)
    ax.set_title('Percentage of Missing Values by Variable', fontsize=16, fontweight='bold', pad=20)

    # 添加数值标签
    for i, (bar, pct) in enumerate(zip(bars, missing_pct.values)):
        ax.text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(missing_pct.values) * 1.1 if len(missing_pct) > 0 else 40)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'missing_percentage.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# 图2: 缺失值模式热图（SCI论文常用）
fig, ax = plt.subplots(figsize=(12, 8))

# 创建缺失值矩阵（1表示缺失，0表示存在）
missing_matrix = X_clinical.isnull().astype(int)

# 只显示有缺失值的变量
vars_with_missing = missing_matrix.columns[missing_matrix.sum() > 0]
if len(vars_with_missing) > 0:
    missing_matrix_subset = missing_matrix[vars_with_missing]

    # 绘制热图
    sns.heatmap(missing_matrix_subset.T,
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Missing (1) vs Present (0)'},
                yticklabels=vars_with_missing,
                xticklabels=False,
                ax=ax)

    ax.set_xlabel('Samples', fontsize=14)
    ax.set_ylabel('Variables', fontsize=14)
    ax.set_title('Missing Value Pattern Heatmap', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'missing_pattern_heatmap.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# ===================== 4. 剔除高缺失率变量 =====================
print("\n" + "=" * 60)
print("3. 剔除高缺失率变量")
print("=" * 60)


def remove_high_missing_vars(df, threshold=0.3):
    """剔除缺失率超过阈值的变量"""
    missing_pct = df.isnull().sum() / len(df)
    keep_vars = missing_pct[missing_pct <= threshold].index.tolist()
    removed_vars = missing_pct[missing_pct > threshold].index.tolist()

    if removed_vars:
        print(f"\n剔除缺失率>{threshold * 100}%的变量:")
        for var in removed_vars:
            print(f"  - {var}: {missing_pct[var] * 100:.1f}%")
    else:
        print(f"\n没有变量的缺失率超过{threshold * 100}%")

    return df[keep_vars], removed_vars


X_clinical_filtered, removed_vars = remove_high_missing_vars(X_clinical, threshold=0.3)
print(f"\n剩余变量数: {X_clinical_filtered.shape[1]}")

# 更新变量列表
numeric_vars_filtered = [var for var in numeric_vars if var in X_clinical_filtered.columns]
categorical_vars_filtered = [var for var in categorical_vars if var in X_clinical_filtered.columns]

# ===================== 5. 异常值检测和处理 =====================
print("\n" + "=" * 60)
print("4. 异常值检测和处理")
print("=" * 60)


def detect_and_handle_outliers(df, numeric_vars, method='IQR'):
    """检测和处理数值变量的异常值"""
    df_clean = df.copy()
    outlier_report = {}

    for var in numeric_vars:
        if var not in df.columns:
            continue

        data = df[var].dropna()

        if len(data) == 0:
            continue

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (data < lower_bound) | (data > upper_bound)
        n_outliers = outliers.sum()

        outlier_report[var] = {
            'n_outliers': n_outliers,
            'pct_outliers': n_outliers / len(data) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        # Winsorization: 将异常值替换为边界值
        if n_outliers > 0:
            df_clean.loc[df_clean[var] < lower_bound, var] = lower_bound
            df_clean.loc[df_clean[var] > upper_bound, var] = upper_bound

    return df_clean, outlier_report


# 处理异常值
X_clinical_clean, outlier_report = detect_and_handle_outliers(
    X_clinical_filtered,
    numeric_vars_filtered
)

# 异常值处理前后对比可视化
if numeric_vars_filtered:
    # 选择有异常值的变量进行展示
    vars_with_outliers = [var for var, info in outlier_report.items()
                          if info['n_outliers'] > 0]

    # 如果没有异常值，展示前5个数值变量
    vars_to_plot = vars_with_outliers if vars_with_outliers else numeric_vars_filtered[:5]

    if vars_to_plot:
        n_vars = len(vars_to_plot)
        fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))

        if n_vars == 1:
            axes = axes.reshape(1, -1)

        for i, var in enumerate(vars_to_plot):
            if var not in X_clinical_filtered.columns:
                continue

            # 处理前的箱线图
            ax1 = axes[i, 0]
            data_before = X_clinical_filtered[var].dropna()

            if len(data_before) > 0:
                # 箱线图
                box1 = ax1.boxplot([data_before], vert=True, patch_artist=True,
                                   boxprops=dict(facecolor='#ff9999', edgecolor='black'),
                                   medianprops=dict(color='red', linewidth=2),
                                   whiskerprops=dict(color='black'),
                                   capprops=dict(color='black'),
                                   flierprops=dict(marker='o', markerfacecolor='red',
                                                   markersize=8, markeredgecolor='black',
                                                   alpha=0.7))

                # 添加统计信息
                ax1.set_title(f'{var} - Before Outlier Treatment', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Value', fontsize=12)
                ax1.grid(True, alpha=0.3)

                # 统计文本
                n_outliers = outlier_report[var]['n_outliers']
                outlier_text = f'Outliers: {n_outliers} ({outlier_report[var]["pct_outliers"]:.1f}%)\n'
                outlier_text += f'Mean: {data_before.mean():.2f}\nStd: {data_before.std():.2f}'
                ax1.text(0.95, 0.95, outlier_text, transform=ax1.transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                         fontsize=10)

            # 处理后的箱线图
            ax2 = axes[i, 1]
            data_after = X_clinical_clean[var].dropna()

            if len(data_after) > 0:
                # 箱线图
                box2 = ax2.boxplot([data_after], vert=True, patch_artist=True,
                                   boxprops=dict(facecolor='#99ccff', edgecolor='black'),
                                   medianprops=dict(color='red', linewidth=2),
                                   whiskerprops=dict(color='black'),
                                   capprops=dict(color='black'),
                                   flierprops=dict(marker='o', markerfacecolor='blue',
                                                   markersize=8, markeredgecolor='black',
                                                   alpha=0.7))

                # 添加统计信息
                ax2.set_title(f'{var} - After Outlier Treatment', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Value', fontsize=12)
                ax2.grid(True, alpha=0.3)

                # 统计文本
                stats_text = f'Treatment: Winsorization\n'
                stats_text += f'Mean: {data_after.mean():.2f}\nStd: {data_after.std():.2f}'
                ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                         fontsize=10)

                # 添加处理边界线（如果有异常值）
                if outlier_report[var]['n_outliers'] > 0:
                    ax1.axhline(outlier_report[var]['lower_bound'],
                                color='green', linestyle='--', alpha=0.7,
                                label='Treatment bounds')
                    ax1.axhline(outlier_report[var]['upper_bound'],
                                color='green', linestyle='--', alpha=0.7)
                    ax1.legend(loc='lower right', fontsize=10)

            # 确保两个子图的y轴范围一致，便于比较
            y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'outlier_treatment_comparison.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n生成异常值处理对比图，包含 {len(vars_to_plot)} 个变量")
else:
    print("\n没有数值变量需要进行异常值处理")

# 打印异常值报告
print("\n异常值检测报告:")
total_outliers = 0
for var, info in outlier_report.items():
    if info['n_outliers'] > 0:
        print(f"{var}: {info['n_outliers']}个异常值 ({info['pct_outliers']:.1f}%)")
        total_outliers += info['n_outliers']

if total_outliers == 0:
    print("未检测到异常值")
else:
    print(f"\n总计检测到 {total_outliers} 个异常值，已使用Winsorization方法处理")

# ===================== 6. 缺失值插补（多重插补） =====================
print("\n" + "=" * 60)
print("5. 缺失值插补（多重插补 MICE）")
print("=" * 60)


def multiple_imputation(df, numeric_vars, categorical_vars):
    """使用多重插补处理缺失值"""
    df_imputed = df.copy()

    # 数值变量：使用MICE（多重插补）
    if numeric_vars and any(df[numeric_vars].isnull().sum() > 0):
        mice_imputer = IterativeImputer(
            random_state=42,
            max_iter=10,
            verbose=0
        )
        df_imputed[numeric_vars] = mice_imputer.fit_transform(df[numeric_vars])
        print(f"完成数值变量的多重插补 ({len(numeric_vars)}个变量)")

    # 分类变量：使用众数插补
    if categorical_vars and any(df[categorical_vars].isnull().sum() > 0):
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_vars] = cat_imputer.fit_transform(df[categorical_vars])
        print(f"完成分类变量的众数插补 ({len(categorical_vars)}个变量)")

    return df_imputed


# 执行插补
X_clinical_imputed = multiple_imputation(
    X_clinical_clean,
    numeric_vars_filtered,
    categorical_vars_filtered
)

# 验证插补结果
remaining_missing = X_clinical_imputed.isnull().sum().sum()
print(f"\n插补后剩余缺失值总数: {remaining_missing}")

# ===================== 7. 插补前后对比可视化 =====================
# 选择前5个数值变量进行可视化
vars_to_plot = numeric_vars_filtered[:5] if len(numeric_vars_filtered) > 5 else numeric_vars_filtered

if vars_to_plot:
    n_vars = len(vars_to_plot)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for i, var in enumerate(vars_to_plot):
        if var not in X_clinical_clean.columns:
            continue

        # 插补前
        ax1 = axes[i, 0]
        data_before = X_clinical_clean[var].dropna()
        if len(data_before) > 0:
            ax1.hist(data_before, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
            ax1.set_title(f'{var} - Before Imputation', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Value', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, alpha=0.3)

            # 添加统计信息
            stats_text = f'Mean: {data_before.mean():.2f}\nStd: {data_before.std():.2f}\nN: {len(data_before)}'
            ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 插补后
        ax2 = axes[i, 1]
        data_after = X_clinical_imputed[var]
        ax2.hist(data_after, bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
        ax2.set_title(f'{var} - After Imputation', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Mean: {data_after.mean():.2f}\nStd: {data_after.std():.2f}\nN: {len(data_after)}'
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'imputation_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# ===================== 8. 保存处理后的数据和报告 =====================
print("\n" + "=" * 60)
print("6. 保存结果")
print("=" * 60)

# 合并所有数据
df_processed = pd.DataFrame({
    'PatientID': patient_id,
    'M_stage': y
})
df_processed = pd.concat([df_processed, X_clinical_imputed], axis=1)

# 保存处理后的数据
output_file = os.path.join(results_dir, 'clinical_features_processed.csv')
df_processed.to_csv(output_file, index=False)
print(f"处理后的数据已保存至: {output_file}")

# 生成数据处理报告
report = f"""
临床数据预处理报告
==========================================
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 数据概况
------------------------------------------
原始数据维度: {df.shape}
处理后数据维度: {df_processed.shape}

结局变量分布:
  - M0 (无转移): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)
  - M1 (有转移): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)

2. 变量类型
------------------------------------------
数值型变量: {len(numeric_vars)}个
分类型变量: {len(categorical_vars)}个

3. 缺失值处理
------------------------------------------
剔除的高缺失率变量 (>30%): {len(removed_vars)}个
"""
if removed_vars:
    report += f"  {', '.join(removed_vars)}\n"

report += f"""
4. 异常值处理
------------------------------------------
异常值处理方法: IQR方法 + Winsorization
检测到异常值的变量:"""

for var, info in outlier_report.items():
    if info['n_outliers'] > 0:
        report += f"\n  - {var}: {info['n_outliers']}个 ({info['pct_outliers']:.1f}%)"

report += f"""

5. 缺失值插补
------------------------------------------
插补方法: 
  - 数值变量: 多重插补（MICE）
  - 分类变量: 众数插补
数值变量插补: {len(numeric_vars_filtered)}个
分类变量插补: {len(categorical_vars_filtered)}个

6. 最终结果
------------------------------------------
最终保留变量数: {X_clinical_imputed.shape[1]}
处理后数据无缺失值: {'是' if remaining_missing == 0 else '否'}
"""

print("\n" + report)

# 保存报告
report_file = os.path.join(results_dir, 'preprocessing_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"处理报告已保存至: {report_file}")

print("\n" + "=" * 60)
print("数据预处理完成！")
print("=" * 60)

# 在代码最后的说明部分更新
print("\n生成的图片文件：")
print(f"- {os.path.join(results_dir, 'missing_percentage.pdf')}：缺失值百分比图")
print(f"- {os.path.join(results_dir, 'missing_pattern_heatmap.pdf')}：缺失值模式图")
print(f"- {os.path.join(results_dir, 'outlier_treatment_comparison.pdf')}：异常值处理前后对比图")
print(f"- {os.path.join(results_dir, 'imputation_comparison.pdf')}：缺失值插补前后对比图")
