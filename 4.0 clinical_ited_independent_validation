import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

# 设置输出目录
OUTPUT_DIR = 'F:/Habitat_radiomics/Publication_Results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置绘图风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# ================== Part 1: 数据加载 ==================

def load_complete_data(dataset_type='zlyy'):
    """加载完整数据，包含所有临床变量"""
    print(f"\n=== Loading {dataset_type.upper()} Dataset ===")

    # 加载临床数据
    clinical_file = f'F:/Habitat_radiomics/clinical_features_processed_{dataset_type}.csv'
    clinical_data = pd.read_csv(clinical_file)

    # 加载iTED特征
    ited_file = f'F:/Habitat_radiomics/iTED_features_{dataset_type}.csv'
    ited_features = pd.read_csv(ited_file)

    # 加载Radiomics特征
    radiomics_file = f'F:/Habitat_radiomics/radiomics_features_{dataset_type}.csv'
    radiomics_features = pd.read_csv(radiomics_file)

    # 加载3D_ITHscore
    ithscore_file = f'F:/Habitat_radiomics/3D_ITHscore_{dataset_type}.csv'
    ithscore_data = pd.read_csv(ithscore_file)

    # 合并数据
    data = clinical_data.copy()

    # 合并iTED特征
    ited_cols = [col for col in ited_features.columns if col != 'PatientID']
    data = pd.merge(data, ited_features, on='PatientID', how='inner')

    # 合并Radiomics特征
    radiomics_cols = [col for col in radiomics_features.columns if col != 'PatientID']
    data = pd.merge(data, radiomics_features, on='PatientID', how='inner')

    # 合并3D_ITHscore
    data = pd.merge(data, ithscore_data, on='PatientID', how='inner')

    # 重置索引
    data = data.reset_index(drop=True)

    # 获取所有临床变量（排除PatientID和结局变量）
    clinical_vars = [col for col in clinical_data.columns
                    if col not in ['PatientID', 'M_stage']]

    print(f"Loaded data shape: {data.shape}")
    print(f"Number of clinical variables: {len(clinical_vars)}")
    print(f"M1 rate: {(data['M_stage'] == 1).mean():.1%}")

    return {
        'full_data': data,
        'clinical_vars': clinical_vars,
        'ited_features': ited_cols,
        'radiomics_features': radiomics_cols,
        'outcome': 'M_stage'
    }

# ================== Part 2: 创建风险分组（修改版） ==================

def create_risk_groups_from_predictions(data, dataset_type, feature_type):
    """从预测文件创建风险分组"""
    print(f"\n=== Processing {feature_type} ===")

    # 根据特征类型加载对应的预测文件
    if feature_type == 'iTED':
        pred_file = f'F:/Habitat_radiomics/results/iTED_predictions_all_patients_{dataset_type}.csv'
    elif feature_type == 'Radiomics':
        pred_file = f'F:/Habitat_radiomics/results/Radiomics_predictions_all_patients_{dataset_type}.csv'
    elif feature_type == '3D_ITH':
        # 3D_ITHscore直接使用原始值
        pred_file = f'F:/Habitat_radiomics/3D_ITHscore_{dataset_type}.csv'
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # 读取预测数据
    pred_data = pd.read_csv(pred_file)

    # 获取预测值
    if feature_type in ['iTED', 'Radiomics']:
        # 第3列是Predicted_Probability
        pred_data = pred_data[['PatientID', 'Predicted_Probability']]
        pred_data.columns = ['PatientID', 'predictions']
    else:  # 3D_ITH
        # 第2列是3D_ITHscore
        pred_data = pred_data[['PatientID', '3D_ITHscore']]
        pred_data.columns = ['PatientID', 'predictions']

    # 合并预测值到主数据
    data = pd.merge(data, pred_data, on='PatientID', how='left')

    # 创建得分列
    score_col = f'{feature_type}_Score'
    risk_col = f'{feature_type}_Risk'

    data[score_col] = data['predictions']

    # 计算最优阈值（使用有标签的数据）
    y = data['M_stage']
    predictions = data['predictions']

    # 移除缺失值
    mask = ~(predictions.isnull() | y.isnull())
    y_clean = y[mask]
    predictions_clean = predictions[mask]

    if len(y_clean) > 0:
        # 计算AUC
        auc = roc_auc_score(y_clean, predictions_clean)

        # 使用最优阈值
        fpr, tpr, thresholds = roc_curve(y_clean, predictions_clean)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds[optimal_idx]

        # 创建风险分组
        data[risk_col] = 'Low'
        data.loc[data[score_col] >= threshold, risk_col] = 'High'

        print(f"AUC: {auc:.3f}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Risk distribution - Low: {(data[risk_col] == 'Low').sum()}, High: {(data[risk_col] == 'High').sum()}")

    # 删除临时predictions列
    data = data.drop('predictions', axis=1)

    return data

# ================== Part 3: 综合单因素分析 ==================

def perform_comprehensive_univariate_analysis(data, clinical_vars):
    """对所有变量进行单因素分析"""

    # 定义分类变量及其参考类别
    categorical_vars_config = {
        'Sex': {'categories': [0, 1], 'labels': ['Female', 'Male'], 'reference': 0},
        'Benign_thyroid_lesions': {'categories': [0, 1], 'labels': ['No', 'Yes'], 'reference': 0},
        'Multifocal': {'categories': [0, 1], 'labels': ['No', 'Yes'], 'reference': 0},
        'Infiltrated_the_adjacent_tissue': {'categories': [0, 1], 'labels': ['No', 'Yes'], 'reference': 0},
        'T_stage': {'categories': ['T1', 'T2', 'T3', 'T4'], 'labels': ['T1', 'T2', 'T3', 'T4'], 'reference': 'T1'},
        'N_stage': {'categories': ['N0', 'N1'], 'labels': ['N0', 'N1'], 'reference': 'N0'},
        'iTED_Risk': {'categories': ['Low', 'High'], 'labels': ['Low', 'High'], 'reference': 'Low'},
        'Radiomics_Risk': {'categories': ['Low', 'High'], 'labels': ['Low', 'High'], 'reference': 'Low'},
        '3D_ITH_Risk': {'categories': ['Low', 'High'], 'labels': ['Low', 'High'], 'reference': 'Low'}
    }

    results = []
    y = data['M_stage']

    # 所有待分析变量
    score_vars = ['iTED_Risk', 'Radiomics_Risk', '3D_ITH_Risk']
    all_vars = score_vars + clinical_vars

    for var in all_vars:
        if var not in data.columns:
            continue

        try:
            # 处理分类变量
            if var in categorical_vars_config:
                config = categorical_vars_config[var]
                var_data = data[var].copy()

                # 获取实际存在的类别
                unique_vals = var_data.dropna().unique()

                # 先添加参考类别 - 确保所有分类变量都有参考类别
                reference = config['reference']
                ref_label = config['labels'][config['categories'].index(reference)] if reference in config['categories'] else str(reference)

                # 无论数据中是否存在参考类别，都添加到结果中
                results.append({
                    'Variable': var,
                    'Category': ref_label,
                    'OR': 1.0,
                    'OR_CI': 'Reference',
                    'P_value': np.nan,
                    'Beta': 0.0,
                    'N': (~var_data.isnull()).sum(),
                    'Events': y[var_data == reference].sum() if reference in unique_vals else 0,
                    'IsReference': True
                })

                # 分析非参考类别
                for val in unique_vals:
                    if val == reference:
                        continue

                    # 创建二分类变量
                    X = (var_data == val).astype(float)

                    # 移除缺失值
                    mask = ~(X.isnull() | y.isnull() | var_data.isnull())
                    X_clean = X[mask].values.reshape(-1, 1)
                    y_clean = y[mask].values

                    if len(y_clean) < 20 or y_clean.sum() < 5:
                        continue

                    # 逻辑回归
                    lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
                    lr.fit(X_clean, y_clean)

                    # 计算统计量
                    coef = lr.coef_[0][0]
                    or_val = np.exp(coef)

                    # Bootstrap CI
                    n_bootstrap = 200
                    bootstrap_coefs = []

                    for _ in range(n_bootstrap):
                        idx = np.random.choice(len(y_clean), len(y_clean), replace=True)
                        X_boot = X_clean[idx]
                        y_boot = y_clean[idx]

                        try:
                            lr_boot = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
                            lr_boot.fit(X_boot, y_boot)
                            bootstrap_coefs.append(lr_boot.coef_[0][0])
                        except:
                            continue

                    if len(bootstrap_coefs) > 50:
                        ci_lower = np.exp(np.percentile(bootstrap_coefs, 2.5))
                        ci_upper = np.exp(np.percentile(bootstrap_coefs, 97.5))

                        se = np.std(bootstrap_coefs)
                        z_score = coef / (se + 1e-8)
                        p_val = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

                        # 获取类别标签
                        if val in config['categories']:
                            cat_label = config['labels'][config['categories'].index(val)]
                        else:
                            cat_label = str(val)

                        results.append({
                            'Variable': var,
                            'Category': cat_label,
                            'OR': or_val,
                            'OR_CI': f"{or_val:.2f} ({ci_lower:.2f}, {ci_upper:.2f})",
                            'CI_Lower': ci_lower,
                            'CI_Upper': ci_upper,
                            'P_value': p_val,
                            'Beta': coef,
                            'N': len(y_clean),
                            'Events': y_clean.sum(),
                            'IsReference': False
                        })

            else:
                # 连续变量
                X = data[[var]]

                # 移除缺失值
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X_clean = X[mask]
                y_clean = y[mask]

                if len(y_clean) < 20 or y_clean.sum() < 5:
                    continue

                # 标准化
                X_std = (X_clean - X_clean.mean()) / (X_clean.std() + 1e-8)

                lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
                lr.fit(X_std, y_clean)

                # 计算统计量
                coef = lr.coef_[0][0]
                or_val = np.exp(coef)

                # Bootstrap CI
                n_bootstrap = 200
                bootstrap_coefs = []

                for _ in range(n_bootstrap):
                    idx = np.random.choice(len(y_clean), len(y_clean), replace=True)
                    X_boot = X_std.iloc[idx]
                    y_boot = y_clean.iloc[idx]

                    try:
                        lr_boot = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
                        lr_boot.fit(X_boot, y_boot)
                        bootstrap_coefs.append(lr_boot.coef_[0][0])
                    except:
                        continue

                if len(bootstrap_coefs) > 50:
                    ci_lower = np.exp(np.percentile(bootstrap_coefs, 2.5))
                    ci_upper = np.exp(np.percentile(bootstrap_coefs, 97.5))

                    se = np.std(bootstrap_coefs)
                    z_score = coef / (se + 1e-8)
                    p_val = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

                    results.append({
                        'Variable': var,
                        'Category': '',
                        'OR': or_val,
                        'OR_CI': f"{or_val:.2f} ({ci_lower:.2f}, {ci_upper:.2f})",
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'P_value': p_val,
                        'Beta': coef,
                        'N': len(y_clean),
                        'Events': y_clean.sum(),
                        'IsReference': False
                    })

        except Exception as e:
            print(f"Error analyzing {var}: {str(e)[:100]}")

    return pd.DataFrame(results)

# ================== Part 4: 多因素分析（全面策略） ==================

def perform_multivariate_comprehensive(data, uni_results):
    """策略2：三个评分 + 所有单因素显著的变量（P<0.1）"""

    # 获取显著变量（P<0.1，排除参考类别）
    significant_df = uni_results[(uni_results['P_value'] < 0.1) & (~uni_results['IsReference'])]
    significant_vars = significant_df['Variable'].unique()

    # 确保三个评分都包含
    core_scores = ['iTED_Risk', 'Radiomics_Risk', '3D_ITH_Risk']
    for score in core_scores:
        if score not in significant_vars and score in data.columns:
            significant_vars = np.append(significant_vars, score)

    print(f"\nSignificant variables for multivariate analysis: {list(significant_vars)}")

    # 定义分类变量配置
    categorical_vars_config = {
        'Sex': {'categories': [0, 1], 'labels': ['Female', 'Male'], 'reference': 0},
        'Benign_thyroid_lesions': {'categories': [0, 1], 'labels': ['No', 'Yes'], 'reference': 0},
        'Multifocal': {'categories': [0, 1], 'labels': ['No', 'Yes'], 'reference': 0},
        'Infiltrated_the_adjacent_tissue': {'categories': [0, 1], 'labels': ['No', 'Yes'], 'reference': 0},
        'T_stage': {'categories': ['T1', 'T2', 'T3', 'T4'], 'labels': ['T1', 'T2', 'T3', 'T4'], 'reference': 'T1'},
        'N_stage': {'categories': ['N0', 'N1'], 'labels': ['N0', 'N1'], 'reference': 'N0'},
        'iTED_Risk': {'categories': ['Low', 'High'], 'labels': ['Low', 'High'], 'reference': 'Low'},
        'Radiomics_Risk': {'categories': ['Low', 'High'], 'labels': ['Low', 'High'], 'reference': 'Low'},
        '3D_ITH_Risk': {'categories': ['Low', 'High'], 'labels': ['Low', 'High'], 'reference': 'Low'}
    }

    results = []

    try:
        # 准备数据
        X_data = pd.DataFrame()
        var_mapping = []

        for var in significant_vars:
            if var not in data.columns:
                continue

            if var in categorical_vars_config:
                config = categorical_vars_config[var]
                var_data = data[var].copy()

                # 获取唯一值
                unique_vals = var_data.dropna().unique()

                # 创建哑变量（排除参考类别）
                for val in unique_vals:
                    if val == config['reference']:
                        continue

                    col_name = f"{var}_{val}"
                    X_data[col_name] = (var_data == val).astype(float)

                    # 获取类别标签
                    if val in config['categories']:
                        cat_label = config['labels'][config['categories'].index(val)]
                    else:
                        cat_label = str(val)

                    var_mapping.append((var, cat_label, col_name))
            else:
                # 连续变量
                col_name = f'{var}_std'
                X_data[col_name] = (data[var] - data[var].mean()) / (data[var].std() + 1e-8)
                var_mapping.append((var, '', col_name))

        if X_data.shape[1] == 0:
            return pd.DataFrame()

        y = data['M_stage']

        # 移除缺失值
        mask = ~(X_data.isnull().any(axis=1) | y.isnull())
        X_clean = X_data[mask]
        y_clean = y[mask]

        print(f"\nMultivariate model:")
        print(f"  Variables included: {len(var_mapping)}")
        print(f"  Sample size: {len(y_clean)}, Events: {y_clean.sum()}")

        # 逻辑回归（增加正则化）
        lr = LogisticRegression(penalty='l2', C=0.5, solver='liblinear',
                               class_weight='balanced', max_iter=1000)
        lr.fit(X_clean, y_clean)

        # 首先添加所有参考类别 - 包括所有分类变量，不仅是显著的
        all_categorical_vars = [v for v in significant_vars if v in categorical_vars_config]

        # 确保包含所有在数据中的分类变量
        for var in categorical_vars_config.keys():
            if var in data.columns and var not in all_categorical_vars:
                all_categorical_vars.append(var)

        for var in all_categorical_vars:
            if var in categorical_vars_config:
                config = categorical_vars_config[var]
                ref_label = config['labels'][config['categories'].index(config['reference'])] \
                           if config['reference'] in config['categories'] else str(config['reference'])

                results.append({
                    'Variable': var,
                    'Category': ref_label,
                    'OR': 1.0,
                    'OR_CI': 'Reference',
                    'P_value': np.nan,
                    'Beta': 0.0,
                    'N': len(y_clean),
                    'Events': y_clean.sum(),
                    'IsReference': True
                })

        # 获取非参考类别的结果
        for i, (var_name, category, col_name) in enumerate(var_mapping):
            coef = lr.coef_[0][i]
            or_val = np.exp(coef)

            # Bootstrap CI
            n_bootstrap = 200
            bootstrap_coefs = []

            for _ in range(n_bootstrap):
                idx = np.random.choice(len(y_clean), len(y_clean), replace=True)
                X_boot = X_clean.iloc[idx]
                y_boot = y_clean.iloc[idx]

                try:
                    lr_boot = LogisticRegression(penalty='l2', C=0.5, solver='liblinear',
                                                class_weight='balanced')
                    lr_boot.fit(X_boot, y_boot)
                    bootstrap_coefs.append(lr_boot.coef_[0][i])
                except:
                    continue

            if len(bootstrap_coefs) > 50:
                ci_lower = np.exp(np.percentile(bootstrap_coefs, 2.5))
                ci_upper = np.exp(np.percentile(bootstrap_coefs, 97.5))

                se = np.std(bootstrap_coefs)
                z_score = coef / (se + 1e-8)
                p_val = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

                results.append({
                    'Variable': var_name,
                    'Category': category,
                    'OR': or_val,
                    'OR_CI': f"{or_val:.2f} ({ci_lower:.2f}, {ci_upper:.2f})",
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'P_value': p_val,
                    'Beta': coef,
                    'N': len(y_clean),
                    'Events': y_clean.sum(),
                    'IsReference': False
                })

    except Exception as e:
        print(f"Error in multivariate analysis: {e}")

    return pd.DataFrame(results)

# ================== Part 5: 生成格式化表格 ==================

def format_combined_results_table(uni_results, multi_results, dataset_name):
    """生成格式化的结果表格"""

    # 定义分类变量的所有类别（用于确保完整性）
    categorical_vars_structure = {
        'iTED_Risk': ['Low', 'High'],
        'Radiomics_Risk': ['Low', 'High'],
        '3D_ITH_Risk': ['Low', 'High'],
        'Sex': ['Female', 'Male'],
        'Benign_thyroid_lesions': ['No', 'Yes'],
        'Multifocal': ['No', 'Yes'],
        'Infiltrated_the_adjacent_tissue': ['No', 'Yes'],
        'T_stage': ['T1', 'T2', 'T3', 'T4'],
        'N_stage': ['N0', 'N1']
    }

    combined = []

    # 获取所有变量（保持顺序）
    all_vars_uni = uni_results['Variable'].unique()
    all_vars_multi = multi_results['Variable'].unique()
    all_vars = list(dict.fromkeys(list(all_vars_uni) + list(all_vars_multi)))

    # 定义变量顺序（评分在前，临床变量在后）
    score_vars = ['iTED_Risk', 'Radiomics_Risk', '3D_ITH_Risk']
    clinical_vars = [v for v in all_vars if v not in score_vars]
    ordered_vars = score_vars + clinical_vars

    # 按变量处理
    for var in ordered_vars:
        if var not in all_vars:
            continue

        # 如果是分类变量，确保按预定义顺序显示所有类别
        if var in categorical_vars_structure:
            categories_to_show = categorical_vars_structure[var]

            for category in categories_to_show:
                # 获取单因素结果
                uni_row = uni_results[(uni_results['Variable'] == var) &
                                      (uni_results['Category'] == category)]

                # 获取多因素结果
                multi_row = multi_results[(multi_results['Variable'] == var) &
                                          (multi_results['Category'] == category)]

                result_row = {
                    'Characteristic': var,
                    'Category': category
                }

                # 添加单因素结果
                if not uni_row.empty:
                    uni_row = uni_row.iloc[0]
                    if uni_row.get('IsReference', False):
                        result_row['Univariate_OR'] = 'Reference'
                        result_row['Univariate_P'] = ''
                    else:
                        result_row['Univariate_OR'] = uni_row.get('OR_CI', '')
                        p_val = uni_row.get('P_value', np.nan)
                        if pd.notna(p_val):
                            result_row['Univariate_P'] = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                        else:
                            result_row['Univariate_P'] = ''
                else:
                    result_row['Univariate_OR'] = ''
                    result_row['Univariate_P'] = ''

                # 添加多因素结果
                if not multi_row.empty:
                    multi_row = multi_row.iloc[0]
                    if multi_row.get('IsReference', False):
                        result_row['Beta'] = ''
                        result_row['Multivariate_OR'] = 'Reference'
                        result_row['Multivariate_P'] = ''
                    else:
                        beta = multi_row.get('Beta', np.nan)
                        result_row['Beta'] = f"{beta:.3f}" if pd.notna(beta) else ''
                        result_row['Multivariate_OR'] = multi_row.get('OR_CI', '')
                        p_val = multi_row.get('P_value', np.nan)
                        if pd.notna(p_val):
                            result_row['Multivariate_P'] = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                        else:
                            result_row['Multivariate_P'] = ''
                else:
                    result_row['Beta'] = ''
                    result_row['Multivariate_OR'] = ''
                    result_row['Multivariate_P'] = ''

                combined.append(result_row)
        else:
            # 连续变量
            # 获取单因素结果
            uni_row = uni_results[(uni_results['Variable'] == var) &
                                  (uni_results['Category'] == '')]

            # 获取多因素结果
            multi_row = multi_results[(multi_results['Variable'] == var) &
                                      (multi_results['Category'] == '')]

            result_row = {
                'Characteristic': var,
                'Category': ''
            }

            # 添加单因素结果
            if not uni_row.empty:
                uni_row = uni_row.iloc[0]
                result_row['Univariate_OR'] = uni_row.get('OR_CI', '')
                p_val = uni_row.get('P_value', np.nan)
                if pd.notna(p_val):
                    result_row['Univariate_P'] = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                else:
                    result_row['Univariate_P'] = ''
            else:
                result_row['Univariate_OR'] = ''
                result_row['Univariate_P'] = ''

            # 添加多因素结果
            if not multi_row.empty:
                multi_row = multi_row.iloc[0]
                beta = multi_row.get('Beta', np.nan)
                result_row['Beta'] = f"{beta:.3f}" if pd.notna(beta) else ''
                result_row['Multivariate_OR'] = multi_row.get('OR_CI', '')
                p_val = multi_row.get('P_value', np.nan)
                if pd.notna(p_val):
                    result_row['Multivariate_P'] = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                else:
                    result_row['Multivariate_P'] = ''
            else:
                result_row['Beta'] = ''
                result_row['Multivariate_OR'] = ''
                result_row['Multivariate_P'] = ''

            combined.append(result_row)

    # 创建DataFrame
    df = pd.DataFrame(combined)

    # 去除重复的变量名（保持类别缩进效果）
    prev_var = None
    for i, row in df.iterrows():
        if row['Characteristic'] == prev_var:
            df.at[i, 'Characteristic'] = ''
        else:
            prev_var = row['Characteristic']

    return df

# ================== Part 6: 生成森林图 ==================

def create_forest_plot(results_df, title, output_path, analysis_type='univariate'):
    """生成顶刊风格的森林图"""

    # 筛选非参考类别
    plot_data = results_df[~results_df['IsReference']].copy()

    # 处理缺失的CI值
    if 'CI_Lower' not in plot_data.columns or 'CI_Upper' not in plot_data.columns:
        return

    plot_data = plot_data.dropna(subset=['CI_Lower', 'CI_Upper', 'OR'])

    if plot_data.empty:
        print(f"No data available for forest plot: {title}")
        return

    # 创建显示标签
    plot_data['Display_Label'] = plot_data.apply(
        lambda x: f"{x['Variable']} ({x['Category']})" if x['Category'] else x['Variable'],
        axis=1
    )

    # 按OR值排序
    plot_data = plot_data.sort_values('OR', ascending=True)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.4)))

    # 设置y轴位置
    y_positions = np.arange(len(plot_data))

    # 绘制置信区间
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        # 置信区间线
        ax.plot([row['CI_Lower'], row['CI_Upper']], [i, i],
               'k-', linewidth=1.5, alpha=0.7)

        # OR点
        color = 'red' if row['P_value'] < 0.05 else 'black'
        ax.scatter(row['OR'], i, s=100, c=color, zorder=3, edgecolors='black', linewidth=0.5)

    # 添加参考线（OR=1）
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 设置标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data['Display_Label'])

    # 设置x轴（对数刻度）
    ax.set_xscale('log')
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11, fontweight='bold')

    # 设置标题
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    # 调整布局
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # 添加网格
    ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='P < 0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='P ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)

    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Forest plot saved: {output_path}")

# ================== Part 7: 主执行函数 ==================

def main():
    """主执行函数"""

    print("="*80)
    print("COMPREHENSIVE ANALYSIS PIPELINE - PUBLICATION STANDARD")
    print("Strategy: All scores + significant clinical variables (P<0.1)")
    print("Output: Tables and Forest Plots (NEJM/JAMA style)")
    print("="*80)

    # 数据集
    datasets = [('zlyy', 'Training'), ('ydyy', 'Validation')]

    # 存储所有结果用于汇总
    all_results = []

    for dataset_type, dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} Dataset")
        print(f"{'='*80}")

        # 1. 加载数据
        data_info = load_complete_data(dataset_type)
        full_data = data_info['full_data']
        clinical_vars = data_info['clinical_vars']

        print(f"\nClinical variables: {clinical_vars}")

        # 2. 创建所有风险分组（从预测文件）
        print("\n--- Creating risk groups from predictions ---")

        # iTED
        full_data = create_risk_groups_from_predictions(full_data, dataset_type, 'iTED')

        # Radiomics
        full_data = create_risk_groups_from_predictions(full_data, dataset_type, 'Radiomics')

        # 3D_ITH
        full_data = create_risk_groups_from_predictions(full_data, dataset_type, '3D_ITH')

        # 3. 执行综合单因素分析
        print(f"\n--- Univariate Analysis ---")
        uni_results = perform_comprehensive_univariate_analysis(full_data, clinical_vars)

        if uni_results.empty:
            print(f"No results for {dataset_name}")
            continue

        # 显示单因素显著的变量
        significant = uni_results[(uni_results['P_value'] < 0.05) & (~uni_results['IsReference'])]
        print(f"\nSignificant variables (P<0.05): {len(significant)}")
        for _, row in significant.head(10).iterrows():
            cat_info = f" ({row['Category']})" if row['Category'] else ""
            print(f"  {row['Variable']}{cat_info}: OR={row.get('OR', np.nan):.2f}, P={row['P_value']:.3f}")

        # 4. 执行多因素分析
        print(f"\n--- Multivariate Analysis ---")
        multi_results = perform_multivariate_comprehensive(full_data, uni_results)

        # 5. 生成表格
        table = format_combined_results_table(uni_results, multi_results, dataset_name)

        # 保存表格
        filename = f"Combined_Analysis_{dataset_name}_Comprehensive.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        table.to_csv(filepath, index=False)
        print(f"✓ Table saved: {filename}")

        # 6. 生成森林图
        # 单因素森林图
        forest_uni_path = os.path.join(OUTPUT_DIR, f"Forest_Univariate_{dataset_name}.png")
        create_forest_plot(uni_results,
                          f"Univariate Analysis - {dataset_name} Dataset",
                          forest_uni_path,
                          'univariate')

        # 多因素森林图
        forest_multi_path = os.path.join(OUTPUT_DIR, f"Forest_Multivariate_{dataset_name}.png")
        create_forest_plot(multi_results,
                          f"Multivariate Analysis - {dataset_name} Dataset",
                          forest_multi_path,
                          'multivariate')

        # 收集结果用于汇总
        all_results.append({
            'dataset': dataset_name,
            'uni_results': uni_results,
            'multi_results': multi_results
        })

    # 7. 生成汇总报告
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    summary_data = []

    for result in all_results:
        dataset_name = result['dataset']
        uni_results = result['uni_results']
        multi_results = result['multi_results']

        # 提取三个评分的结果
        for score in ['iTED_Risk', 'Radiomics_Risk', '3D_ITH_Risk']:
            # 单因素结果
            uni_high = uni_results[(uni_results['Variable'] == score) &
                                   (uni_results['Category'] == 'High')]

            # 多因素结果
            multi_high = multi_results[(multi_results['Variable'] == score) &
                                      (multi_results['Category'] == 'High')]

            if not uni_high.empty and not multi_high.empty:
                uni_high = uni_high.iloc[0]
                multi_high = multi_high.iloc[0]

                summary_data.append({
                    'Dataset': dataset_name,
                    'Score': score.replace('_Risk', ''),
                    'Univariate_OR': uni_high.get('OR_CI', '-'),
                    'Univariate_P': f"{uni_high['P_value']:.3f}" if uni_high['P_value'] >= 0.001 else "<0.001",
                    'Multivariate_OR': multi_high.get('OR_CI', '-'),
                    'Multivariate_P': f"{multi_high['P_value']:.3f}" if multi_high['P_value'] >= 0.001 else "<0.001"
                })

    # 保存汇总
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'Summary_Three_Scores.csv'), index=False)
        print("✓ Summary report saved")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)

    print("\n📂 Files generated:")
    print(f"  Location: {OUTPUT_DIR}")
    print("\n  Tables:")
    print("  • Combined_Analysis_Training_Comprehensive.csv")
    print("  • Combined_Analysis_Validation_Comprehensive.csv")
    print("\n  Forest Plots:")
    print("  • Forest_Univariate_Training.png")
    print("  • Forest_Univariate_Validation.png")
    print("  • Forest_Multivariate_Training.png")
    print("  • Forest_Multivariate_Validation.png")
    print("\n  Summary:")
    print("  • Summary_Three_Scores.csv")

    print("\n" + "="*80)
    print("Publication Ready:")
    print("- Tables show all categories with clear reference groups")
    print("- Forest plots follow NEJM/JAMA style guidelines")
    print("- P<0.05 highlighted in red in forest plots")
    print("="*80)

# 运行主程序
if __name__ == "__main__":
    main()
