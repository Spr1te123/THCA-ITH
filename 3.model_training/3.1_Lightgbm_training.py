import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve, auc,
                             f1_score, accuracy_score, recall_score,
                             precision_score, brier_score_loss,
                             balanced_accuracy_score, matthews_corrcoef,
                             fbeta_score, make_scorer)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SVMSMOTE
import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna
from optuna.visualization import plot_optimization_history
import shap
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score

# 额外导入的库
from sklearn.utils import resample
import joblib
from math import pi
import os
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler

# 创建结果目录
results_dir = '/results/lightgbm_model_results'
os.makedirs(results_dir, exist_ok=True)

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

print("=" * 80)
print("多模态特征融合LightGBM模型构建 - 甲状腺癌远处转移预测")
print("改进版：处理类别不平衡问题")
print("=" * 80)


def calculate_balanced_metrics(y_true, y_pred, y_proba):
    """计算适合极度不平衡数据的评估指标"""
    from imblearn.metrics import geometric_mean_score
    from sklearn.metrics import fbeta_score

    # 基础指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 避免除零
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 计算更适合不平衡数据的指标
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'g_mean': geometric_mean_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'auc_pr': average_precision_score(y_true, y_proba),
        'auc_roc': roc_auc_score(y_true, y_proba),
        'brier_score': brier_score_loss(y_true, y_proba),
        'recall_minority': sensitivity,
        'detection_rate': tp / len(y_true),  # 新增：检出率
        'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred),
        'f2_score': fbeta_score(y_true, y_pred, beta=2),
        'f3_score': fbeta_score(y_true, y_pred, beta=3),  # 新增：F3-Score
        'ppv': precision,  # 阳性预测值
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # 阴性预测值
        'lr_positive': sensitivity / (1 - specificity) if specificity < 1 else np.inf,  # 阳性似然比
        'lr_negative': (1 - sensitivity) / specificity if specificity > 0 else 0,  # 阴性似然比（新增重点）
        'dor': (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf,  # 诊断优势比
        'informedness': sensitivity + specificity - 1,  # Youden指数
        'markedness': precision + (tn / (tn + fn) if (tn + fn) > 0 else 0) - 1,  # 标记性
    }

    return metrics


# ===================== 新增辅助函数 =====================
def calculate_stability_asymmetric(internal_val, external_val, metric_name=''):
    """
    非对称的稳定性评分
    - 轻微提升（≤10%）：高分（好现象）
    - 中等提升（10-20%）：中高分（需关注）
    - 大幅提升（>20%）：中低分（异常，需调查）
    - 轻微下降（≤10%）：中高分（可接受）
    - 中等下降（10-20%）：中分（需要注意）
    - 大幅下降（>20%）：低分（过拟合）
    """

    if internal_val <= 0:
        return 0.5  # 无法计算时给中等分

    # 计算相对变化（正值=提升，负值=下降）
    relative_change = (external_val - internal_val) / internal_val

    # 非对称评分函数
    if relative_change >= 0:  # 外部优于内部
        if relative_change <= 0.10:  # ≤10% 提升
            # 轻微提升是好的，给高分
            stability = 0.9 + relative_change  # 0.9-1.0
        elif relative_change <= 0.20:  # 10-20% 提升
            # 中等提升需要关注，给中高分
            stability = 0.8 - (relative_change - 0.10) * 2  # 0.8-0.6
        else:  # >20% 提升
            # 大幅提升可能有问题，给中低分
            stability = max(0.3, 0.6 - (relative_change - 0.20))
    else:  # 外部低于内部
        abs_change = abs(relative_change)
        if abs_change <= 0.10:  # ≤10% 下降
            # 轻微下降可接受，给中高分
            stability = 0.85 - abs_change * 1.5  # 0.85-0.70
        elif abs_change <= 0.20:  # 10-20% 下降
            # 中等下降需注意，给中分
            stability = 0.70 - (abs_change - 0.10) * 3  # 0.70-0.40
        else:  # >20% 下降
            # 大幅下降表示过拟合，给低分
            stability = max(0.1, 0.40 - (abs_change - 0.20) * 2)

    # 对于筛查工具的关键指标，调整评分策略
    if metric_name in ['sensitivity', 'npv']:
        # 对这些指标，下降的惩罚更严重
        if relative_change < 0:
            stability *= 0.8  # 额外20%惩罚

    return stability


def interpret_stability_score(stability_score, change_percent, metric_name):
    """
    解释稳定性得分的含义
    """
    if change_percent > 0:  # 外部优于内部
        if change_percent <= 10:
            interpretation = f"优秀 - {metric_name}轻微提升{change_percent:.1f}%（良好泛化）"
        elif change_percent <= 20:
            interpretation = f"良好 - {metric_name}中等提升{change_percent:.1f}%（需验证原因）"
        else:
            interpretation = f"异常 - {metric_name}大幅提升{change_percent:.1f}%（需调查）"
    else:  # 外部低于内部
        change_percent = abs(change_percent)
        if change_percent <= 10:
            interpretation = f"良好 - {metric_name}轻微下降{change_percent:.1f}%（可接受）"
        elif change_percent <= 20:
            interpretation = f"一般 - {metric_name}中等下降{change_percent:.1f}%（轻微过拟合）"
        else:
            interpretation = f"较差 - {metric_name}大幅下降{change_percent:.1f}%（明显过拟合）"

    return interpretation

def optimize_threshold_for_metric(y_true, y_proba, metric='f2'):
    """找到最优化特定指标的阈值"""
    from imblearn.metrics import geometric_mean_score

    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'g_mean':
            if len(np.unique(y_pred)) > 1:
                score = geometric_mean_score(y_true, y_pred)
            else:
                score = 0
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'youden':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            score = 0

        scores.append(score)

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return best_threshold, best_score


def optimize_threshold_for_screening(y_true, y_proba, min_sensitivity=0.90, min_npv=0.95):
    """专门为筛查工具优化阈值"""
    best_threshold = 0.5
    best_score = -np.inf

    thresholds = np.arange(0.05, 0.50, 0.01)

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 必须满足筛查要求
        if sensitivity >= min_sensitivity and npv >= min_npv:
            # 综合评分
            score = sensitivity * 0.5 + npv * 0.3 + specificity * 0.2
            if score > best_score:
                best_score = score
                best_threshold = threshold

    # 如果没有满足要求的，选择敏感性最高的
    if best_threshold == 0.5:
        best_threshold = thresholds[np.argmax([
            recall_score(y_true, (y_proba >= t).astype(int))
            for t in thresholds
        ])]

    return best_threshold, best_score

# ===================== 1. 数据加载和整合 =====================
print("\n1. 数据加载和整合")
print("-" * 60)


# 定义读取CSV的函数，自动检测编码
def read_csv_with_encoding(filepath):
    """尝试不同的编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']

    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取: {os.path.basename(filepath)}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时出错 ({encoding}): {str(e)}")
            continue

    raise ValueError(f"无法读取文件 {filepath}，尝试了所有编码: {encodings}")


# 加载所有数据
try:
    clinical_df = read_csv_with_encoding('/data/clinical_features_processed_zlyy.csv')
    radiomics_df = read_csv_with_encoding('/data/radiomics_features_zlyy.csv')
    iTED_df = read_csv_with_encoding('/data/iTED_features_zlyy.csv')
    ITHscore_df = read_csv_with_encoding('/data/3D_ITHscore_zlyy.csv')
except Exception as e:
    print(f"数据加载失败: {str(e)}")
    raise

print(f"\n数据维度:")
print(f"临床特征: {clinical_df.shape}")
print(f"传统影像组学特征: {radiomics_df.shape}")
print(f"iTED特征: {iTED_df.shape}")
print(f"3D_ITHscore: {ITHscore_df.shape}")

# 检查PatientID是否一致
print("\n检查PatientID一致性...")
patient_ids_clinical = set(clinical_df['PatientID'])
patient_ids_radiomics = set(radiomics_df['PatientID'])
patient_ids_iTED = set(iTED_df['PatientID'])
patient_ids_ITH = set(ITHscore_df['PatientID'])

# 找出共同的PatientID
common_patients = patient_ids_clinical & patient_ids_radiomics & patient_ids_iTED & patient_ids_ITH
print(f"共同患者数: {len(common_patients)}")

if len(common_patients) < len(patient_ids_clinical):
    print(f"警告: 有 {len(patient_ids_clinical) - len(common_patients)} 个患者在某些特征文件中缺失")

# 合并所有数据
df_merged = clinical_df.merge(radiomics_df, on='PatientID', how='inner')
df_merged = df_merged.merge(iTED_df, on='PatientID', how='inner')
df_merged = df_merged.merge(ITHscore_df, on='PatientID', how='inner')

print(f"\n合并后数据维度: {df_merged.shape}")
print(f"最终患者数量: {len(df_merged)}")

# 提取标签
y = df_merged['M_stage'].values
patient_ids = df_merged['PatientID'].values

# 定义各类特征的列名
clinical_feature_names = ['Sex', 'Age', 'BMI', 'Benign_thyroid_lesions', 'Multifocal',
                         'Tumor_size', 'Infiltrated_the_adjacent_tissue',
                         'Number_of_metastatic_lymph_nodes', 'T_stage', 'N_stage',
                         'WBC', 'RBC', 'PLT', 'HGB', 'LYM', 'MONO', 'NE', 'EOS',
                         'BASO', 'NLR', 'PLR', 'LMR', 'SII', 'TSH', 'TG', 'TGAb', 'TPOAb', 'GLU']

# 获取各数据框的特征列名（排除PatientID）
radiomics_feature_names = [col for col in radiomics_df.columns if col not in ['PatientID']]
iTED_feature_names = [col for col in iTED_df.columns if col not in ['PatientID']]

# 基于列名分离不同类型的特征
clinical_features = df_merged[clinical_feature_names].copy()
radiomics_features = df_merged[radiomics_feature_names].copy()
iTED_features = df_merged[iTED_feature_names].copy()
ITHscore_feature = df_merged[['3D_ITHscore']].copy()

# 更新特征名称列表
clinical_names = clinical_features.columns.tolist()
radiomics_names = radiomics_features.columns.tolist()
iTED_names = iTED_features.columns.tolist()

# ========== 在这里添加调试代码 ==========
# 调试：检查特征分离结果
print("\n=== 特征分离调试信息 ===")
print(f"临床特征列数: {clinical_features.shape[1]}")
print(f"影像组学特征列数: {radiomics_features.shape[1]}")
print(f"iTED特征列数: {iTED_features.shape[1]}")

# 检查是否有缺失的列
missing_clinical = [col for col in clinical_feature_names if col not in df_merged.columns]
if missing_clinical:
    print(f"警告：以下临床特征在合并数据中缺失: {missing_clinical}")

# 检查shape_Elongation的位置
if 'shape_Elongation' in df_merged.columns:
    shape_col_index = df_merged.columns.get_loc('shape_Elongation')
    print(f"shape_Elongation在合并数据的第 {shape_col_index} 列")

# 额外检查：确认shape_Elongation在哪个特征集中
if 'shape_Elongation' in clinical_names:
    print("⚠️ shape_Elongation在临床特征中!")
elif 'shape_Elongation' in radiomics_names:
    print("✓ shape_Elongation正确在影像组学特征中!")
else:
    print("❌ shape_Elongation未找到!")
# ========== 调试代码结束 ==========

print(f"\n各类特征数量:")
print(f"临床特征: {len(clinical_names)}")
print(f"影像组学特征: {len(radiomics_names)}")
print(f"iTED特征: {len(iTED_names)}")
print(f"3D_ITHscore: 1")

# 检查是否有shape_开头的特征被错误分类
for col in clinical_features.columns:
    if col.startswith(('shape_', 'firstorder_', 'glcm_', 'glrlm_', 'glszm_', 'gldm_')):
        print(f"警告：{col} 可能被错误归类为临床特征！")

# 确认shape_Elongation在正确的位置
if 'shape_Elongation' in radiomics_features.columns:
    print("✓ shape_Elongation正确归类为影像组学特征")
else:
    print("✗ shape_Elongation未在影像组学特征中找到")

# 更新特征名称列表
clinical_names = clinical_features.columns.tolist()
radiomics_names = radiomics_features.columns.tolist()
iTED_names = iTED_features.columns.tolist()

print(f"\n各类特征数量:")
print(f"临床特征: {len(clinical_names)}")
print(f"影像组学特征: {len(radiomics_names)}")
print(f"iTED特征: {len(iTED_names)}")
print(f"3D_ITHscore: 1")

# ===================== 加载外部验证数据集 =====================
print("\n加载外部验证数据集...")
print("-" * 60)

try:
    external_clinical_df = read_csv_with_encoding('/data/clinical_features_processed_ydyy.csv')
    external_radiomics_df = read_csv_with_encoding('/data/radiomics_features_ydyy.csv')
    external_iTED_df = read_csv_with_encoding('/data/iTED_features_ydyy.csv')
    external_ITHscore_df = read_csv_with_encoding('/data/3D_ITHscore_ydyy.csv')

    print(f"\n外部验证数据维度:")
    print(f"临床特征: {external_clinical_df.shape}")
    print(f"传统影像组学特征: {external_radiomics_df.shape}")
    print(f"iTED特征: {external_iTED_df.shape}")
    print(f"3D_ITHscore: {external_ITHscore_df.shape}")

    # 合并外部验证数据
    external_df_merged = external_clinical_df.merge(external_radiomics_df, on='PatientID', how='inner')
    external_df_merged = external_df_merged.merge(external_iTED_df, on='PatientID', how='inner')
    external_df_merged = external_df_merged.merge(external_ITHscore_df, on='PatientID', how='inner')

    print(f"\n外部验证集合并后数据维度: {external_df_merged.shape}")
    print(f"外部验证集患者数量: {len(external_df_merged)}")

    # 提取外部验证集的标签和特征
    y_external = external_df_merged['M_stage'].values
    patient_ids_external = external_df_merged['PatientID'].values

    # 分离外部验证集的不同类型特征（基于列名）
    # 使用与训练集相同的特征名列表
    external_clinical_features = external_df_merged[clinical_names].copy()
    external_radiomics_features = external_df_merged[radiomics_names].copy()
    external_iTED_features = external_df_merged[iTED_names].copy()
    external_ITHscore_feature = external_df_merged[['3D_ITHscore']].copy()

    print(f"\n外部验证集特征维度检查:")
    print(f"临床特征: {external_clinical_features.shape}")
    print(f"影像组学特征: {external_radiomics_features.shape}")
    print(f"iTED特征: {external_iTED_features.shape}")

    print(f"\n外部验证集类别分布:")
    print(f"M0 (无转移): {(y_external == 0).sum()} ({(y_external == 0).sum() / len(y_external) * 100:.1f}%)")
    print(f"M1 (有转移): {(y_external == 1).sum()} ({(y_external == 1).sum() / len(y_external) * 100:.1f}%)")

    external_validation_available = True
except Exception as e:
    print(f"外部验证数据加载失败: {str(e)}")
    external_validation_available = False

# ===================== 类别不平衡分析 =====================
print("\n" + "=" * 60)
print("类别不平衡分析")
print("=" * 60)

# 分析训练数据的类别分布
n_total = len(y)
n_positive = np.sum(y == 1)
n_negative = np.sum(y == 0)
imbalance_ratio = n_negative / n_positive if n_positive > 0 else np.inf

print(f"\n训练数据集类别分布:")
print(f"总样本数: {n_total}")
print(f"M0 (阴性): {n_negative} ({n_negative / n_total * 100:.1f}%)")
print(f"M1 (阳性): {n_positive} ({n_positive / n_total * 100:.1f}%)")
print(f"不平衡比例: {imbalance_ratio:.1f}:1")

if imbalance_ratio > 3:
    print("\n警告: 检测到严重的类别不平衡！")
    print("将使用SMOTE过采样和调整后的评估策略。")

# ===================== 2. 处理分类变量 =====================
print("\n2. 处理分类变量")
print("-" * 60)

# 明确定义分类变量列表
categorical_columns = [
    #'TG', 'TGAb', 'TPOAb',
    'N_stage', 'T_stage',
    'Infiltrated_the_adjacent_tissue', #'Tumor_size',
    'Multifocal', 'Benign_thyroid_lesions', 'Sex'
]

# 其余为数值变量
numeric_columns = [col for col in clinical_features.columns if col not in categorical_columns]

print(f"分类变量 ({len(categorical_columns)}): {categorical_columns}")
print(f"数值变量 ({len(numeric_columns)}): {numeric_columns}")

# 创建编码映射字典
encoding_mappings = {
    #'TG': {'＜62.9': 0, '≥62.9': 1},
    #'TGAb': {'＜115': 0, '≥115': 1},
    #'TPOAb': {'＜40': 0, '≥40': 1},
    'N_stage': {'N0': 0, 'N1': 1},
    'T_stage': {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3},
    'Infiltrated_the_adjacent_tissue': {'No': 0, 'Yes': 1},
    #'Tumor_size': {'≤1': 0, '＞1 and ≤2': 1, '＞2': 2},
    'Multifocal': {'No': 0, 'Yes': 1},
    'Benign_thyroid_lesions': {'No': 0, 'Yes': 1},
    'Sex': {'Female': 0, 'Male': 1}
}

# 应用编码
clinical_features_encoded = clinical_features.copy()

for col, mapping in encoding_mappings.items():
    if col in clinical_features_encoded.columns:
        # 使用map函数进行编码
        clinical_features_encoded[col] = clinical_features_encoded[col].map(mapping)

        # 检查是否有未匹配的值（NaN）
        if clinical_features_encoded[col].isna().any():
            print(f"警告：{col} 列存在未匹配的值")
            print(f"  原始值: {clinical_features[col].unique()}")
            print(f"  映射字典: {mapping}")
            # 填充为参考类别（0）
            clinical_features_encoded[col].fillna(0, inplace=True)
            print(f"  已将缺失值填充为参考类别: 0")

# 对外部验证集进行相同的编码
if external_validation_available:
    external_clinical_features_encoded = external_clinical_features.copy()

    for col, mapping in encoding_mappings.items():
        if col in external_clinical_features_encoded.columns:
            external_clinical_features_encoded[col] = external_clinical_features_encoded[col].map(mapping)

            # 检查是否有未匹配的值（NaN）
            if clinical_features_encoded[col].isna().any():
                print(f"警告：{col} 列存在未匹配的值")
                print(f"  原始值: {clinical_features[col].unique()}")
                print(f"  映射字典: {mapping}")
                # 根据实际情况选择处理策略：
                # clinical_features_encoded[col].fillna(0, inplace=True)  # 填充为参考类别
                # 或者抛出错误
                # raise ValueError(f"{col} 列存在未匹配的值")

# 转换为numpy数组
X_clinical = clinical_features_encoded.values
X_radiomics = radiomics_features.values
X_iTED = iTED_features.values
X_ITHscore = ITHscore_feature.values

if external_validation_available:
    X_clinical_external = external_clinical_features_encoded.values
    X_radiomics_external = external_radiomics_features.values
    X_iTED_external = external_iTED_features.values
    X_ITHscore_external = external_ITHscore_feature.values

# 计算类别权重
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"\n类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")

# ===================== 3. 数据集划分 =====================
print("\n3. 数据集划分")
print("-" * 60)

# 分层划分数据集
indices = np.arange(len(y))
train_val_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y[train_val_idx]
)

print(f"训练集: {len(train_idx)} 样本 (M1: {y[train_idx].sum()})")
print(f"验证集: {len(val_idx)} 样本 (M1: {y[val_idx].sum()})")
print(f"测试集: {len(test_idx)} 样本 (M1: {y[test_idx].sum()})")


# ===================== 校准相关函数 =====================
def calibrate_probabilities(model, X_train, y_train, X_val, y_val, method='isotonic'):
    """对模型进行概率校准"""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    else:  # sigmoid/platt
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()

    # 获取验证集的预测概率
    val_proba = model.predict_proba(X_val)[:, 1]

    # 训练校准器
    calibrator.fit(val_proba.reshape(-1, 1), y_val)

    return calibrator


def apply_calibration(proba, calibrator):
    """应用校准器"""
    return calibrator.predict(proba.reshape(-1, 1))


def select_features_rfecv(X_train, y_train, feature_names, feature_type,
                          min_features_to_select=1, max_features_to_select=None,
                          step=1, cv=5):
    """完全数据驱动的RFECV特征选择"""
    print(f"\n使用RFECV选择 {feature_type} 特征...")
    print(f"原始特征数: {len(feature_names)}")

    # ===== 关键修改7：调整最小特征数 =====
    # 根据特征类型设置不同的最小值
    if 'Clinical' in feature_type:
        min_features_to_select = max(3, int(len(feature_names) * 0.2))  # 至少3个或20%
    elif 'All' in feature_type:
        min_features_to_select = max(10, int(len(feature_names) * 0.15))  # 至少10个或15%
    else:
        min_features_to_select = max(5, int(len(feature_names) * 0.15))  # 至少5个或15%

    print(f"设置最小特征数: {min_features_to_select}")

    # 创建基础估计器
    estimator = LGBMClassifier(
        n_estimators=200,  # 增加（原来是100）
        max_depth=6,  # 增加（原来是4）
        num_leaves=63,  # 增加（原来是31）
        random_state=RANDOM_STATE,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        objective='binary',
        metric='binary_logloss'
    )

    # 创建RFECV选择器 - 让它自由选择
    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE),
        scoring='average_precision',  # 使用AUC-PR作为评分标准
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
        verbose=0
    )

    # 拟合RFECV
    print("正在进行交叉验证特征选择，请稍候...")
    rfecv.fit(X_train, y_train)

    # 获取结果
    n_features_optimal = rfecv.n_features_
    selected_mask = rfecv.support_
    selected_features = np.array(feature_names)[selected_mask]

    # 打印详细结果
    print(f"RFECV自动确定的最优特征数: {n_features_optimal}")

    # 输出交叉验证得分
    try:
        if hasattr(rfecv, 'cv_results_'):
            best_score_idx = np.argmax(rfecv.cv_results_['mean_test_score'])
            best_score = rfecv.cv_results_['mean_test_score'][best_score_idx]
            best_std = rfecv.cv_results_['std_test_score'][best_score_idx]
            print(f"最佳交叉验证AUC-PR: {best_score:.4f} ± {best_std:.4f}")

            # 显示不同特征数的性能
            print(f"不同特征数的AUC-PR:")
            n_features_tested = len(rfecv.cv_results_['mean_test_score'])
            for i in range(0, min(5, n_features_tested)):
                n_feat = len(feature_names) - i * step if step >= 1 else len(feature_names) * (1 - step) ** i
                score = rfecv.cv_results_['mean_test_score'][-(i + 1)]
                print(f"  {int(n_feat)}个特征: {score:.4f}")
    except Exception as e:
        print(f"无法显示详细得分: {str(e)}")

    # 可视化部分
    try:
        # 计算实际测试的特征数量序列
        n_features = len(feature_names)
        if step >= 1:
            n_features_list = list(range(n_features, max(min_features_to_select - step, 0), -step))
            if min_features_to_select not in n_features_list:
                n_features_list.append(min_features_to_select)
            n_features_list = sorted(n_features_list)
        else:
            n_features_list = []
            n_current = n_features
            while n_current > min_features_to_select:
                n_features_list.append(n_current)
                n_current = int(n_current - n_current * step)
            n_features_list.append(min_features_to_select)
            n_features_list = sorted(set(n_features_list))

        plt.figure(figsize=(10, 6))

        if hasattr(rfecv, 'cv_results_'):
            if len(rfecv.cv_results_['mean_test_score']) == len(n_features_list):
                x_axis = n_features_list
            else:
                x_axis = range(len(rfecv.cv_results_['mean_test_score']))

            plt.plot(x_axis, rfecv.cv_results_['mean_test_score'], 'b-', label='Mean CV Score', linewidth=2)

            plt.fill_between(x_axis,
                             rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
                             rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
                             alpha=0.2, color='blue', label='±1 std')

            # 标记最优点
            best_idx = np.argmax(rfecv.cv_results_['mean_test_score'])
            if isinstance(x_axis, list):
                best_x = x_axis[best_idx]
            else:
                best_x = best_idx
            best_y = rfecv.cv_results_['mean_test_score'][best_idx]

            plt.scatter([best_x], [best_y], color='red', s=100, zorder=5,
                        label=f'Optimal: {n_features_optimal} features')
            plt.axvline(x=best_x, color='red', linestyle='--', alpha=0.5)

        plt.xlabel('Number of features selected')
        plt.ylabel('Cross validation score (AUC-PR)')
        plt.title(f'RFECV for {feature_type} Features\n(Data-driven selection)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图像
        safe_filename = feature_type.replace("/", "_").replace("+", "_").lower()
        plt.savefig(os.path.join(results_dir, f'rfecv_{safe_filename}_features.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"可视化失败: {str(e)}")
        plt.close()

    # 保存特征重要性
    try:
        if hasattr(rfecv.estimator_, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': rfecv.estimator_.feature_importances_[:len(selected_features)]
            }).sort_values('importance', ascending=False)

            safe_filename = feature_type.replace("/", "_").replace("+", "_").lower()
            feature_importance.to_csv(
                os.path.join(results_dir, f'feature_importance_{safe_filename}.csv'),
                index=False
            )

            print(f"\nTop 10 {feature_type} 特征:")
            print(feature_importance.head(10))

        # 保存所有特征的排名
        all_features_ranking = pd.DataFrame({
            'feature': feature_names,
            'selected': selected_mask,
            'ranking': rfecv.ranking_ if hasattr(rfecv, 'ranking_') else [999] * len(feature_names)
        }).sort_values('ranking')

        safe_filename = feature_type.replace("/", "_").replace("+", "_").lower()
        all_features_ranking.to_csv(
            os.path.join(results_dir, f'feature_ranking_{safe_filename}.csv'),
            index=False
        )

    except Exception as e:
        print(f"保存特征重要性时出错: {str(e)}")
        print("继续执行...")

    return selected_mask, selected_features.tolist()


# ===================== 5. 改进的两阶段特征选择 =====================
print("\n4. 两阶段特征选择")
print("-" * 60)

# 第一阶段：各模态内部预筛选
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif


def improved_first_stage_selection(X, y, feature_names, modality_name):
    """
    改进的第一阶段特征筛选，更适合不平衡数据和筛查目的
    """
    from scipy import stats
    from sklearn.feature_selection import mutual_info_classif, f_classif
    from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    print(f"\n改进的数据驱动筛选 - {modality_name}")
    print(f"原始特征数: {X.shape[1]}")

    # 特殊处理：特征很少时保留全部
    if X.shape[1] <= 3:
        print(f"特征数≤3，保留全部")
        return np.ones(X.shape[1], dtype=bool), feature_names

    # ========== 计算多个评价指标 ==========

    # 1. 互信息（非线性关系）
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    # 2. F统计量和p值
    f_scores, p_values = f_classif(X, y)

    # 3. 单变量AUC-ROC
    auc_scores = []
    for i in range(X.shape[1]):
        try:
            auc = roc_auc_score(y, X[:, i])
            auc = max(auc, 1 - auc)  # 处理负相关
        except:
            auc = 0.5
        auc_scores.append(auc)
    auc_scores = np.array(auc_scores)

    # 4. 单变量AUC-PR（对不平衡数据更敏感）
    auc_pr_scores = []
    for i in range(X.shape[1]):
        try:
            # 标准化特征到[0,1]作为概率
            x_norm = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min() + 1e-10)
            auc_pr = average_precision_score(y, x_norm)
            # 也考虑反向关系
            auc_pr_inv = average_precision_score(y, 1 - x_norm)
            auc_pr = max(auc_pr, auc_pr_inv)
        except:
            auc_pr = np.mean(y)  # baseline
        auc_pr_scores.append(auc_pr)
    auc_pr_scores = np.array(auc_pr_scores)

    # 5. 单变量敏感性（在最优阈值下）
    sensitivity_scores = []
    for i in range(X.shape[1]):
        try:
            # 找到最优阈值（Youden指数）
            thresholds = np.percentile(X[:, i], np.arange(10, 91, 10))
            best_sens = 0
            for thresh in thresholds:
                y_pred = (X[:, i] >= thresh).astype(int)
                sens = recall_score(y, y_pred)
                # 也考虑反向
                sens_inv = recall_score(y, 1 - y_pred)
                best_sens = max(best_sens, sens, sens_inv)
            sensitivity_scores.append(best_sens)
        except:
            sensitivity_scores.append(0)
    sensitivity_scores = np.array(sensitivity_scores)

    # 6. 基于树的特征重要性（能捕捉交互效应）
    try:
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=RANDOM_STATE,
            class_weight='balanced'  # 处理不平衡
        )
        rf.fit(X, y)
        tree_importance = rf.feature_importances_
    except:
        tree_importance = np.zeros(X.shape[1])

    # 7. 相关系数的绝对值（简单但有效）
    correlation_scores = []
    for i in range(X.shape[1]):
        corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        correlation_scores.append(corr if not np.isnan(corr) else 0)
    correlation_scores = np.array(correlation_scores)

    # ========== 标准化各指标到[0,1] ==========
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    def safe_normalize(scores):
        if np.std(scores) > 0:
            return scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        else:
            return np.ones_like(scores) * 0.5

    mi_norm = safe_normalize(mi_scores)
    auc_norm = safe_normalize(auc_scores)
    auc_pr_norm = safe_normalize(auc_pr_scores)
    sensitivity_norm = safe_normalize(sensitivity_scores)
    tree_importance_norm = safe_normalize(tree_importance)
    correlation_norm = safe_normalize(correlation_scores)

    # p值反转（越小越好）
    p_norm = 1 - safe_normalize(p_values)

    # ========== 综合评分（针对筛查工具优化权重） ==========

    # 对于筛查工具，更重视敏感性相关指标
    weights = {
        'sensitivity': 0.20,  # 单变量敏感性
        'auc_pr': 0.20,  # AUC-PR（对不平衡数据敏感）
        'tree_importance': 0.15,  # 树模型重要性（捕捉交互）
        'mi': 0.15,  # 互信息
        'p': 0.10,  # p值
        'auc_roc': 0.10,  # AUC-ROC
        'correlation': 0.10  # 相关系数
    }

    combined_scores = (
            sensitivity_norm * weights['sensitivity'] +
            auc_pr_norm * weights['auc_pr'] +
            tree_importance_norm * weights['tree_importance'] +
            mi_norm * weights['mi'] +
            p_norm * weights['p'] +
            auc_norm * weights['auc_roc'] +
            correlation_norm * weights['correlation']
    )

    # ========== 输出各指标得分（用于调试） ==========
    print(f"\n特征评分详情（{modality_name}）:")
    print(f"{'特征名':<30} {'综合分':<8} {'敏感性':<8} {'AUC-PR':<8} {'树重要性':<8}")
    print("-" * 70)

    # 按综合分排序
    sorted_indices = np.argsort(combined_scores)[::-1]
    for idx in sorted_indices[:10]:  # 显示前10个
        print(f"{feature_names[idx]:<30} {combined_scores[idx]:.3f}   "
              f"{sensitivity_scores[idx]:.3f}   {auc_pr_scores[idx]:.3f}   "
              f"{tree_importance[idx]:.3f}")

    # ========== 确定选择数量（数据驱动） ==========

        # 方法1：基于累积贡献率
        sorted_scores = np.sort(combined_scores)[::-1]
        cumsum_scores = np.cumsum(sorted_scores)
        total_score = cumsum_scores[-1]

        # 找到包含90%总分的特征数（原来是85%）
        n_85_score = np.argmax(cumsum_scores >= 0.90 * total_score) + 1  # 改为90%

        # 方法2：基于得分下降率（拐点检测）
        score_changes = []
        for i in range(1, min(len(sorted_scores), 20)):
            if sorted_scores[i - 1] > 0:
                change = (sorted_scores[i - 1] - sorted_scores[i]) / sorted_scores[i - 1]
                score_changes.append(change)
            else:
                score_changes.append(0)

        # 找到下降率>20%的第一个点（原来是30%）
        elbow_idx = 1
        for i, change in enumerate(score_changes):
            if change > 0.2:  # 改为0.2
                elbow_idx = i + 1
                break

        # 方法3：基于统计显著性
        mean_score = np.mean(combined_scores)
        std_score = np.std(combined_scores)

        # 选择得分高于均值+0.25标准差的特征（原来是0.5）
        n_significant = np.sum(combined_scores > mean_score + 0.25 * std_score)  # 改为0.25

        # ===== 关键修改6：设置最小特征数 =====
        # 确保每个模态至少保留一定数量的特征
        min_features_by_modality = {
            'Clinical': max(5, int(X.shape[1] * 0.3)),  # 至少5个或30%
            'Radiomics': max(10, int(X.shape[1] * 0.2)),  # 至少10个或20%
            'iTED': max(10, int(X.shape[1] * 0.2)),  # 至少10个或20%
            '3D_ITHscore': 1
        }

        min_features = min_features_by_modality.get(modality_name, max(3, int(X.shape[1] * 0.2)))

        # 综合决策
        candidate_numbers = []
        if n_85_score >= min_features:
            candidate_numbers.append(n_85_score)
        if elbow_idx >= min_features:
            candidate_numbers.append(elbow_idx)
        if n_significant >= min_features:
            candidate_numbers.append(n_significant)

        # 如果没有候选，使用最小特征数
        if not candidate_numbers:
            candidate_numbers.append(min_features)

        # 选择中位数作为最终特征数
        n_select = int(np.median(candidate_numbers))
        n_select = max(min_features, min(n_select, X.shape[1]))  # 确保不低于最小值

    print(f"\n选择策略:")
    print(f"  85%累积贡献: {n_85_score}个")
    print(f"  得分拐点: {elbow_idx}个")
    print(f"  显著特征(>μ+0.5σ): {n_significant}个")
    print(f"  最终选择: {n_select}个")

    # ========== 选择特征 ==========
    top_indices = np.argsort(combined_scores)[-n_select:]
    selected_mask = np.zeros(X.shape[1], dtype=bool)
    selected_mask[top_indices] = True

    selected_features = [feature_names[i] for i in top_indices]

    # 特别检查3D_ITHscore和iTED特征
    if modality_name in ['iTED', '3D_ITHscore'] or '3D_ITHscore' in feature_names:
        for i, name in enumerate(feature_names):
            if name == '3D_ITHscore' or 'iTED' in name:
                score_rank = np.argsort(combined_scores)[::-1].tolist().index(i) + 1
                print(f"\n特别关注: {name}")
                print(f"  综合得分: {combined_scores[i]:.3f} (排名: {score_rank}/{len(feature_names)})")
                print(f"  各项指标: 敏感性={sensitivity_scores[i]:.3f}, "
                      f"AUC-PR={auc_pr_scores[i]:.3f}, "
                      f"树重要性={tree_importance[i]:.3f}")
                print(f"  是否入选: {'是' if selected_mask[i] else '否'}")

    return selected_mask, selected_features


# 对各模态进行第一阶段筛选
print("\n=== 第一阶段：数据驱动特征筛选 ===")

# 临床特征
clinical_mask_stage1, clinical_names_stage1 = improved_first_stage_selection(
    X_clinical[train_idx], y[train_idx], clinical_names, "Clinical"
)
X_clinical_stage1 = X_clinical[:, clinical_mask_stage1]

# 影像组学特征
radiomics_mask_stage1, radiomics_names_stage1 = improved_first_stage_selection(
    X_radiomics[train_idx], y[train_idx], radiomics_names, "Radiomics"
)
X_radiomics_stage1 = X_radiomics[:, radiomics_mask_stage1]

# iTED特征（重点：从93个中筛选）
iTED_mask_stage1, iTED_names_stage1 = improved_first_stage_selection(
    X_iTED[train_idx], y[train_idx], iTED_names, "iTED"
)
X_iTED_stage1 = X_iTED[:, iTED_mask_stage1]

# 3D_ITHscore不需要筛选
X_ITHscore_stage1 = X_ITHscore
ITHscore_names_stage1 = ["3D_ITHscore"]

print("\n第一阶段筛选结果汇总:")
print(f"Clinical: {len(clinical_names)} → {len(clinical_names_stage1)}")
print(f"Radiomics: {len(radiomics_names)} → {len(radiomics_names_stage1)}")
print(f"iTED: {len(iTED_names)} → {len(iTED_names_stage1)}")
print(f"3D_ITHscore: 1 → 1")
print(f"总特征数(第一阶段后): {len(clinical_names_stage1) + len(radiomics_names_stage1) + len(iTED_names_stage1) + 1}")

# 第二阶段：基于第一阶段结果构建组合
print("\n=== 第二阶段：组合特征精细筛选 ===")

# 使用第一阶段筛选后的特征构建组合
raw_feature_combinations = {
    # 单模态 - 不设置限制
    "Clinical": {
        "features": X_clinical_stage1,
        "names": clinical_names_stage1
    },
    "Radiomics": {
        "features": X_radiomics_stage1,
        "names": radiomics_names_stage1
    },
    "iTED": {
        "features": X_iTED_stage1,
        "names": iTED_names_stage1
    },
    "3D_ITHscore": {
        "features": X_ITHscore_stage1,
        "names": ITHscore_names_stage1
    },

    # 双模态组合
    "Clinical+Radiomics": {
        "features": np.hstack([X_clinical_stage1, X_radiomics_stage1]),
        "names": clinical_names_stage1 + radiomics_names_stage1
    },
    "Clinical+iTED": {
        "features": np.hstack([X_clinical_stage1, X_iTED_stage1]),
        "names": clinical_names_stage1 + iTED_names_stage1
    },
    "Clinical+3D_ITHscore": {
        "features": np.hstack([X_clinical_stage1, X_ITHscore_stage1]),
        "names": clinical_names_stage1 + ITHscore_names_stage1
    },
    "Radiomics+iTED": {
        "features": np.hstack([X_radiomics_stage1, X_iTED_stage1]),
        "names": radiomics_names_stage1 + iTED_names_stage1
    },
    "Radiomics+3D_ITHscore": {
        "features": np.hstack([X_radiomics_stage1, X_ITHscore_stage1]),
        "names": radiomics_names_stage1 + ITHscore_names_stage1
    },
    "iTED+3D_ITHscore": {
        "features": np.hstack([X_iTED_stage1, X_ITHscore_stage1]),
        "names": iTED_names_stage1 + ITHscore_names_stage1
    },

    # 三模态组合
    "Clinical+Radiomics+iTED": {
        "features": np.hstack([X_clinical_stage1, X_radiomics_stage1, X_iTED_stage1]),
        "names": clinical_names_stage1 + radiomics_names_stage1 + iTED_names_stage1
    },
    "Clinical+Radiomics+3D_ITHscore": {
        "features": np.hstack([X_clinical_stage1, X_radiomics_stage1, X_ITHscore_stage1]),
        "names": clinical_names_stage1 + radiomics_names_stage1 + ITHscore_names_stage1
    },
    "Clinical+iTED+3D_ITHscore": {
        "features": np.hstack([X_clinical_stage1, X_iTED_stage1, X_ITHscore_stage1]),
        "names": clinical_names_stage1 + iTED_names_stage1 + ITHscore_names_stage1
    },
    "All_Imaging": {
        "features": np.hstack([X_radiomics_stage1, X_iTED_stage1, X_ITHscore_stage1]),
        "names": radiomics_names_stage1 + iTED_names_stage1 + ITHscore_names_stage1
    },

    # 四模态组合
    "All_Features": {
        "features": np.hstack([X_clinical_stage1, X_radiomics_stage1, X_iTED_stage1, X_ITHscore_stage1]),
        "names": clinical_names_stage1 + radiomics_names_stage1 + iTED_names_stage1 + ITHscore_names_stage1
    }
}

# 对每种组合进行特征筛选
print("\n对每种特征组合进行RFECV特征筛选...")
feature_combinations = {}
selected_features_info = {}

for name, combo in raw_feature_combinations.items():
    print(f"\n处理 {name} 组合...")
    print(f"原始特征数: {len(combo['names'])}")

    # 3D_ITHscore只有一个特征，不需要筛选
    if name == "3D_ITHscore":
        feature_combinations[name] = {
            "features": combo["features"],
            "names": combo["names"]
        }
        selected_features_info[name] = {
            "selected_mask": np.array([True]),
            "selected_names": combo["names"],
            "n_features": 1
        }
        print(f"跳过筛选（只有1个特征）")
        continue

    # 对其他组合进行RFECV特征筛选
    X_train_combo = combo["features"][train_idx]

    # 根据特征数量动态调整step
    n_features = len(combo['names'])
    if n_features > 100:
        step = 10  # 特征很多时，每次删除10个
    elif n_features > 50:
        step = 5   # 特征较多时，每次删除5个
    elif n_features > 20:
        step = 2   # 特征中等时，每次删除2个
    else:
        step = 1   # 特征较少时，每次删除1个

    # 执行RFECV - 让RFECV自己决定最优特征数
    # min_features_to_select设为1，让算法完全自主决定
    selected_mask, selected_names = select_features_rfecv(
        X_train_combo,
        y[train_idx],
        combo["names"],
        f"{name}组合",
        min_features_to_select=1,  # 最少1个特征
        max_features_to_select=None,  # 不限制最大特征数
        step=step,
        cv=5
    )

    # 保存筛选后的特征
    feature_combinations[name] = {
        "features": combo["features"][:, selected_mask],
        "names": selected_names
    }

    selected_features_info[name] = {
        "selected_mask": selected_mask,
        "selected_names": selected_names,
        "n_features": len(selected_names)
    }

    print(f"RFECV自动选择了: {len(selected_names)}个特征")

# 打印特征筛选汇总
print("\n" + "=" * 60)
print("特征筛选结果汇总")
print("=" * 60)
for name, info in selected_features_info.items():
    print(f"{name}: {len(raw_feature_combinations[name]['names'])} → {info['n_features']} 特征")

# 构建外部验证集的特征组合（使用相同的特征选择）
if external_validation_available:
    print("\n构建外部验证集的特征组合...")

    # 对外部验证集应用第一阶段的特征选择
    X_clinical_external_stage1 = X_clinical_external[:, clinical_mask_stage1]
    X_radiomics_external_stage1 = X_radiomics_external[:, radiomics_mask_stage1]
    X_iTED_external_stage1 = X_iTED_external[:, iTED_mask_stage1]
    X_ITHscore_external_stage1 = X_ITHscore_external

    external_feature_combinations = {}

    for name, info in selected_features_info.items():
        if name == "3D_ITHscore":
            external_feature_combinations[name] = {
                "features": X_ITHscore_external_stage1,
                "names": ["3D_ITHscore"]
            }
            continue

        # 根据组合名称重建外部验证集的特征矩阵（使用第一阶段筛选后的特征）
        if name == "Clinical":
            X_external_combo = X_clinical_external_stage1
        elif name == "Radiomics":
            X_external_combo = X_radiomics_external_stage1
        elif name == "iTED":
            X_external_combo = X_iTED_external_stage1
        elif name == "3D_ITHscore":
            X_external_combo = X_ITHscore_external_stage1

        # 双模态组合
        elif name == "Clinical+Radiomics":
            X_external_combo = np.hstack([X_clinical_external_stage1, X_radiomics_external_stage1])
        elif name == "Clinical+iTED":
            X_external_combo = np.hstack([X_clinical_external_stage1, X_iTED_external_stage1])
        elif name == "Clinical+3D_ITHscore":
            X_external_combo = np.hstack([X_clinical_external_stage1, X_ITHscore_external_stage1])
        elif name == "Radiomics+iTED":
            X_external_combo = np.hstack([X_radiomics_external_stage1, X_iTED_external_stage1])
        elif name == "Radiomics+3D_ITHscore":
            X_external_combo = np.hstack([X_radiomics_external_stage1, X_ITHscore_external_stage1])
        elif name == "iTED+3D_ITHscore":
            X_external_combo = np.hstack([X_iTED_external_stage1, X_ITHscore_external_stage1])

        # 三模态组合
        elif name == "Clinical+Radiomics+iTED":
            X_external_combo = np.hstack(
                [X_clinical_external_stage1, X_radiomics_external_stage1, X_iTED_external_stage1])
        elif name == "Clinical+Radiomics+3D_ITHscore":
            X_external_combo = np.hstack(
                [X_clinical_external_stage1, X_radiomics_external_stage1, X_ITHscore_external_stage1])
        elif name == "Clinical+iTED+3D_ITHscore":
            X_external_combo = np.hstack(
                [X_clinical_external_stage1, X_iTED_external_stage1, X_ITHscore_external_stage1])
        elif name == "All_Imaging":  # Radiomics+iTED+3D_ITHscore
            X_external_combo = np.hstack(
                [X_radiomics_external_stage1, X_iTED_external_stage1, X_ITHscore_external_stage1])

        # 四模态组合
        elif name == "All_Features":
            X_external_combo = np.hstack([X_clinical_external_stage1, X_radiomics_external_stage1,
                                          X_iTED_external_stage1, X_ITHscore_external_stage1])
        else:
            # 如果出现未知的组合名称，抛出错误
            raise ValueError(f"未知的特征组合名称: {name}")

        # 应用第二阶段的特征选择
        external_feature_combinations[name] = {
            "features": X_external_combo[:, info["selected_mask"]],
            "names": info["selected_names"]
        }

print("\n特征组合构建完成!")
print(f"总共 {len(feature_combinations)} 种组合")
print("\n各组合特征数:")
for name, combo in feature_combinations.items():
    print(f"  {name}: {combo['features'].shape[1]} 特征")

# ===================== 继续原代码的模型训练部分 =====================

# ===================== 添加置信区间计算函数 =====================
def calculate_confidence_interval(y_true, y_pred_proba, metric_func, n_bootstraps=1000, ci=95):
    """使用Bootstrap计算置信区间"""
    scores = []
    n_size = len(y_true)

    for i in range(n_bootstraps):
        # Bootstrap采样
        indices = np.random.choice(n_size, n_size, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred_proba[indices])
        scores.append(score)

    # 计算置信区间
    alpha = 100 - ci
    lower = np.percentile(scores, alpha / 2)
    upper = np.percentile(scores, 100 - alpha / 2)

    return lower, upper


# ===================== 定义F2-Score =====================
def f2_score(y_true, y_pred):
    """计算F2分数（更重视召回率）"""
    return fbeta_score(y_true, y_pred, beta=2)


# 创建F2 scorer for cross-validation
f2_scorer = make_scorer(f2_score)


# ===================== 6. 改进的超参数优化函数（考虑不平衡） =====================
# 先添加自定义评估函数（在optimize_hyperparameters_balanced函数之前）
def custom_eval_metric(y_pred, dtrain):
    """
    自定义评估指标，更重视召回率（减少假阴性）
    """
    y_true = dtrain.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 计算各种指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # 代价敏感评分：假阴性的代价是假阳性的5倍
    fn_cost = 5.0
    fp_cost = 1.0

    # 计算总代价（越低越好）
    total_cost = fn_cost * fn + fp_cost * fp

    # 转换为收益指标（越高越好）
    max_cost = fn_cost * np.sum(y_true == 1) + fp_cost * np.sum(y_true == 0)
    cost_score = 1 - (total_cost / max_cost)

    return 'cost_score', cost_score


def optimize_hyperparameters_balanced(X_train, y_train, n_trials=50, use_smote=True):
    """使用Optuna优化LightGBM超参数 - 优先提高敏感性"""

    # 计算类别分布
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    base_scale_pos_weight = n_negative / n_positive

    print(f"  优化目标：最大化敏感性（召回率）")

    def objective(trial):
        # 选择使用哪种不平衡处理方式（二选一）
        balance_mode = trial.suggest_categorical('balance_mode', ['scale_pos_weight', 'is_unbalance'])

        params = {
            # LightGBM基础参数
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',

            # ===== 关键修改1：增加树的数量范围 =====
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),  # 原来是100-500

            # ===== 关键修改2：增加树的深度和复杂度 =====
            'max_depth': trial.suggest_int('max_depth', 5, 12),  # 原来是3-10
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),  # 原来是10-100
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),  # 保持不变

            # ===== 关键修改3：调整学习率范围 =====
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),  # 原来是0.01-0.3

            # 采样参数保持不变
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # 微调：原来是0.5-1.0
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # 微调：原来是0.5-1.0

            # ===== 关键修改4：减少正则化强度 =====
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),  # 原来是0-3
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),  # 原来是0-3

            # 其他参数
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.5),  # 原来是0-1
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'random_state': RANDOM_STATE,
            'verbosity': -1,
            'n_jobs': -1
        }

        # 根据选择的模式设置不平衡处理参数（只能二选一）
        if balance_mode == 'scale_pos_weight':
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight',
                                                             base_scale_pos_weight * 1.0,
                                                             base_scale_pos_weight * 10.0)
        else:  # is_unbalance
            params['is_unbalance'] = True

        if use_smote:
            smote_strategy = trial.suggest_float('smote_strategy', 0.5, 1.0)
            k_neighbors = min(3, n_positive - 1)

            smote = SMOTE(
                sampling_strategy=smote_strategy,
                k_neighbors=k_neighbors,
                random_state=RANDOM_STATE
            )

            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', LGBMClassifier(**params))
            ])

            cv_scores_recall = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                scoring='recall'
            )

            cv_scores_precision = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                scoring='precision'
            )

            avg_precision = cv_scores_precision.mean()
            avg_recall = cv_scores_recall.mean()

            if avg_precision < 0.1:
                return avg_recall * 0.5
            else:
                return avg_recall
        else:
            model = LGBMClassifier(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                scoring='recall'
            )
            return cv_scores.mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params

    # 清理参数
    if use_smote and 'smote_strategy' in best_params:
        smote_strategy = best_params.pop('smote_strategy')
        print(f"  最佳SMOTE采样策略: {smote_strategy:.2f}")

    # 移除balance_mode（不是LightGBM的参数）
    balance_mode = best_params.pop('balance_mode', 'scale_pos_weight')
    print(f"  使用的平衡策略: {balance_mode}")

    # 设置固定参数
    best_params['boosting_type'] = 'gbdt'
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['random_state'] = RANDOM_STATE
    best_params['verbosity'] = -1
    best_params['n_jobs'] = -1

    return best_params


# ===================== 7. 模型训练和评估（改进版） =====================
print("\n5. 训练和评估模型（改进版：处理类别不平衡）")
print("-" * 60)

# 存储所有模型和结果
models = {}
results = {}
external_results = {} if external_validation_available else None
predictions_dict = {}
external_predictions_dict = {} if external_validation_available else None
# 存储SMOTE策略选择结果
smote_selection_results = {}

# 决定是否使用SMOTE
use_smote = imbalance_ratio > 3
print(f"\n使用SMOTE过采样: {'是' if use_smote else '否'}")

# 设置是否使用校准
use_calibration = True
print(f"使用概率校准: {'是' if use_calibration else '否'}")

# 训练每个特征组合的模型
for name, combo in feature_combinations.items():
    print(f"\n训练 {name} 模型...")

    # 获取数据
    X = combo['features']
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    # 超参数优化
    best_params = optimize_hyperparameters_balanced(X_train, y[train_idx], n_trials=30, use_smote=use_smote)

    if use_smote:
        print(f"  应用激进的重采样策略（提高敏感性）...")

        # 存储当前模型的所有策略测试结果
        strategy_results = []

        # 定义重采样策略（确保比例<=1.0）
        sampling_strategies = {
            'SMOTE_0.7': SMOTE(sampling_strategy=0.7, k_neighbors=min(3, y[train_idx].sum() - 1),
                               random_state=RANDOM_STATE),
            'SMOTE_1.0': SMOTE(sampling_strategy=1.0, k_neighbors=min(3, y[train_idx].sum() - 1),
                               random_state=RANDOM_STATE),
            'BorderlineSMOTE_0.8': BorderlineSMOTE(sampling_strategy=0.8, k_neighbors=min(3, y[train_idx].sum() - 1),
                                                   random_state=RANDOM_STATE),
            'ADASYN_0.8': ADASYN(sampling_strategy=0.8, n_neighbors=min(3, y[train_idx].sum() - 1),
                                 random_state=RANDOM_STATE),
            'BorderlineSMOTE_1.0': BorderlineSMOTE(sampling_strategy=1.0, kind='borderline-1',
                                                   k_neighbors=min(3, y[train_idx].sum() - 1),
                                                   random_state=RANDOM_STATE),
        }

        best_val_score = -np.inf
        best_X_resampled = None
        best_y_resampled = None
        best_strategy_name = None

        for strategy_name, sampler in sampling_strategies.items():
            try:
                print(f"    测试策略: {strategy_name}")
                X_temp, y_temp = sampler.fit_resample(X_train, y[train_idx])

                # 使用LightGBM进行快速验证
                temp_model = LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    num_leaves=31,
                    scale_pos_weight=scale_pos_weight * 2,
                    random_state=RANDOM_STATE,
                    objective='binary',
                    metric='binary_logloss',
                    verbosity=-1
                )

                temp_model.fit(
                    X_temp, y_temp,
                    eval_set=[(X_val, y[val_idx])],
                    feature_name=combo['names'],
                )

                # 评估召回率和其他指标
                val_proba = temp_model.predict_proba(X_val)[:, 1]
                val_pred = (val_proba >= 0.3).astype(int)
                val_recall = recall_score(y[val_idx], val_pred)
                val_precision = precision_score(y[val_idx], val_pred, zero_division=0)
                val_f1 = f1_score(y[val_idx], val_pred)

                print(f"    验证集召回率: {val_recall:.4f}, 精确率: {val_precision:.4f}, F1: {val_f1:.4f}")

                # 记录详细结果
                strategy_results.append({
                    'strategy': strategy_name,
                    'validation_recall': val_recall,
                    'validation_precision': val_precision,
                    'validation_f1': val_f1,
                    'resampled_size': len(y_temp),
                    'minority_samples': sum(y_temp),
                    'majority_samples': len(y_temp) - sum(y_temp)
                })

                if val_recall > best_val_score:
                    best_val_score = val_recall
                    best_X_resampled = X_temp
                    best_y_resampled = y_temp
                    best_strategy_name = strategy_name

            except Exception as e:
                print(f"    策略 {strategy_name} 失败: {str(e)}")
                strategy_results.append({
                    'strategy': strategy_name,
                    'validation_recall': 0,
                    'error': str(e)
                })
                continue

        # 保存该模型的SMOTE选择结果
        smote_selection_results[name] = {
            'strategies_tested': strategy_results,
            'best_strategy': best_strategy_name,
            'best_recall': best_val_score,
            'original_samples': len(y[train_idx]),
            'original_minority': sum(y[train_idx]),
            'final_samples': len(best_y_resampled) if best_y_resampled is not None else len(y[train_idx]),
            'final_minority': sum(best_y_resampled) if best_y_resampled is not None else sum(y[train_idx])
        }

        # 打印详细对比表
        print(f"\n  {name} 模型SMOTE策略对比:")
        print(f"  {'策略':<25} {'验证召回率':<12} {'精确率':<10} {'F1分数':<10} {'重采样后':<10}")
        print(f"  {'-' * 70}")
        for result in strategy_results:
            if 'error' not in result:
                print(f"  {result['strategy']:<25} {result['validation_recall']:<12.4f} "
                      f"{result['validation_precision']:<10.4f} {result['validation_f1']:<10.4f} "
                      f"{result['resampled_size']:<10}")
        print(f"  {'-' * 70}")
        print(f"  ★ 最佳策略: {best_strategy_name} (召回率: {best_val_score:.4f})")

        if best_X_resampled is not None:
            X_train_resampled = best_X_resampled
            y_train_resampled = best_y_resampled
            print(f"  重采样后: {len(y_train_resampled)} 样本 (M1: {sum(y_train_resampled)})")
        else:
            X_train_resampled, y_train_resampled = X_train, y[train_idx]
            print(f"  所有重采样策略失败，使用原始数据")
    else:
        X_train_resampled, y_train_resampled = X_train, y[train_idx]

    # 训练最终模型
    model = LGBMClassifier(**best_params)
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val, y[val_idx])],
        feature_name=combo['names'],  # 添加这一行，使用实际的特征名称
        # callbacks=[lgb.log_evaluation(50)]  # 只保留日志，移除早停
    )

    # ===== 关键修改9：添加训练监控 =====
    print(f"  实际训练轮数: {model.n_estimators_}")
    if hasattr(model, 'best_iteration_'):
        print(f"  最佳迭代: {model.best_iteration_}")

    # 检查特征使用情况
    feature_importance = model.feature_importances_
    n_used_features = np.sum(feature_importance > 0)
    print(f"  实际使用的特征数: {n_used_features}/{len(combo['names'])}")

    # 打印最重要的5个特征
    if n_used_features > 0:
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        print(f"  Top 5 特征:")
        for idx in top_features_idx:
            if feature_importance[idx] > 0:
                print(f"    - {combo['names'][idx]}: {feature_importance[idx]:.3f}")

    # 获取预测概率
    y_test_proba_raw = model.predict_proba(X_test)[:, 1]

    # 在验证集上优化阈值 - 优先考虑敏感性
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # 使用新的阈值优化函数
    optimal_threshold, optimal_metric = optimize_threshold_for_screening(
        y[val_idx], y_val_proba, min_sensitivity=0.7, min_npv=0.50  # NPV门槛可按需调
    )
    print(f"  最优阈值（高敏感性）: {optimal_threshold:.3f} (敏感性: {optimal_metric:.3f})")

    # 如果阈值太高，强制降低（对筛查工具更激进）
    if optimal_threshold > 0.25:  # 从0.3降到0.25
        optimal_threshold = 0.25
        print(f"  强制降低阈值到: {optimal_threshold:.3f}")

    # ========== 添加概率校准 ==========
    if use_calibration and name not in ['3D_ITHscore']:  # 3D_ITHscore可能不需要校准
        try:
            print(f"  应用概率校准...")
            calibrator = calibrate_probabilities(model, X_train_resampled, y_train_resampled,
                                                 X_val, y[val_idx], method='isotonic')

            # 校准测试集概率
            y_test_proba = apply_calibration(y_test_proba_raw, calibrator)

            # 保存校准器到模型对象
            model.calibrator = calibrator
        except Exception as e:
            print(f"  校准失败，使用原始概率: {str(e)}")
            y_test_proba = y_test_proba_raw
    else:
        # 不使用校准
        y_test_proba = y_test_proba_raw

    # 使用优化的阈值进行预测
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)


    # 在第1250行左右，保存模型前添加
    # ===== 关键修改10：验证模型复杂度 =====
    def validate_model_complexity(model, model_name, min_trees=50, min_features=5):
        """验证模型是否足够复杂"""
        # 获取实际的树数量
        if hasattr(model, 'booster_'):
            n_trees = model.booster_.num_trees()
        else:
            n_trees = model.n_estimators_

        # 获取使用的特征数
        n_used_features = np.sum(model.feature_importances_ > 0)

        print(f"\n模型复杂度检查 - {model_name}:")
        print(f"  树的数量: {n_trees}")
        print(f"  使用的特征数: {n_used_features}")

        if n_trees < min_trees:
            print(f"  ⚠️ 警告：树的数量过少！建议至少{min_trees}棵树")
        if n_used_features < min_features:
            print(f"  ⚠️ 警告：使用的特征过少！建议至少{min_features}个特征")

        return n_trees >= min_trees and n_used_features >= min_features


    # 在保存模型前调用
    is_complex_enough = validate_model_complexity(model, name)
    if not is_complex_enough:
        print(f"  ⚠️ 模型{name}可能过于简单，考虑重新训练")

    # 保存模型
    models[name] = model

    # 保存预测概率
    predictions_dict[name] = {
        'PatientID': patient_ids[test_idx],
        'True_Label': y[test_idx],
        'Predicted_Probability': y_test_proba,
        'Predicted_Class': y_test_pred
    }

    # 计算各种指标及其置信区间
    print("  计算内部测试集性能指标及置信区间...")

    # 计算所有平衡指标
    balanced_metrics = calculate_balanced_metrics(y[test_idx], y_test_pred, y_test_proba)

    # 计算置信区间（只对关键指标）
    auc_pr_ci = calculate_confidence_interval(y[test_idx], y_test_proba, average_precision_score)
    auc_roc_ci = calculate_confidence_interval(y[test_idx], y_test_proba, roc_auc_score)
    brier_ci = calculate_confidence_interval(y[test_idx], y_test_proba, brier_score_loss)

    # 保存内部测试集结果
    results[name] = {
        'model': model,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'optimal_threshold': optimal_threshold,  # 这里保存optimal_threshold
        **balanced_metrics,  # 包含所有新指标
        'auc_roc_ci': auc_roc_ci,
        'auc_pr_ci': auc_pr_ci,
        'brier_ci': brier_ci,
        'feature_names': combo['names']
    }

    # 输出关键指标
    print(f"  内部测试集 - AUC-PR: {balanced_metrics['auc_pr']:.4f} (95% CI: {auc_pr_ci[0]:.4f}-{auc_pr_ci[1]:.4f})")
    print(f"  内部测试集 - F2-Score: {balanced_metrics['f2_score']:.4f}")
    print(f"  内部测试集 - Sensitivity: {balanced_metrics['sensitivity']:.4f}")
    print(f"  内部测试集 - G-Mean: {balanced_metrics['g_mean']:.4f}")

    # ========== 外部验证集评估 ==========
    # 注意：这部分代码现在在内部测试集结果保存之后
    if external_validation_available:
        print("  计算外部验证集性能指标及置信区间...")

        X_external = external_feature_combinations[name]['features']

        # 现在optimal_threshold已经在当前作用域中定义了，可以直接使用
        # 不需要从results[name]中获取

        # 应用校准到外部验证集
        if use_calibration and hasattr(model, 'calibrator'):
            y_external_proba_uncalibrated = model.predict_proba(X_external)[:, 1]
            y_external_proba = apply_calibration(y_external_proba_uncalibrated, model.calibrator)
            # 使用相同的优化阈值
            y_external_pred = (y_external_proba >= optimal_threshold).astype(int)
        else:
            y_external_proba = model.predict_proba(X_external)[:, 1]
            # 使用相同的优化阈值
            y_external_pred = (y_external_proba >= optimal_threshold).astype(int)

        # 保存外部验证预测概率
        external_predictions_dict[name] = {
            'PatientID': patient_ids_external,
            'True_Label': y_external,
            'Predicted_Probability': y_external_proba,
            'Predicted_Class': y_external_pred
        }

        # 计算外部验证集的平衡指标
        external_balanced_metrics = calculate_balanced_metrics(y_external, y_external_pred, y_external_proba)

        # 计算置信区间
        ext_auc_roc = external_balanced_metrics['auc_roc']
        ext_auc_roc_ci = calculate_confidence_interval(y_external, y_external_proba, roc_auc_score)

        ext_auc_pr = external_balanced_metrics['auc_pr']
        ext_auc_pr_ci = calculate_confidence_interval(y_external, y_external_proba, average_precision_score)

        ext_brier = external_balanced_metrics['brier_score']
        ext_brier_ci = calculate_confidence_interval(y_external, y_external_proba, brier_score_loss)

        # 构建结果字典
        external_results[name] = {
            'y_pred': y_external_pred,
            'y_proba': y_external_proba,
            'optimal_threshold': optimal_threshold,  # 保存使用的阈值
            **external_balanced_metrics,  # 包含所有平衡指标
            'auc_roc_ci': ext_auc_roc_ci,
            'auc_pr_ci': ext_auc_pr_ci,
            'brier_ci': ext_brier_ci
        }

        print(f"  外部验证集 - AUC-ROC: {ext_auc_roc:.4f} (95% CI: {ext_auc_roc_ci[0]:.4f}-{ext_auc_roc_ci[1]:.4f})")
        print(f"  外部验证集 - AUC-PR: {ext_auc_pr:.4f} (95% CI: {ext_auc_pr_ci[0]:.4f}-{ext_auc_pr_ci[1]:.4f})")
        print(f"  外部验证集 - F2-Score: {external_balanced_metrics['f2_score']:.4f}")
        print(f"  外部验证集 - G-Mean: {external_balanced_metrics['g_mean']:.4f}")
        print(f"  外部验证集 - Sensitivity: {external_balanced_metrics['sensitivity']:.4f}")
        print(f"  外部验证集 - Specificity: {external_balanced_metrics['specificity']:.4f}")

# 循环结束

# 在第1359行（模型训练循环结束）后添加：

# 生成SMOTE策略选择汇总
if use_smote and smote_selection_results:
    print("\n" + "=" * 80)
    print("SMOTE策略选择汇总报告")
    print("=" * 80)

    # 创建汇总DataFrame
    smote_summary_data = []
    for model_name, info in smote_selection_results.items():
        smote_summary_data.append({
            'Model': model_name,
            'Best_Strategy': info['best_strategy'],
            'Validation_Recall': info['best_recall'],
            'Original_Samples': info['original_samples'],
            'Original_Minority': info['original_minority'],
            'Final_Samples': info['final_samples'],
            'Final_Minority': info['final_minority'],
            'Minority_Increase': info['final_minority'] - info['original_minority'],
            'Balance_Ratio': info['final_minority'] / (info['final_samples'] - info['final_minority']) if info[
                                                                                                              'final_samples'] >
                                                                                                          info[
                                                                                                              'final_minority'] else 1
        })

    smote_summary_df = pd.DataFrame(smote_summary_data)

    # 按召回率排序
    smote_summary_df = smote_summary_df.sort_values('Validation_Recall', ascending=False)

    print("\n各模型SMOTE策略选择结果：")
    print(smote_summary_df.to_string(index=False))

    # 保存到CSV
    smote_summary_df.to_csv(
        os.path.join(results_dir, 'smote_strategy_selection_summary.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"\n✅ SMOTE策略选择汇总已保存至: smote_strategy_selection_summary.csv")

    # 创建所有策略对比的详细表
    all_strategies_data = []
    for model_name, info in smote_selection_results.items():
        for strategy in info['strategies_tested']:
            if 'error' not in strategy:
                all_strategies_data.append({
                    'Model': model_name,
                    'Strategy': strategy['strategy'],
                    'Validation_Recall': strategy['validation_recall'],
                    'Validation_Precision': strategy.get('validation_precision', 0),
                    'Validation_F1': strategy.get('validation_f1', 0),
                    'Resampled_Size': strategy['resampled_size'],
                    'Minority_Samples': strategy['minority_samples'],
                    'Is_Selected': strategy['strategy'] == info['best_strategy']
                })

    if all_strategies_data:
        all_strategies_df = pd.DataFrame(all_strategies_data)
        all_strategies_df = all_strategies_df.sort_values(['Model', 'Validation_Recall'], ascending=[True, False])

        all_strategies_df.to_csv(
            os.path.join(results_dir, 'smote_all_strategies_comparison.csv'),
            index=False, encoding='utf-8-sig'
        )
        print(f"✅ 所有SMOTE策略对比已保存至: smote_all_strategies_comparison.csv")

        # 统计最受欢迎的策略
        print("\n策略选择统计：")
        strategy_counts = smote_summary_df['Best_Strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: 被{count}个模型选择")

# ===================== 改进的模型选择策略（评分机制） =====================
print("\n" + "=" * 60)
print("模型选择分析（评分机制）")
print("=" * 60)


def calculate_comprehensive_model_scores_improved(results, external_results=None):
    """
    全面改进的模型评分机制 - 筛查工具导向

    改进要点：
    1. 内外部验证集同等重要（各占40%，稳定性20%）
    2. 所有10个指标都在内外部进行评估
    3. 筛查工具硬性门槛要求
    4. 全面的稳定性评估（使用非对称评分）
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # 初始化评分表
    model_names = list(results.keys())

    # ========== PART 1: 定义筛查工具的硬性要求 ==========
    SCREENING_REQUIREMENTS = {
        'min_internal_sensitivity': 0.70,  # 内部验证集最低敏感性
        'min_internal_npv': 0.85,  # 内部验证集最低NPV
        'min_external_sensitivity': 0.65,  # 外部验证集最低敏感性（略低）
        'min_external_npv': 0.80,  # 外部验证集最低NPV（略低）
        'max_sensitivity_drop': 0.15,  # 内外部敏感性最大允许下降
        'max_npv_drop': 0.10  # 内外部NPV最大允许下降
    }

    # ========== PART 2: 定义评分权重体系 ==========
    METRIC_WEIGHTS = {
        # 第一梯队 - 核心筛查指标 (55%)
        'sensitivity': 0.30,  # 最重要：减少漏诊
        'npv': 0.15,  # 次重要：阴性预测价值
        'f3': 0.10,  # F3比F2更重视召回率

        # 第二梯队 - 平衡指标 (30%)
        'auc_pr': 0.12,  # 不平衡数据的重要指标
        'detection_rate': 0.08,  # 实际检出率
        'mcc': 0.05,  # 整体平衡性
        'g_mean': 0.05,  # 几何平均

        # 第三梯队 - 辅助指标 (15%)
        'brier': 0.05,  # 校准度
        'specificity': 0.05,  # 特异性
        'ppv': 0.03,  # 阳性预测值
        'lr_negative': 0.02  # 阴性似然比
    }

    # ========== PART 3: 提取并标准化所有指标 ==========
    def extract_metrics(results_dict, model_names_list):
        """提取所有模型的指标"""
        metrics = {}

        # 基础指标
        metrics['sensitivity'] = [results_dict[m]['sensitivity'] for m in model_names_list]
        metrics['specificity'] = [results_dict[m]['specificity'] for m in model_names_list]
        metrics['npv'] = [results_dict[m]['npv'] for m in model_names_list]
        metrics['ppv'] = [results_dict[m]['ppv'] for m in model_names_list]
        metrics['f3'] = [results_dict[m]['f3_score'] for m in model_names_list]
        metrics['auc_pr'] = [results_dict[m]['auc_pr'] for m in model_names_list]
        metrics['auc_roc'] = [results_dict[m]['auc_roc'] for m in model_names_list]
        metrics['mcc'] = [results_dict[m]['mcc'] for m in model_names_list]
        metrics['g_mean'] = [results_dict[m]['g_mean'] for m in model_names_list]
        metrics['detection_rate'] = [results_dict[m]['detection_rate'] for m in model_names_list]
        metrics['brier'] = [results_dict[m]['brier_score'] for m in model_names_list]

        # 处理可能的无穷值
        metrics['lr_negative'] = []
        for m in model_names_list:
            lr_neg = results_dict[m]['lr_negative']
            if lr_neg == np.inf or lr_neg > 100:
                lr_neg = 100  # 封顶处理
            metrics['lr_negative'].append(lr_neg)

        return metrics

    # 提取内部验证集指标
    internal_metrics = extract_metrics(results, model_names)

    # 提取外部验证集指标（如果有）
    if external_results is not None:
        external_metrics = extract_metrics(external_results, model_names)
        has_external = True
    else:
        external_metrics = None
        has_external = False

    # ========== PART 4: 标准化指标到0-100分 ==========
    scaler = MinMaxScaler(feature_range=(0, 100))  # 确保是0-100范围

    def safe_scale(values, higher_better=True):
        """安全的标准化处理"""
        values_array = np.array(values).reshape(-1, 1)

        if np.std(values_array) == 0:
            return np.full(len(values), 50.0)

        if not higher_better:
            # 对于越低越好的指标（Brier Score, LR Negative）
            # 方法：先标准化，然后反转
            scaled = scaler.fit_transform(values_array).flatten()
            return 100 - scaled  # 反转：原来的0分变100分，100分变0分

        # 对于越高越好的指标，直接返回标准化结果
        return scaler.fit_transform(values_array).flatten()

    # ========== PART 5: 计算内部验证集得分 ==========
    internal_scores = {}
    for metric in METRIC_WEIGHTS.keys():
        if metric in internal_metrics:
            # Brier score和LR negative越低越好
            higher_better = metric not in ['brier', 'lr_negative']
            internal_scores[metric] = safe_scale(internal_metrics[metric], higher_better)
        else:
            internal_scores[metric] = np.full(len(model_names), 50.0)

    # ========== PART 6: 计算外部验证集得分（如果有） ==========
    if has_external:
        external_scores = {}
        for metric in METRIC_WEIGHTS.keys():
            if metric in external_metrics:
                higher_better = metric not in ['brier', 'lr_negative']
                external_scores[metric] = safe_scale(external_metrics[metric], higher_better)
            else:
                external_scores[metric] = np.full(len(model_names), 50.0)

    # ========== PART 7: 计算稳定性得分（改进版，使用非对称评分） ==========
    stability_scores = []
    stability_details = []  # 保存详细信息用于报告

    # ##### 在这里添加调试代码 #####
    if has_external:
        print(f"\n稳定性分析：处理{len(model_names)}个模型")
        for i, model in enumerate(model_names):
            print(f"  {i + 1}. {model}: ", end="")
            # 这里只是预检查，实际的stability_details还没有生成
            # 所以改为检查是否有外部结果数据
            if model in external_results:
                print(f"有外部验证数据")
            else:
                print(f"缺少外部验证数据")
    # ##### 调试代码结束 #####

    if has_external:
        for i, model in enumerate(model_names):
            stability_components = {}

            # 定义关键指标及其重要性权重
            key_metrics_weights = {
                'sensitivity': 0.35,  # 最重要
                'npv': 0.25,  # 次重要
                'f3': 0.15,
                'auc_pr': 0.15,
                'mcc': 0.10
            }

            weighted_stability = 0

            for metric, weight in key_metrics_weights.items():
                if metric in internal_metrics and metric in external_metrics:
                    int_val = internal_metrics[metric][i]
                    ext_val = external_metrics[metric][i]

                    # 使用非对称稳定性计算
                    metric_stability = calculate_stability_asymmetric(
                        int_val, ext_val, metric
                    )

                    stability_components[metric] = {
                        'internal': int_val,
                        'external': ext_val,
                        'change': (ext_val - int_val) / int_val if int_val > 0 else 0,
                        'stability': metric_stability
                    }

                    weighted_stability += metric_stability * weight

            stability_scores.append(weighted_stability * 100)  # 转换到0-100分
            stability_details.append(stability_components)

            # 输出稳定性分析（用于调试，只显示前3个模型）
            if i < 3:
                print(f"\n{model} 稳定性分析:")
                for metric, details in stability_components.items():
                    if metric in details:  # 确保键存在
                        change_pct = details['change'] * 100
                        direction = '↑' if change_pct > 0 else '↓'
                        print(f"  {metric}: {details['internal']:.3f} → {details['external']:.3f} "
                              f"({direction}{abs(change_pct):.1f}%) | 稳定性: {details['stability']:.2f}")
    else:
        stability_scores = [50] * len(model_names)
        stability_details = [{}] * len(model_names)

    # ========== PART 8: 应用硬性门槛检查 ==========
    penalty_factors = []
    disqualified_models = []

    for i, model in enumerate(model_names):
        penalty = 1.0  # 初始无惩罚
        reasons = []

        # 检查内部验证集硬性要求
        if internal_metrics['sensitivity'][i] < SCREENING_REQUIREMENTS['min_internal_sensitivity']:
            penalty *= 0.5
            reasons.append(f"内部敏感性不足({internal_metrics['sensitivity'][i]:.3f})")

        if internal_metrics['npv'][i] < SCREENING_REQUIREMENTS['min_internal_npv']:
            penalty *= 0.7
            reasons.append(f"内部NPV不足({internal_metrics['npv'][i]:.3f})")

        # 检查外部验证集硬性要求（如果有）
        if has_external:
            if external_metrics['sensitivity'][i] < SCREENING_REQUIREMENTS['min_external_sensitivity']:
                penalty *= 0.5
                reasons.append(f"外部敏感性不足({external_metrics['sensitivity'][i]:.3f})")

            if external_metrics['npv'][i] < SCREENING_REQUIREMENTS['min_external_npv']:
                penalty *= 0.7
                reasons.append(f"外部NPV不足({external_metrics['npv'][i]:.3f})")

            # 检查稳定性要求
            sens_drop = internal_metrics['sensitivity'][i] - external_metrics['sensitivity'][i]
            if sens_drop > SCREENING_REQUIREMENTS['max_sensitivity_drop']:
                penalty *= 0.8
                reasons.append(f"敏感性下降过大({sens_drop:.3f})")

            npv_drop = internal_metrics['npv'][i] - external_metrics['npv'][i]
            if npv_drop > SCREENING_REQUIREMENTS['max_npv_drop']:
                penalty *= 0.8
                reasons.append(f"NPV下降过大({npv_drop:.3f})")

        penalty_factors.append(penalty)

        if reasons:
            disqualified_models.append({
                'model': model,
                'penalty': penalty,
                'reasons': '; '.join(reasons)
            })

    # ========== PART 9: 计算综合得分 ==========
    total_scores = []

    if has_external:
        # 有外部验证时：内部40% + 外部40% + 稳定性20%
        for i in range(len(model_names)):
            # 内部得分（加权）
            internal_weighted = sum(
                internal_scores[metric][i] * METRIC_WEIGHTS.get(metric, 0)
                for metric in internal_scores
            )

            # 外部得分（加权）
            external_weighted = sum(
                external_scores[metric][i] * METRIC_WEIGHTS.get(metric, 0)
                for metric in external_scores
            )

            # 综合得分
            total = (
                    internal_weighted * 0.4 +
                    external_weighted * 0.4 +
                    stability_scores[i] * 0.2
            )

            # 应用惩罚因子
            total *= penalty_factors[i]

            total_scores.append(total)
    else:
        # 仅有内部验证时：内部100%
        for i in range(len(model_names)):
            internal_weighted = sum(
                internal_scores[metric][i] * METRIC_WEIGHTS.get(metric, 0)
                for metric in internal_scores
            )
            total = internal_weighted * penalty_factors[i]
            total_scores.append(total)

    # ========== PART 10: 创建详细评分表 ==========
    detailed_scores = {
        'Model': model_names,
        'Total_Score': total_scores,
        'Penalty_Factor': penalty_factors
    }

    # 添加内部验证集各项得分（0-100分）
    for metric in METRIC_WEIGHTS.keys():
        if metric in internal_scores:
            col_name = f'{metric.capitalize()}_Score'
            detailed_scores[col_name] = internal_scores[metric]

    # 添加外部验证集各项得分（如果有）
    if has_external:
        for metric in METRIC_WEIGHTS.keys():
            if metric in external_scores:
                col_name = f'External_{metric}_Score'
                detailed_scores[col_name] = external_scores[metric]

        # 添加稳定性和泛化得分
        detailed_scores['Stability_Score'] = stability_scores

    # 添加原始值
    for metric in internal_metrics:
        detailed_scores[f'Internal_{metric}_Raw'] = internal_metrics[metric]
        detailed_scores[f'Internal_{metric}'] = internal_metrics[metric]  # 兼容旧代码

    if has_external:
        for metric in external_metrics:
            detailed_scores[f'External_{metric}_Raw'] = external_metrics[metric]
            detailed_scores[f'External_{metric}'] = external_metrics[metric]  # 兼容旧代码

    # 创建DataFrame
    score_df = pd.DataFrame(detailed_scores)

    # 按总分排序
    score_df = score_df.sort_values('Total_Score', ascending=False)
    score_df['Rank'] = range(1, len(score_df) + 1)

    # ========== PART 11: 生成评分报告 ==========
    print("\n" + "=" * 80)
    print("改进版模型综合评分报告（筛查工具优化）")
    print("=" * 80)

    print("\n评分体系说明:")
    print("-" * 60)
    if has_external:
        print("评分构成: 内部验证40% + 外部验证40% + 稳定性20%")
        print("稳定性评分: 使用非对称评分（轻微提升=好，大幅变化=需调查）")
    else:
        print("评分构成: 内部验证100%（无外部验证数据）")

    print("\n指标权重分配:")
    print("第一梯队（核心筛查指标 55%）:")
    print(f"  - Sensitivity: 30%")
    print(f"  - NPV: 15%")
    print(f"  - F3-Score: 10%")
    print("第二梯队（平衡指标 30%）:")
    print(f"  - AUC-PR: 12%")
    print(f"  - Detection Rate: 8%")
    print(f"  - MCC: 5%")
    print(f"  - G-Mean: 5%")
    print("第三梯队（辅助指标 15%）:")
    print(f"  - Brier Score: 5%")
    print(f"  - Specificity: 5%")
    print(f"  - PPV: 3%")
    print(f"  - LR Negative: 2%")

    if disqualified_models:
        print("\n受惩罚的模型:")
        print("-" * 60)
        for item in disqualified_models[:5]:  # 只显示前5个
            print(f"  {item['model']}: 惩罚系数={item['penalty']:.2f}")
            print(f"    原因: {item['reasons']}")

    print("\n" + "-" * 60)
    print("Top 5 模型详细评分:")
    print("-" * 60)

    for idx in range(min(5, len(score_df))):
        row = score_df.iloc[idx]
        print(f"\n排名 {row['Rank']}: {row['Model']}")
        print(f"  总分: {row['Total_Score']:.2f}")

        if has_external:
            # 显示稳定性详情
            if idx < len(stability_details) and stability_details[idx]:
                print(f"  稳定性分析:")
                for metric in ['sensitivity', 'npv']:
                    if metric in stability_details[idx]:
                        detail = stability_details[idx][metric]
                        change_pct = detail['change'] * 100
                        interpretation = interpret_stability_score(
                            detail['stability'], change_pct, metric
                        )
                        print(f"    {metric}: {interpretation}")

        print(f"  惩罚系数: {row['Penalty_Factor']:.2f}")

    # 获取最佳模型
    best_model = score_df.iloc[0]['Model']

    # ========== PART 12: 保存结果 ==========
    score_df.to_csv(os.path.join(results_dir, 'model_scores_improved_comprehensive.csv'),
                    index=False, encoding='utf-8-sig')

    print(f"\n✅ 改进版评分结果已保存")

    # ============ 在这里添加新代码 ============
    # 创建模型名称到stability_details的映射字典
    stability_dict = {}
    for i, model_name in enumerate(model_names):
        if i < len(stability_details):
            stability_dict[model_name] = stability_details[i]
        else:
            stability_dict[model_name] = {}

    # 修改返回值，添加stability_dict（替换原来的return语句）
    return best_model, score_df, stability_details, stability_dict  # 添加了stability_dict


# ===================== 执行评分并生成可视化 =====================

# 计算综合评分
# 执行改进版综合评分机制...
print("\n执行改进版综合评分机制...")
# 修改：接收4个返回值（添加了stability_dict）
best_model_name, score_df, stability_details, stability_dict = calculate_comprehensive_model_scores_improved(
    results,
    external_results if external_validation_available else None
)

# ===================== 定义稳定性分析相关函数 =====================
# 注意：这些函数必须在调用之前定义

def determine_quadrant(sens_change, npv_change):
    """确定变化所在的象限"""
    if abs(sens_change) <= 10 and abs(npv_change) <= 10:
        return 'Stable (Green Zone)'
    elif sens_change > 0 and npv_change > 0:
        return 'Both Improved'
    elif sens_change < 0 and npv_change < 0:
        return 'Both Declined'
    elif sens_change > 0 and npv_change < 0:
        return 'Sensitivity Up, NPV Down'
    else:
        return 'Sensitivity Down, NPV Up'


def interpret_overall_stability(stability_score):
    """解释整体稳定性得分"""
    if stability_score >= 0.8:
        return 'Excellent'
    elif stability_score >= 0.6:
        return 'Good'
    elif stability_score >= 0.4:
        return 'Fair'
    else:
        return 'Poor'


def plot_stability_analysis_corrected(score_df, stability_dict, top_n=15):
    """
    修正版：使用字典而非列表索引来确保数据匹配正确
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 敏感性变化瀑布图
    ax1 = axes[0, 0]
    models = score_df['Model'].head(top_n).values
    sens_changes = []
    colors = []

    print("\n" + "=" * 80)
    print("Stability Analysis - 敏感性指标提取详情（修正版）")
    print("=" * 80)
    print(f"{'模型名称':<30} {'内部敏感性':>12} {'外部敏感性':>12} {'变化(%)':>10} {'稳定性评分':>12}")
    print("-" * 80)

    for model_name in models:
        # 直接从字典中获取该模型的数据
        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]
            if 'sensitivity' in detail:
                internal_sens = detail['sensitivity']['internal']
                external_sens = detail['sensitivity']['external']
                change = detail['sensitivity']['change'] * 100
                stability_score = detail['sensitivity']['stability']

                print(
                    f"{model_name:<30} {internal_sens:>12.4f} {external_sens:>12.4f} {change:>+10.2f} {stability_score:>12.4f}")

                sens_changes.append(change)
                # 颜色编码
                if abs(change) <= 10:
                    colors.append('green')
                elif abs(change) <= 20:
                    colors.append('orange')
                else:
                    colors.append('red')
            else:
                sens_changes.append(0)
                colors.append('gray')
                print(f"{model_name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>12}")
        else:
            sens_changes.append(0)
            colors.append('gray')
            print(f"{model_name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>12}")

    print("-" * 80)
    print("=" * 80 + "\n")

    bars = ax1.bar(range(len(models)), sens_changes, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='±10% (Good)')
    ax1.axhline(y=-10, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='±20% (Caution)')
    ax1.axhline(y=-20, color='orange', linestyle='--', alpha=0.5)

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Change (%)')
    ax1.set_title('Sensitivity: External vs Internal Change\n(Green=Stable, Orange=Caution, Red=Unstable)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加数值标注
    for i, (bar, val) in enumerate(zip(bars, sens_changes)):
        if val != 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, val,
                     f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                     fontsize=8, fontweight='bold')

    # 2. 稳定性得分热图
    ax2 = axes[0, 1]

    metrics = ['sensitivity', 'npv', 'f3', 'auc_pr', 'mcc']
    heatmap_data = []
    annotations = []

    for model_name in models:
        row_scores = []
        row_annotations = []

        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]
            for metric in metrics:
                if metric in detail:
                    score = detail[metric]['stability']
                    change = detail[metric]['change'] * 100
                    row_scores.append(score)
                    arrow = '↑' if change > 0 else '↓' if change < 0 else '='
                    row_annotations.append(f'{score:.2f}\n{arrow}{abs(change):.0f}%')
                else:
                    row_scores.append(0.5)
                    row_annotations.append('N/A')
        else:
            row_scores = [0.5] * len(metrics)
            row_annotations = ['N/A'] * len(metrics)

        heatmap_data.append(row_scores)
        annotations.append(row_annotations)

    im = ax2.imshow(np.array(heatmap_data).T, cmap='RdYlGn', aspect='auto',
                    vmin=0, vmax=1)

    ax2.set_xticks(range(len(models)))
    ax2.set_yticks(range(len(metrics)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_yticklabels([m.upper() for m in metrics])

    # 添加详细标注
    for i in range(len(metrics)):
        for j in range(len(heatmap_data)):
            text = annotations[j][i]
            color = 'white' if heatmap_data[j][i] < 0.5 else 'black'
            ax2.text(j, i, text, ha='center', va='center',
                     color=color, fontsize=7)

    ax2.set_title('Stability Scores with Direction\n(↑=Improved, ↓=Declined)')
    plt.colorbar(im, ax=ax2, label='Stability Score')

    # 3. 稳定性分类散点图
    ax3 = axes[1, 0]

    sens_changes_all = []
    npv_changes_all = []
    model_names_all = []

    for model_name in models:
        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]

            sens_change = 0
            if 'sensitivity' in detail:
                sens_change = detail['sensitivity']['change'] * 100

            npv_change = 0
            if 'npv' in detail:
                npv_change = detail['npv']['change'] * 100
        else:
            sens_change = 0
            npv_change = 0

        sens_changes_all.append(sens_change)
        npv_changes_all.append(npv_change)
        model_names_all.append(model_name)

    if sens_changes_all:
        # 创建分区背景
        ax3.axhspan(-10, 10, alpha=0.1, color='green')
        ax3.axvspan(-10, 10, alpha=0.1, color='green')

        # 绘制所有点
        for i in range(len(model_names_all)):
            ax3.scatter(sens_changes_all[i], npv_changes_all[i],
                        s=200, c=plt.cm.viridis(i / len(model_names_all)),
                        alpha=0.7, edgecolors='black', linewidth=2,
                        zorder=10 + i)

            # 在圆圈内添加编号
            ax3.text(sens_changes_all[i], npv_changes_all[i], f'{i + 1}',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white', zorder=20 + i)

        # 在图的右侧添加图例
        legend_text = '\n'.join([f'{i + 1}: {name[:20]}' for i, name in enumerate(model_names_all)])
        ax3.text(1.02, 0.5, legend_text, transform=ax3.transAxes,
                 fontsize=7, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Sensitivity Change (%)', fontsize=11)
        ax3.set_ylabel('NPV Change (%)', fontsize=11)
        ax3.set_title('Stability Classification\n(Numbers in circles, see legend for models)')
        ax3.grid(True, alpha=0.3)

        # 动态调整坐标轴范围
        x_min = min(sens_changes_all) if sens_changes_all else -10
        x_max = max(sens_changes_all) if sens_changes_all else 10
        y_min = min(npv_changes_all) if npv_changes_all else -10
        y_max = max(npv_changes_all) if npv_changes_all else 10

        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 5
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 5

        x_limit = max(30, abs(x_min - x_margin), abs(x_max + x_margin))
        y_limit = max(30, abs(y_min - y_margin), abs(y_max + y_margin))

        ax3.set_xlim(-x_limit, x_limit)
        ax3.set_ylim(-y_limit, y_limit)

    # 4. 稳定性评分分布
    ax4 = axes[1, 1]

    if 'Stability_Score' in score_df.columns:
        stability_scores = score_df['Stability_Score'].head(top_n).values
        model_names_15 = score_df['Model'].head(top_n).values

        colors_hist = ['green' if s >= 70 else 'orange' if s >= 50 else 'red'
                       for s in stability_scores]

        bars = ax4.bar(range(len(stability_scores)), stability_scores,
                       color=colors_hist, alpha=0.7, edgecolor='black')

        # 添加数值标注
        for i, (bar, score) in enumerate(zip(bars, stability_scores)):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=8)

        ax4.set_xticks(range(len(model_names_15)))
        ax4.set_xticklabels(model_names_15, rotation=45, ha='right', fontsize=8)
        ax4.set_xlabel('Model', fontsize=11)
        ax4.set_ylabel('Stability Score (0-100)', fontsize=11)
        ax4.set_title(f'Overall Stability Score Distribution (Top {top_n} Models)')
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Comprehensive Stability Analysis (Corrected)\n(Asymmetric Scoring: Slight improvement is good, large changes need investigation)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def export_stability_analysis_to_csv_corrected(score_df, stability_dict, top_n=15, results_dir=''):
    """
    修正版：使用字典确保数据正确匹配，保留所有原有功能
    """
    import pandas as pd
    import numpy as np

    # 1. 导出敏感性变化数据（对应瀑布图）
    sensitivity_changes_data = []
    models = score_df['Model'].head(top_n).values

    for model_name in models:
        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]

            if 'sensitivity' in detail:
                internal = detail['sensitivity']['internal']
                external = detail['sensitivity']['external']
                change = detail['sensitivity']['change'] * 100
                stability = detail['sensitivity']['stability']
            else:
                internal = external = change = stability = np.nan
        else:
            internal = external = change = stability = np.nan

        sensitivity_changes_data.append({
            'Model': model_name,
            'Internal_Sensitivity': internal,
            'External_Sensitivity': external,
            'Change_Percent': change,
            'Stability_Score': stability,
            'Change_Category': 'Good' if abs(change) <= 10 else 'Caution' if abs(change) <= 20 else 'Unstable'
        })

    sensitivity_df = pd.DataFrame(sensitivity_changes_data)
    sensitivity_df.to_csv(os.path.join(results_dir, 'stability_sensitivity_changes_corrected.csv'),
                          index=False, encoding='utf-8-sig')

    # 2. 导出稳定性得分热图数据
    metrics = ['sensitivity', 'npv', 'f3', 'auc_pr', 'mcc']
    heatmap_data = []

    for model_name in models:
        row_data = {'Model': model_name}

        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]

            for metric in metrics:
                if metric in detail:
                    row_data[f'{metric}_stability'] = detail[metric]['stability']
                    row_data[f'{metric}_change_pct'] = detail[metric]['change'] * 100
                    row_data[f'{metric}_internal'] = detail[metric]['internal']
                    row_data[f'{metric}_external'] = detail[metric]['external']
                else:
                    row_data[f'{metric}_stability'] = np.nan
                    row_data[f'{metric}_change_pct'] = np.nan
                    row_data[f'{metric}_internal'] = np.nan
                    row_data[f'{metric}_external'] = np.nan
        else:
            for metric in metrics:
                row_data[f'{metric}_stability'] = np.nan
                row_data[f'{metric}_change_pct'] = np.nan
                row_data[f'{metric}_internal'] = np.nan
                row_data[f'{metric}_external'] = np.nan

        heatmap_data.append(row_data)

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.to_csv(os.path.join(results_dir, 'stability_heatmap_data_corrected.csv'),
                      index=False, encoding='utf-8-sig')

    # 3. 导出稳定性分类散点图数据（Sensitivity vs NPV changes）
    scatter_data = []

    for i, model_name in enumerate(models):
        sens_change = 0
        npv_change = 0

        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]
            if 'sensitivity' in detail:
                sens_change = detail['sensitivity']['change'] * 100
            if 'npv' in detail:
                npv_change = detail['npv']['change'] * 100

        scatter_data.append({
            'Model': model_name,
            'Model_Index': i + 1,
            'Sensitivity_Change_Percent': sens_change,
            'NPV_Change_Percent': npv_change,
            'Quadrant': determine_quadrant(sens_change, npv_change)
        })

    scatter_df = pd.DataFrame(scatter_data)
    scatter_df.to_csv(os.path.join(results_dir, 'stability_scatter_data_corrected.csv'),
                      index=False, encoding='utf-8-sig')

    # 4. 导出整体稳定性得分分布数据
    if 'Stability_Score' in score_df.columns:
        stability_distribution = score_df[['Model', 'Rank', 'Total_Score', 'Stability_Score']].head(top_n).copy()
        stability_distribution['Stability_Category'] = pd.cut(
            stability_distribution['Stability_Score'],
            bins=[0, 50, 70, 100],
            labels=['Poor', 'Fair', 'Good']
        )
        stability_distribution.to_csv(os.path.join(results_dir, 'stability_distribution_corrected.csv'),
                                      index=False, encoding='utf-8-sig')
    else:
        stability_distribution = None

    # 5. 创建综合汇总表
    summary_data = []

    for i, model_name in enumerate(models):
        # 计算各指标的平均稳定性
        avg_stability = 0
        count = 0
        max_change = 0
        worst_metric = ''

        if model_name in stability_dict and stability_dict[model_name]:
            detail = stability_dict[model_name]
            for metric in ['sensitivity', 'npv', 'f3', 'auc_pr', 'mcc']:
                if metric in detail:
                    avg_stability += detail[metric]['stability']
                    count += 1
                    change = abs(detail[metric]['change'] * 100)
                    if change > max_change:
                        max_change = change
                        worst_metric = metric

        if count > 0:
            avg_stability = avg_stability / count

        # 获取Overall_Stability_Score
        overall_stability = np.nan
        if 'Stability_Score' in score_df.columns and i < len(score_df):
            overall_stability = score_df.iloc[i]['Stability_Score']

        summary_data.append({
            'Model': model_name,
            'Rank': i + 1,
            'Overall_Stability_Score': overall_stability,
            'Avg_Metric_Stability': avg_stability,
            'Max_Change_Percent': max_change,
            'Most_Unstable_Metric': worst_metric,
            'Interpretation': interpret_overall_stability(avg_stability)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'stability_summary_corrected.csv'),
                      index=False, encoding='utf-8-sig')

    print("\n✅ 修正后的稳定性分析数据已导出为CSV文件：")
    print("   1. stability_sensitivity_changes_corrected.csv - 敏感性变化数据")
    print("   2. stability_heatmap_data_corrected.csv - 热图完整数据")
    print("   3. stability_scatter_data_corrected.csv - 散点图数据")
    print("   4. stability_distribution_corrected.csv - 稳定性得分分布")
    print("   5. stability_summary_corrected.csv - 综合汇总表")

    return sensitivity_df, None  # 返回简化，只返回两个值

# ===================== 函数定义结束 =====================

# 现在可以安全地调用这些函数了

# 生成稳定性分析可视化（使用修正后的函数）
if external_validation_available:
    print("\n生成稳定性分析可视化（修正版）...")
    fig = plot_stability_analysis_corrected(score_df, stability_dict, top_n=15)
    plt.savefig(os.path.join(results_dir, 'stability_analysis_asymmetric_corrected.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 修正后的稳定性分析可视化已保存")

    # 导出修正后的CSV数据
    sensitivity_df, full_df = export_stability_analysis_to_csv_corrected(
        score_df, stability_dict, top_n=15, results_dir=results_dir
    )

# ===================== 步骤5：兼容性处理（在这里！） =====================
# 添加兼容性处理，确保列名存在
if 'F3_Score' not in score_df.columns and 'F3_score' in score_df.columns:
    score_df['F3_Score'] = score_df['F3_score']
if 'Sensitivity_Score' not in score_df.columns and 'Sensitivity_score' in score_df.columns:
    score_df['Sensitivity_Score'] = score_df['Sensitivity_score']
if 'Npv_Score' not in score_df.columns and 'Npv_score' in score_df.columns:
    score_df['NPV_Score'] = score_df['Npv_score']
if 'Auc_pr_Score' not in score_df.columns and 'Auc_pr_score' in score_df.columns:
    score_df['AUC_PR_Score'] = score_df['Auc_pr_score']
if 'Detection_rate_Score' not in score_df.columns and 'Detection_rate_score' in score_df.columns:
    score_df['Detection_Rate_Score'] = score_df['Detection_rate_score']
if 'Mcc_Score' not in score_df.columns and 'Mcc_score' in score_df.columns:
    score_df['MCC_Score'] = score_df['Mcc_score']
if 'Specificity_Score' not in score_df.columns and 'Specificity_score' in score_df.columns:
    score_df['Specificity_Score'] = score_df['Specificity_score']
if 'Lr_negative_Score' not in score_df.columns and 'Lr_negative_score' in score_df.columns:
    score_df['LR_Negative_Score'] = score_df['Lr_negative_score']
if 'Brier_Score' not in score_df.columns and 'Brier_score' in score_df.columns:
    score_df['Brier_Score'] = score_df['Brier_score']
if 'G_mean_Score' not in score_df.columns and 'G_mean_score' in score_df.columns:
    score_df['G_Mean_Score'] = score_df['G_mean_score']
if 'Ppv_Score' not in score_df.columns and 'Ppv_score' in score_df.columns:
    score_df['PPV_Score'] = score_df['Ppv_score']

# ===================== 完整的兼容性修复代码 =====================
print("\n处理DataFrame列名兼容性...")


# 定义修复函数
def fix_score_df_columns(score_df):
    """修复score_df的列名，确保与原可视化函数兼容"""

    # 打印当前列名用于调试
    print("当前列名:", score_df.columns.tolist()[:20])  # 只显示前20个

    # 定义所有需要的标准列名映射
    column_mappings = {
        'sensitivity': 'Sensitivity_Score',
        'npv': 'NPV_Score',
        'f3': 'F3_Score',
        'auc_pr': 'AUC_PR_Score',
        'detection_rate': 'Detection_Rate_Score',
        'mcc': 'MCC_Score',
        'brier': 'Brier_Score',
        'specificity': 'Specificity_Score',
        'lr_negative': 'LR_Negative_Score',
        'g_mean': 'G_Mean_Score',
        'ppv': 'PPV_Score',
        'stability': 'Stability_Score'
    }

    # 应用映射
    for base_name, target_name in column_mappings.items():
        if target_name not in score_df.columns:
            # 查找可能的源列名
            possible_names = [
                base_name + '_score',
                base_name + '_Score',
                base_name.capitalize() + '_score',
                base_name.capitalize() + '_Score',
                base_name.upper() + '_Score'
            ]

            for possible in possible_names:
                if possible in score_df.columns:
                    score_df[target_name] = score_df[possible]
                    print(f"  成功映射: {possible} -> {target_name}")
                    break
            else:
                # 如果找不到，使用默认值
                if target_name != 'Stability_Score':  # Stability_Score可能不存在
                    print(f"  警告: {target_name} 不存在，使用默认值50.0")
                    score_df[target_name] = 50.0

    # 确保基本列存在
    if 'Model' not in score_df.columns:
        print("错误：Model列不存在！")
    if 'Rank' not in score_df.columns and 'Total_Score' in score_df.columns:
        score_df = score_df.sort_values('Total_Score', ascending=False)
        score_df['Rank'] = range(1, len(score_df) + 1)

    return score_df


# 应用修复
score_df = fix_score_df_columns(score_df)
print("列名兼容性处理完成\n")

# 生成详细的评分可视化
def visualize_model_scores_enhanced(score_df):
    """增强的评分可视化"""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 总分条形图（横向）
    ax1 = fig.add_subplot(gs[0, :2])
    models = score_df['Model'].head(15)
    scores = score_df['Total_Score'].head(15)
    colors = ['red' if i == 0 else 'gold' if i == 1 else 'silver' if i == 2 else 'skyblue'
              for i in range(len(models))]

    bars = ax1.barh(range(len(models)), scores, color=colors, edgecolor='navy', linewidth=1)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=10)
    ax1.set_xlabel('Total Score (0-100)', fontsize=12)
    ax1.set_title('Model Rankings by Total Score', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)

    # 添加分数标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{score:.1f}', va='center', fontsize=9, fontweight='bold')

    # 2. 热力图显示各项得分（正确的缩进）
    ax2 = fig.add_subplot(gs[1:, :])

    # 包含所有12个评分列
    score_columns = [
        'Sensitivity_Score',  # 30%
        'NPV_Score',  # 15%
        'F3_Score',  # 10%
        'AUC_PR_Score',  # 12%
        'Detection_Rate_Score',  # 8%
        'MCC_Score',  # 5%
        'G_Mean_Score',  # 5%
        'Brier_Score',  # 5%
        'Specificity_Score',  # 5%
        'PPV_Score',  # 3%
        'LR_Negative_Score',  # 2%
        'Stability_Score'  # 20%（独立）
    ]

    # 检查哪些列实际存在
    available_score_columns = [col for col in score_columns if col in score_df.columns]

    missing_columns = [col for col in score_columns if col not in score_df.columns]
    if missing_columns:
        print(f"警告：以下评分列不存在：{missing_columns}")

    # 创建热力图数据
    heatmap_data = score_df.head(15)[available_score_columns].values.T

    # 权重映射
    weight_map = {
        'Sensitivity_Score': 'Sensitivity (30%)',
        'NPV_Score': 'NPV (15%)',
        'F3_Score': 'F3-Score (10%)',
        'AUC_PR_Score': 'AUC-PR (12%)',
        'Detection_Rate_Score': 'Detection Rate (8%)',
        'MCC_Score': 'MCC (5%)',
        'G_Mean_Score': 'G-Mean (5%)',
        'Brier_Score': 'Brier Score (5%)',
        'Specificity_Score': 'Specificity (5%)',
        'PPV_Score': 'PPV (3%)',
        'LR_Negative_Score': 'LR Negative (2%)',
        'Stability_Score': 'Stability (20% Indep.)'
    }

    yticklabels = [weight_map.get(col, col.replace('_Score', ''))
                   for col in available_score_columns]

    # 使用seaborn绘制热力图
    sns.heatmap(heatmap_data, ax=ax2,
                cmap='RdYlGn',
                vmin=0, vmax=100,
                cbar_kws={'label': 'Score (0-100)'},
                xticklabels=score_df['Model'].head(15),
                yticklabels=yticklabels,
                annot=True,
                fmt='.0f',
                annot_kws={'size': 7},
                linewidths=0.5,
                linecolor='gray')

    ax2.set_title('Detailed Scores Heatmap - All 12 Key Metrics (Top 15 Models)',
                  fontsize=14, fontweight='bold')

    # 更新文本框说明（正确版本）
    textstr = ('Metric Weights:\n'
               '━━━━━━━━━━━\n'
               'Tier 1 (55%):\n'
               '• Sensitivity: 30%\n'
               '• NPV: 15%\n'
               '• F3-Score: 10%\n\n'
               'Tier 2 (30%):\n'
               '• AUC-PR: 12%\n'
               '• Detection: 8%\n'
               '• MCC: 5%\n'
               '• G-Mean: 5%\n\n'  # 不是General
               'Tier 3 (15%):\n'
               '• Brier: 5%\n'
               '• Specificity: 5%\n'
               '• PPV: 3%\n'
               '• LR Neg.: 2%\n\n'
               'Independent:\n'
               '• Stability: 20%*\n'
               '\n*Only with external')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(1.02, 0.5, textstr, transform=ax2.transAxes, fontsize=8,
             verticalalignment='center', bbox=props)

    # 3. 雷达图对比前3名（包含NPV，共6个指标）
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')

    categories = ['Sensitivity', 'NPV', 'AUC-PR', 'F3', 'MCC', 'Specificity']  # 6个指标
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 前3个模型
    colors_radar = ['red', 'blue', 'green']
    for idx in range(min(3, len(score_df))):
        values = [
            score_df.iloc[idx]['Sensitivity_Score'],
            score_df.iloc[idx]['NPV_Score'],  # NPV
            score_df.iloc[idx]['AUC_PR_Score'],
            score_df.iloc[idx]['F3_Score'],
            score_df.iloc[idx]['MCC_Score'],
            score_df.iloc[idx]['Specificity_Score']
        ]
        values += values[:1]  # 闭合

        ax3.plot(angles, values, 'o-', linewidth=2,
                 label=f"#{idx + 1} {score_df.iloc[idx]['Model'][:15]}",
                 color=colors_radar[idx])
        ax3.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=10)
    ax3.set_ylim(0, 100)
    ax3.grid(True)
    ax3.set_title('Top 3 Models Comparison (6 Key Metrics)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.suptitle('Comprehensive Model Scoring Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_scores_visualization_enhanced.pdf'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return fig


# 只调用一次
print("\n生成评分可视化...")
visualize_model_scores_enhanced(score_df)

# ===================== 添加综合指标热图 =====================
print("\n生成综合指标热图...")


def create_comprehensive_heatmap(score_df, results, external_results=None):
    """创建包含所有指标的综合热图"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10),
                                   gridspec_kw={'width_ratios': [3, 1]})

    # 准备数据 - 包含所有12个指标
    primary_metrics = [
        'Sensitivity_Score',      # 30%
        'NPV_Score',              # 15%
        'F3_Score',               # 10%
        'AUC_PR_Score',           # 12%
        'Detection_Rate_Score',   # 8%
        'MCC_Score',              # 5%
        'G_Mean_Score',           # 5%
        'Brier_Score',            # 5%
        'Specificity_Score',      # 5%
        'PPV_Score',              # 3%
        'LR_Negative_Score',      # 2%
        'Stability_Score'         # 20%（独立）
    ]

    # 只使用实际存在的列
    available_primary = [m for m in primary_metrics if m in score_df.columns]

    # 获取前20个模型
    top_models = score_df.head(20)

    # 创建热图数据
    heatmap_data = top_models[available_primary].values.T

    # 主热图
    im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                    vmin=0, vmax=100)

    # 设置刻度
    ax1.set_xticks(np.arange(len(top_models)))
    ax1.set_yticks(np.arange(len(available_primary)))
    ax1.set_xticklabels(top_models['Model'], rotation=45, ha='right', fontsize=8)

    # Y轴标签 - 修正：为所有指标添加权重信息
    y_labels = []
    for metric in available_primary:
        if metric == 'Sensitivity_Score':
            y_labels.append('Sensitivity (30%)')
        elif metric == 'NPV_Score':
            y_labels.append('NPV (15%)')
        elif metric == 'F3_Score':
            y_labels.append('F3-Score (10%)')
        elif metric == 'AUC_PR_Score':
            y_labels.append('AUC-PR (12%)')
        elif metric == 'Detection_Rate_Score':
            y_labels.append('Detection Rate (8%)')
        elif metric == 'MCC_Score':
            y_labels.append('MCC (5%)')
        elif metric == 'G_Mean_Score':
            y_labels.append('G-Mean (5%)')  # 添加权重
        elif metric == 'Brier_Score':
            y_labels.append('Brier Score (5%)')
        elif metric == 'Specificity_Score':
            y_labels.append('Specificity (5%)')
        elif metric == 'PPV_Score':
            y_labels.append('PPV (3%)')  # 添加权重
        elif metric == 'LR_Negative_Score':
            y_labels.append('LR Negative (2%)')
        elif metric == 'Stability_Score':
            y_labels.append('Stability (20% Indep.)')  # 添加权重
        else:
            y_labels.append(metric.replace('_Score', ''))

    ax1.set_yticklabels(y_labels, fontsize=9)

    # 修正：确保所有格子都有数值标注，并放大字体
    for i in range(len(available_primary)):
        for j in range(len(top_models)):
            value = heatmap_data[i, j]
            # 所有格子都显示数值，不再有条件限制
            text_color = "white" if value > 60 else "black"  # 根据背景色调整文字颜色
            text = ax1.text(j, i, f'{value:.0f}',
                            ha="center", va="center",
                            color=text_color,
                            fontsize=8,  # 增大字体从6到8
                            fontweight='bold')  # 加粗

    ax1.set_title('Comprehensive Performance Metrics Heatmap (Top 20 Models)',
                  fontsize=16, fontweight='bold')

    # 添加网格线
    ax1.set_xticks(np.arange(len(top_models)) - .5, minor=True)
    ax1.set_yticks(np.arange(len(available_primary)) - .5, minor=True)
    ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.3)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.04)
    cbar.set_label('Score (0-100)', rotation=270, labelpad=15)

    # 右侧：显示权重表和统计信息
    ax2.axis('off')

    # 创建权重和统计表格
    table_data = [
        ['═══ SCREENING TOOL WEIGHTS ═══'],
        [''],
        ['TIER 1 - Core (55%)'],
        ['  Sensitivity: 30%'],
        ['  NPV: 15%'],
        ['  F3-Score: 10%'],
        [''],
        ['TIER 2 - Balance (30%)'],
        ['  AUC-PR: 12%'],
        ['  Detection Rate: 8%'],
        ['  MCC: 5%'],  # 移到第二梯队
        ['  G-Mean: 5%'],  # 添加G-Mean
        [''],
        ['TIER 3 - Support (15%)'],
        ['  Brier Score: 5%'],
        ['  Specificity: 5%'],
        ['  PPV: 3%'],  # 添加PPV
        ['  LR Negative: 2%'],
        [''],
        ['INDEPENDENT (20%)'],  # 添加独立部分
        ['  Stability: 20%*'],  # Stability作为独立指标
        [''],
        ['═══ TOP MODEL STATS ═══'],
        [''],
        [f'Best Model: {score_df.iloc[0]["Model"]}'],
        [f'Total Score: {score_df.iloc[0]["Total_Score"]:.1f}'],
        [''],
        ['*Only with external validation'],
    ]

    # 在右侧显示文本
    text_str = '\n'.join([row[0] for row in table_data])
    ax2.text(0.1, 0.5, text_str, transform=ax2.transAxes,
             fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Complete Model Performance Analysis with Screening Tool Weights\n' +
                 f'Dataset: {"Internal + External" if external_results else "Internal Only"}',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


# 调用函数生成综合热图
try:
    if len(score_df) > 0:
        fig_heatmap = create_comprehensive_heatmap(score_df, results,
                                                   external_results if external_validation_available else None)
        plt.savefig(os.path.join(results_dir, 'comprehensive_metrics_heatmap.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 综合指标热图已生成: comprehensive_metrics_heatmap.pdf")

except Exception as e:
    print(f"⚠️ 生成综合热图时出错: {str(e)}")
    print("继续执行其他分析...")

# ===================== 生成评分对比分析 =====================

print("\n生成评分对比分析...")

# 创建一个综合对比表
# 先检查列名格式
sensitivity_col = 'Internal_sensitivity_Raw' if 'Internal_sensitivity_Raw' in score_df.columns else 'Internal_sensitivity'
auc_pr_col = 'Internal_auc_pr_Raw' if 'Internal_auc_pr_Raw' in score_df.columns else 'Internal_auc_pr'

# 如果还是找不到，尝试其他可能的列名
if sensitivity_col not in score_df.columns:
    for col in score_df.columns:
        if 'internal' in col.lower() and 'sensitivity' in col.lower():
            sensitivity_col = col
            break

if auc_pr_col not in score_df.columns:
    for col in score_df.columns:
        if 'internal' in col.lower() and 'auc_pr' in col.lower():
            auc_pr_col = col
            break

comparison_analysis = pd.DataFrame({
    'Model': score_df['Model'],
    'Rank': score_df['Rank'],
    'Total_Score': score_df['Total_Score'].round(2),
    'Score_Category': pd.cut(score_df['Total_Score'],
                             bins=[0, 30, 50, 70, 100],
                             labels=['Poor', 'Fair', 'Good', 'Excellent']),
    'Best_Metric': score_df[['Sensitivity_Score', 'AUC_PR_Score', 'F3_Score',
                             'MCC_Score']].idxmax(axis=1).str.replace('_Score', ''),
    'Worst_Metric': score_df[['Sensitivity_Score', 'AUC_PR_Score', 'F3_Score',
                              'MCC_Score']].idxmin(axis=1).str.replace('_Score', ''),
    'Sensitivity_Rank': score_df[sensitivity_col].rank(ascending=False).astype(
        int) if sensitivity_col in score_df.columns else 0,
    'AUC_PR_Rank': score_df[auc_pr_col].rank(ascending=False).astype(int) if auc_pr_col in score_df.columns else 0,
    'Feature_Count': [len(results[m]['feature_names']) for m in score_df['Model']]
})

comparison_analysis.to_csv(os.path.join(results_dir, 'model_scores_comparison_analysis.csv'), index=False)
print(f"✅ 评分对比分析已保存: model_scores_comparison_analysis.csv")

# 打印最终结果
print("\n" + "=" * 60)
print(f"最终选择的最佳模型: {best_model_name}")
print(f"总分: {score_df.iloc[0]['Total_Score']:.2f}")
print("=" * 60)

# ===================== 保存每个模型的预测概率 =====================
print("\n7. 保存预测概率文件")
print("-" * 60)

# 为每个模型保存内部测试集预测概率
for model_name, pred_data in predictions_dict.items():
    pred_df = pd.DataFrame({
        'PatientID': pred_data['PatientID'],
        'True_Label': pred_data['True_Label'],
        'Predicted_Probability': pred_data['Predicted_Probability'],
        'Predicted_Class': pred_data['Predicted_Class']
    })

    filename = f'predictions_internal_{model_name.replace("+", "_")}.csv'
    pred_df.to_csv(os.path.join(results_dir, filename), index=False)
    print(f"  {model_name} (内部): 保存至 {filename}")

# 为每个模型保存外部验证集预测概率
if external_validation_available:
    for model_name, pred_data in external_predictions_dict.items():
        pred_df = pd.DataFrame({
            'PatientID': pred_data['PatientID'],
            'True_Label': pred_data['True_Label'],
            'Predicted_Probability': pred_data['Predicted_Probability'],
            'Predicted_Class': pred_data['Predicted_Class']
        })

        filename = f'predictions_external_{model_name.replace("+", "_")}.csv'
        pred_df.to_csv(os.path.join(results_dir, filename), index=False)
        print(f"  {model_name} (外部): 保存至 {filename}")


# ===================== 改进的可视化函数 =====================

def create_sensitivity_specificity_plot_with_labels(results, external_results, ax, best_model_name=None):
    """创建带有模型标注的敏感性vs特异性图，突出最佳模型"""

    # 为每个模型创建不同的颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '8', 'P', '+', 'x']

    # 收集所有点的坐标以调整图的范围
    all_specs = []
    all_sens = []

    for idx, (model_name, color) in enumerate(zip(results.keys(), colors)):
        marker = markers[idx % len(markers)]

        # 内部测试集
        sens_int = results[model_name]['sensitivity']
        spec_int = results[model_name]['specificity']
        all_specs.extend([spec_int])
        all_sens.extend([sens_int])

        # 外部验证集
        sens_ext = external_results[model_name]['sensitivity']
        spec_ext = external_results[model_name]['specificity']
        all_specs.extend([spec_ext])
        all_sens.extend([sens_ext])

        # 根据是否是最佳模型调整样式
        if model_name == best_model_name:
            size = 250
            edge_width = 3
            color = 'red'
            zorder = 100
        else:
            size = 150
            edge_width = 1
            zorder = 50

        # 绘制点
        ax.scatter(spec_int, sens_int, marker=marker, color=color, s=size,
                   edgecolors='black', linewidth=edge_width, alpha=0.7, zorder=zorder)
        ax.scatter(spec_ext, sens_ext, marker=marker, color=color, s=size,
                   edgecolors='black', linewidth=edge_width + 1, alpha=0.9, zorder=zorder)

        # 连接内外部的点
        ax.plot([spec_int, spec_ext], [sens_int, sens_ext],
                color=color, linestyle='--', alpha=0.3, linewidth=1)

        # 添加模型名称标注
        if model_name == best_model_name:
            # 最佳模型使用特殊标注
            ax.annotate(f'★ {model_name} (BEST)', (spec_int, sens_int),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='red',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow',
                                  edgecolor='red', linewidth=2, alpha=0.9))
        else:
            offset_x = 10 if idx % 2 == 0 else -10
            offset_y = 10 if idx % 3 == 0 else -10
            ax.annotate(f'{model_name}', (spec_int, sens_int),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=8, alpha=0.8, color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # 添加理想点标注
    ax.scatter(1, 1, marker='*', s=500, color='gold', edgecolors='black',
               linewidth=2, zorder=150, label='Ideal Point')

    # 动态设置坐标轴范围
    margin = 0.05
    ax.set_xlim(min(all_specs) - margin, 1 + margin)
    ax.set_ylim(min(all_sens) - margin, 1 + margin)

    ax.set_xlabel('Specificity', fontsize=14)
    ax.set_ylabel('Sensitivity', fontsize=14)
    ax.set_title('Sensitivity vs Specificity Trade-off - All Models\n(Red = Best Model)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加参考线
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.5,
               label='90% Sensitivity Target')
    ax.axvline(x=0.9, color='blue', linestyle=':', alpha=0.5,
               label='90% Specificity Target')

    # 创建自定义图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, markeredgewidth=1, markeredgecolor='black',
                   label='Internal Test Set'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, markeredgewidth=2, markeredgecolor='black',
                   label='External Validation Set'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=12, markeredgewidth=3, markeredgecolor='black',
                   label='Best Model (Clinical+iTED)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12)


def plot_calibration_curve_improved(y_true, y_proba, model_name, ax, n_bins=5):
    """改进的校准曲线，适应小样本和不平衡数据"""

    try:
        # 按预测概率分位数划分
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(y_proba, quantiles)
        bins[0] = 0
        bins[-1] = 1

        # 确保bins是唯一的
        bins = np.unique(bins)

        if len(bins) < 3:
            bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # 计算每个bin的统计
        bin_centers = []
        fraction_positives = []
        counts = []

        for i in range(len(bins) - 1):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
            if i == len(bins) - 2:
                mask = (y_proba >= bins[i]) & (y_proba <= bins[i + 1])

            if mask.sum() > 0:
                bin_prob = y_proba[mask].mean()
                bin_true = y_true[mask].mean()
                bin_centers.append(bin_prob)
                fraction_positives.append(bin_true)
                counts.append(mask.sum())

        # 绘制校准曲线
        if len(bin_centers) > 1:
            ax.plot(bin_centers, fraction_positives,
                    marker='o', linewidth=2, markersize=8,
                    label=f'{model_name}\nn_bins={len(bin_centers)}')

            # 为每个点添加样本数标注
            for x, y, n in zip(bin_centers, fraction_positives, counts):
                ax.annotate(f'n={n}', (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, alpha=0.7)

    except Exception as e:
        print(f"校准曲线绘制失败 {model_name}: {e}")

    # 完美校准线
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)

    # 计算Brier Score
    brier = brier_score_loss(y_true, y_proba)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{model_name} - Brier: {brier:.3f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def plot_dca_improved_v2(results, y_true, title="Decision Curve Analysis", highlight_models=None):
    """改进的DCA曲线，更好的可视化"""

    plt.figure(figsize=(14, 10))

    # 使用更合理的阈值范围
    prevalence = np.mean(y_true)
    print(f"疾病流行率: {prevalence:.3f}")

    # 根据是否是外部验证集来设置不同的阈值范围
    if "External" in title:
        # 外部验证集用更大的范围
        thresholds = np.linspace(0, 1.0, 100)
        x_limit = 1.0
    else:
        # 内部测试集保持原有范围
        thresholds = np.linspace(0, 0.8, 80)
        x_limit = 0.8

    # 定义更好的颜色方案
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    # 计算Treat All的净收益
    net_benefit_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)

    # 为每个模型计算净收益
    all_net_benefits = {}
    clinical_useful_ranges = {}  # 存储每个模型的临床有用阈值范围

    for idx, (name, result) in enumerate(results.items()):
        net_benefits = []

        for threshold in thresholds:
            # 计算净收益
            y_pred = (result['y_proba'] >= threshold).astype(int)

            # True Positives and False Positives
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)

            # 净收益计算
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)

        all_net_benefits[name] = net_benefits

        # 找出净收益同时高于Treat All和Treat None的阈值范围
        useful_indices = []
        for i, (nb_model, nb_all) in enumerate(zip(net_benefits, net_benefit_all)):
            if nb_model > 0 and nb_model > nb_all:  # 高于Treat None (0) 且高于 Treat All
                useful_indices.append(i)

        if useful_indices:
            # 找到连续的范围
            min_idx = useful_indices[0]
            max_idx = useful_indices[-1]
            clinical_useful_ranges[name] = (thresholds[min_idx], thresholds[max_idx])

        # 设置线条样式
        if highlight_models and name in highlight_models:
            linewidth = 3
            alpha = 1.0
            linestyle = '-'
            zorder = 10
        else:
            linewidth = 1.5
            alpha = 0.6
            linestyle = '-'
            zorder = 5

        plt.plot(thresholds, net_benefits,
                 color=colors[idx],
                 lw=linewidth,
                 alpha=alpha,
                 linestyle=linestyle,
                 label=name,
                 zorder=zorder)

    # 添加参考线
    plt.plot(thresholds, net_benefit_all, 'r:', lw=2.5, label='Treat All', alpha=0.8, zorder=3)

    # Treat none
    plt.axhline(y=0, color='black', linestyle='--', lw=2.5, label='Treat None', alpha=0.8, zorder=3)

    # 在x轴（y=0）上标注最佳模型的临床有用阈值范围
    if highlight_models and len(highlight_models) > 0:
        best_model = highlight_models[0]
        if best_model in clinical_useful_ranges:
            min_t, max_t = clinical_useful_ranges[best_model]

            # 在x轴上用粗线段标记有用范围
            plt.plot([min_t, max_t], [0, 0], 'g-', lw=10, alpha=0.8, zorder=20,
                     solid_capstyle='round')

            # 在范围两端添加标记
            plt.scatter([min_t, max_t], [0, 0], color='darkgreen', s=200,
                        marker='|', linewidths=3, zorder=21)

            # 添加阈值数值标注
            plt.text(min_t, -0.008, f'{min_t:.3f}', ha='center', va='top',
                     fontsize=10, fontweight='bold', color='darkgreen')
            plt.text(max_t, -0.008, f'{max_t:.3f}', ha='center', va='top',
                     fontsize=10, fontweight='bold', color='darkgreen')

            # 在范围中间添加说明文字
            mid_t = (min_t + max_t) / 2
            plt.text(mid_t, -0.015, f'{best_model} Clinically Useful Range',
                     ha='center', va='top', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen",
                               edgecolor='darkgreen', linewidth=1.5, alpha=0.9))

    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')

    # 改进图例布局
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    plt.grid(True, alpha=0.3)
    plt.xlim([0, x_limit])  # 使用动态的x_limit
    plt.ylim([-0.05, 0.2])

    # 添加说明文字
    plt.text(0.02, 0.18, f'Disease Prevalence: {prevalence:.1%}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=11, fontweight='bold')

    # 不再添加子图

    plt.tight_layout()
    return plt.gcf()


def analyze_class_imbalance_impact(y_test, predictions_dict):
    """分析类别不平衡对模型性能的影响"""

    print("\n" + "=" * 60)
    print("类别不平衡影响分析")
    print("=" * 60)

    # 统计信息
    n_total = len(y_test)
    n_positive = np.sum(y_test == 1)
    n_negative = np.sum(y_test == 0)
    imbalance_ratio = n_negative / n_positive

    print(f"\n测试集统计:")
    print(f"总样本数: {n_total}")
    print(f"M0 (阴性): {n_negative} ({n_negative / n_total * 100:.1f}%)")
    print(f"M1 (阳性): {n_positive} ({n_positive / n_total * 100:.1f}%)")
    print(f"不平衡比例: {imbalance_ratio:.1f}:1")

    # 分析每个模型的预测分布
    print("\n各模型预测分布:")
    imbalance_analysis = []

    for model_name, pred_data in predictions_dict.items():
        y_pred = pred_data['Predicted_Class']
        n_pred_positive = np.sum(y_pred == 1)
        n_pred_negative = np.sum(y_pred == 0)

        print(f"\n{model_name}:")
        print(f"  预测M1: {n_pred_positive} ({n_pred_positive / n_total * 100:.1f}%)")
        print(f"  预测M0: {n_pred_negative} ({n_pred_negative / n_total * 100:.1f}%)")

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        imbalance_analysis.append({
            'Model': model_name,
            'Predicted_M1': n_pred_positive,
            'Predicted_M1_Pct': n_pred_positive / n_total * 100,
            'TP': tp,
            'FN': fn,
            'Detection_Rate': tp / n_positive * 100 if n_positive > 0 else 0
        })

    # 保存分析结果
    imbalance_df = pd.DataFrame(imbalance_analysis)
    imbalance_df.to_csv(os.path.join(results_dir, 'class_imbalance_analysis.csv'), index=False)

    print("\n建议:")
    print("1. 当前已使用SMOTE过采样平衡训练数据")
    print("2. 重点关注AUC-PR和F2-Score等适合不平衡数据的指标")
    print("3. 考虑调整决策阈值以提高敏感性")
    print("4. 在临床应用中，可能需要根据假阴性的成本调整模型")


def plot_calibration_curves_comparison(results, y_true, title="Calibration Curves Comparison"):
    """绘制所有模型的校准曲线在一个图中，便于比较"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    # 左图：所有模型的校准曲线
    for idx, (name, result) in enumerate(results.items()):
        y_proba = result['y_proba']

        # 计算校准曲线
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10, strategy='quantile'
            )

            ax1.plot(mean_predicted_value, fraction_of_positives,
                     marker='o', linewidth=2, markersize=6,
                     label=f'{name} (Brier: {result["brier_score"]:.3f})',
                     color=colors[idx], alpha=0.8)
        except Exception as e:
            print(f"校准曲线计算失败 {name}: {str(e)}")
            continue

    # 完美校准线
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Perfect calibration')

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curves - All Models', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # 右图：Brier Score比较
    model_names_list = list(results.keys())
    brier_scores = [results[m]['brier_score'] for m in model_names_list]

    # 按Brier Score排序
    sorted_indices = np.argsort(brier_scores)
    sorted_model_names = [model_names_list[i] for i in sorted_indices]
    sorted_scores = [brier_scores[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    bars = ax2.barh(range(len(sorted_model_names)), sorted_scores, color=sorted_colors)
    ax2.set_yticks(range(len(sorted_model_names)))
    ax2.set_yticklabels(sorted_model_names)
    ax2.set_xlabel('Brier Score (lower is better)', fontsize=12)
    ax2.set_title('Brier Score Comparison', fontsize=14)
    ax2.grid(True, axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', va='center', fontsize=9)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


# ===================== 生成精简的综合性能比较表 =====================
print("\n" + "=" * 60)
print("生成精简的综合性能比较表")
print("=" * 60)

# ========== 1. 生成内部验证集所有指标的性能比较表 ==========
print("\n生成内部验证集综合性能表...")

internal_comprehensive = []
for model_name in results.keys():
    metrics = {
        'Model': model_name,

        # 基础信息
        'N_Features': len(results[model_name]['feature_names']),
        'Optimal_Threshold': results[model_name]['optimal_threshold'],

        # 主要性能指标
        'AUC_ROC': results[model_name]['auc_roc'],
        'AUC_ROC_CI': f"{results[model_name]['auc_roc_ci'][0]:.4f}-{results[model_name]['auc_roc_ci'][1]:.4f}",
        'AUC_PR': results[model_name]['auc_pr'],
        'AUC_PR_CI': f"{results[model_name]['auc_pr_ci'][0]:.4f}-{results[model_name]['auc_pr_ci'][1]:.4f}",

        # 分类性能指标
        'Sensitivity': results[model_name]['sensitivity'],
        'Specificity': results[model_name]['specificity'],
        'Precision': results[model_name]['precision'],
        'NPV': results[model_name]['npv'],
        'PPV': results[model_name]['ppv'],

        # F系列指标
        'F1_Score': results[model_name]['f1_score'],
        'F2_Score': results[model_name]['f2_score'],
        'F3_Score': results[model_name]['f3_score'],

        # 平衡性指标
        'MCC': results[model_name]['mcc'],
        'Cohen_Kappa': results[model_name]['cohen_kappa'],
        'G_Mean': results[model_name]['g_mean'],
        'Balanced_Accuracy': results[model_name]['balanced_accuracy'],

        # 校准和风险指标
        'Brier_Score': results[model_name]['brier_score'],
        'Brier_CI': f"{results[model_name]['brier_ci'][0]:.4f}-{results[model_name]['brier_ci'][1]:.4f}",

        # 检测率和错误率
        'Detection_Rate': results[model_name]['detection_rate'],
        'False_Negative_Rate': results[model_name]['false_negative_rate'],

        # 似然比
        'LR_Positive': results[model_name]['lr_positive'] if results[model_name]['lr_positive'] != np.inf else 'Inf',
        'LR_Negative': results[model_name]['lr_negative'],

        # 诊断优势比
        'DOR': results[model_name]['dor'] if results[model_name]['dor'] != np.inf else 'Inf',

        # Youden指数和标记性
        'Informedness_Youden': results[model_name]['informedness'],
        'Markedness': results[model_name]['markedness'],

        # 9个评分体系指标（原始值，非0-100分）
        'Sensitivity_Weight30': results[model_name]['sensitivity'],  # 权重30%
        'NPV_Weight15': results[model_name]['npv'],  # 权重15%
        'F3_Score_Weight10': results[model_name]['f3_score'],  # 权重10%
        'AUC_PR_Weight12': results[model_name]['auc_pr'],  # 权重12%
        'Detection_Rate_Weight8': results[model_name]['detection_rate'],  # 权重8%
        'MCC_Weight5': results[model_name]['mcc'],  # 权重5%
        'Brier_Score_Weight5': results[model_name]['brier_score'],  # 权重5%
        'Specificity_Weight3': results[model_name]['specificity'],  # 权重3%
        'LR_Negative_Weight2': results[model_name]['lr_negative'],  # 权重2%
    }
    internal_comprehensive.append(metrics)

internal_comprehensive_df = pd.DataFrame(internal_comprehensive)

# 按AUC-ROC降序排序
internal_comprehensive_df = internal_comprehensive_df.sort_values('AUC_ROC', ascending=False)

# 保存内部验证集综合表
internal_comprehensive_df.to_csv(
    os.path.join(results_dir, 'comprehensive_performance_internal.csv'),
    index=False,
    encoding='utf-8-sig'
)
print(f"✅ 内部验证集综合性能表已保存: comprehensive_performance_internal.csv")
print(f"   包含 {len(internal_comprehensive_df)} 个模型，{len(internal_comprehensive_df.columns)} 个指标")

# ========== 2. 生成外部验证集所有指标的性能比较表 ==========
if external_validation_available:
    print("\n生成外部验证集综合性能表...")

    external_comprehensive = []
    for model_name in external_results.keys():
        metrics = {
            'Model': model_name,

            # 基础信息
            'N_Features': len(results[model_name]['feature_names']),
            'Optimal_Threshold': external_results[model_name]['optimal_threshold'],

            # 主要性能指标
            'AUC_ROC': external_results[model_name]['auc_roc'],
            'AUC_ROC_CI': f"{external_results[model_name]['auc_roc_ci'][0]:.4f}-{external_results[model_name]['auc_roc_ci'][1]:.4f}",
            'AUC_PR': external_results[model_name]['auc_pr'],
            'AUC_PR_CI': f"{external_results[model_name]['auc_pr_ci'][0]:.4f}-{external_results[model_name]['auc_pr_ci'][1]:.4f}",

            # 分类性能指标
            'Sensitivity': external_results[model_name]['sensitivity'],
            'Specificity': external_results[model_name]['specificity'],
            'Precision': external_results[model_name]['precision'],
            'NPV': external_results[model_name]['npv'],
            'PPV': external_results[model_name]['ppv'],

            # F系列指标
            'F1_Score': external_results[model_name]['f1_score'],
            'F2_Score': external_results[model_name]['f2_score'],
            'F3_Score': external_results[model_name]['f3_score'],

            # 平衡性指标
            'MCC': external_results[model_name]['mcc'],
            'Cohen_Kappa': external_results[model_name]['cohen_kappa'],
            'G_Mean': external_results[model_name]['g_mean'],
            'Balanced_Accuracy': external_results[model_name]['balanced_accuracy'],

            # 校准和风险指标
            'Brier_Score': external_results[model_name]['brier_score'],
            'Brier_CI': f"{external_results[model_name]['brier_ci'][0]:.4f}-{external_results[model_name]['brier_ci'][1]:.4f}",

            # 检测率和错误率
            'Detection_Rate': external_results[model_name]['detection_rate'],
            'False_Negative_Rate': external_results[model_name]['false_negative_rate'],

            # 似然比
            'LR_Positive': external_results[model_name]['lr_positive'] if external_results[model_name][
                                                                              'lr_positive'] != np.inf else 'Inf',
            'LR_Negative': external_results[model_name]['lr_negative'],

            # 诊断优势比
            'DOR': external_results[model_name]['dor'] if external_results[model_name]['dor'] != np.inf else 'Inf',

            # Youden指数和标记性
            'Informedness_Youden': external_results[model_name]['informedness'],
            'Markedness': external_results[model_name]['markedness'],

            # 9个评分体系指标（原始值，非0-100分）
            'Sensitivity_Weight30': external_results[model_name]['sensitivity'],  # 权重30%
            'NPV_Weight15': external_results[model_name]['npv'],  # 权重15%
            'F3_Score_Weight10': external_results[model_name]['f3_score'],  # 权重10%
            'AUC_PR_Weight12': external_results[model_name]['auc_pr'],  # 权重12%
            'Detection_Rate_Weight8': external_results[model_name]['detection_rate'],  # 权重8%
            'MCC_Weight5': external_results[model_name]['mcc'],  # 权重5%
            'Brier_Score_Weight5': external_results[model_name]['brier_score'],  # 权重5%
            'Specificity_Weight3': external_results[model_name]['specificity'],  # 权重3%
            'LR_Negative_Weight2': external_results[model_name]['lr_negative'],  # 权重2%
        }
        external_comprehensive.append(metrics)

    external_comprehensive_df = pd.DataFrame(external_comprehensive)

    # 按AUC-ROC降序排序
    external_comprehensive_df = external_comprehensive_df.sort_values('AUC_ROC', ascending=False)

    # 保存外部验证集综合表
    external_comprehensive_df.to_csv(
        os.path.join(results_dir, 'comprehensive_performance_external.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 外部验证集综合性能表已保存: comprehensive_performance_external.csv")
    print(f"   包含 {len(external_comprehensive_df)} 个模型，{len(external_comprehensive_df.columns)} 个指标")

# ========== 3. 生成基于10个评分体系的模型比较表 ==========
print("\n生成评分体系模型比较表...")

# 定义所有评分列（11个METRIC_WEIGHTS指标 + 1个稳定性）
scoring_columns = [
    'Model', 'Rank', 'Total_Score',
    'Sensitivity_Score',      # 30%
    'NPV_Score',              # 15%
    'F3_Score',               # 10%
    'AUC_PR_Score',           # 12%
    'Detection_Rate_Score',   # 8%
    'MCC_Score',              # 5%
    'G_Mean_Score',           # 5%
    'Brier_Score',            # 5%
    'Specificity_Score',      # 5%
    'PPV_Score',              # 3%
    'LR_Negative_Score',      # 2%
    'Stability_Score'         # 20%（独立权重）
]

# 检查哪些列实际存在
available_scoring_columns = [col for col in scoring_columns if col in score_df.columns]

# 创建评分比较表
scoring_comparison_df = score_df[available_scoring_columns].copy()

# 添加权重信息到列名
column_rename = {
    'Sensitivity_Score': 'Sensitivity_Score_30pct',
    'NPV_Score': 'NPV_Score_15pct',
    'F3_Score': 'F3_Score_10pct',
    'AUC_PR_Score': 'AUC_PR_Score_12pct',
    'Detection_Rate_Score': 'Detection_Rate_Score_8pct',
    'MCC_Score': 'MCC_Score_5pct',
    'G_Mean_Score': 'G_Mean_Score_5pct',
    'Brier_Score': 'Brier_Score_5pct',
    'Specificity_Score': 'Specificity_Score_5pct',
    'PPV_Score': 'PPV_Score_3pct',
    'LR_Negative_Score': 'LR_Negative_Score_2pct',
    'Stability_Score': 'Stability_Score_20pct_independent'  # 标明是独立的
}

# 重命名列以包含权重信息
for old_name, new_name in column_rename.items():
    if old_name in scoring_comparison_df.columns:
        scoring_comparison_df.rename(columns={old_name: new_name}, inplace=True)

scoring_comparison_df['Total_Weight'] = '100%'  # 所有模型的权重总和

# 添加最佳模型标记
scoring_comparison_df['Is_Best_Model'] = scoring_comparison_df['Model'] == best_model_name

# 按Rank排序
scoring_comparison_df = scoring_comparison_df.sort_values('Rank')

# 保存评分体系比较表
scoring_comparison_df.to_csv(
    os.path.join(results_dir, 'model_scoring_system_comparison.csv'),
    index=False,
    encoding='utf-8-sig'
)
print(f"✅ 评分体系模型比较表已保存: model_scoring_system_comparison.csv")
print(f"   包含 {len(scoring_comparison_df)} 个模型的评分")

# 显示最终生成的文件清单
print("\n" + "=" * 60)
print("精简后的性能比较文件清单：")
print("=" * 60)
print("1. comprehensive_performance_internal.csv - 内部验证集所有指标")
print("2. comprehensive_performance_external.csv - 外部验证集所有指标")
print("3. model_scoring_system_comparison.csv - 10个评分体系指标比较")
print("\n其他保留的重要文件：")
print("- model_scores_top10.csv - Top10模型")
print("- model_scores_comparison_analysis.csv - 评分对比分析")
print("- class_imbalance_analysis.csv - 类别不平衡分析")
print("- threshold_analysis.csv - 阈值分析")
print("- confusion_matrices_data_internal/external.csv - 混淆矩阵数据")

# ===================== 9. 生成改进的可视化结果 =====================
print("\n8. 生成改进的可视化结果")
print("-" * 60)

# 3. ROC曲线 - 内部测试集
plt.figure(figsize=(10, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

for (name, result), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y[test_idx], result['y_proba'])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {result["auc_roc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves Comparison - Internal Test Set', fontsize=16, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curves_comparison_internal.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# 外部验证集ROC曲线
if external_validation_available:
    plt.figure(figsize=(10, 8))

    for (name, result), color in zip(external_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_external, result['y_proba'])
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {result["auc_roc"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves Comparison - External Validation Set', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves_comparison_external.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# 4. PR曲线 - 内部测试集
plt.figure(figsize=(10, 8))

for (name, result), color in zip(results.items(), colors):
    precision, recall, _ = precision_recall_curve(y[test_idx], result['y_proba'])
    plt.plot(recall, precision, color=color, lw=2,
             label=f'{name} (AUC-PR = {result["auc_pr"]:.3f})')

# 添加基线（随机分类器）
baseline = np.sum(y[test_idx] == 1) / len(y[test_idx])
plt.axhline(y=baseline, color='k', linestyle='--', lw=1.5,
            label=f'Baseline = {baseline:.3f}')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curves - Internal Test Set', fontsize=16, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'pr_curves_comparison_internal.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# 外部验证集PR曲线
if external_validation_available:
    plt.figure(figsize=(10, 8))

    for (name, result), color in zip(external_results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_external, result['y_proba'])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{name} (AUC-PR = {result["auc_pr"]:.3f})')

    baseline_ext = np.sum(y_external == 1) / len(y_external)
    plt.axhline(y=baseline_ext, color='k', linestyle='--', lw=1.5,
                label=f'Baseline = {baseline_ext:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves - External Validation Set', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pr_curves_comparison_external.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# 5. 改进的校准曲线 - 内部测试集
fig, axes = plt.subplots(3, 5, figsize=(25, 18))  # 改为3x5=15个子图
axes = axes.flatten()

for idx, (name, result) in enumerate(results.items()):
    if idx < 15:
        plot_calibration_curve_improved(
            y[test_idx], result['y_proba'], name, axes[idx], n_bins=5
        )

for idx in range(len(results), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Improved Calibration Curves - Internal Test Set',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'calibration_curves_improved_internal.pdf'),
            dpi=300, bbox_inches='tight')
plt.close()

# 外部验证集校准曲线
if external_validation_available:
    fig, axes = plt.subplots(3, 5, figsize=(25, 18))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(external_results.items()):
        if idx < 15:
            plot_calibration_curve_improved(
                y_external, result['y_proba'], name, axes[idx], n_bins=5
            )

    for idx in range(len(external_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Improved Calibration Curves - External Validation Set',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'calibration_curves_improved_external.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    # ===================== 生成所有模型的混淆矩阵 =====================
    print("\n生成所有模型的混淆矩阵...")

    # 内部测试集 - 所有模型混淆矩阵
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx < 15:
            cm = confusion_matrix(y[test_idx], result['y_pred'])

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # 创建标注
            labels = np.array([[f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                                for j in range(2)] for i in range(2)])

            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[idx],
                        xticklabels=['M0', 'M1'], yticklabels=['M0', 'M1'],
                        cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{name}\nSens:{result["sensitivity"]:.3f}, Spec:{result["specificity"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices - All Models (Internal Test Set)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices_all_models_internal.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 外部验证集 - 所有模型混淆矩阵
    if external_validation_available:
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        axes = axes.flatten()

        for idx, (name, result) in enumerate(external_results.items()):
            if idx < 15:
                cm = confusion_matrix(y_external, result['y_pred'])

                # 计算百分比
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

                # 创建标注
                labels = np.array([[f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                                    for j in range(2)] for i in range(2)])

                sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', ax=axes[idx],
                            xticklabels=['M0', 'M1'], yticklabels=['M0', 'M1'],
                            cbar_kws={'label': 'Count'})
                axes[idx].set_title(f'{name}\nSens:{result["sensitivity"]:.3f}, Spec:{result["specificity"]:.3f}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')

        # 隐藏多余的子图
        for idx in range(len(external_results), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Confusion Matrices - All Models (External Validation Set)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrices_all_models_external.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 保存混淆矩阵数据到CSV
    # 内部测试集
    cm_data_internal = []
    for name, result in results.items():
        cm = confusion_matrix(y[test_idx], result['y_pred'])
        cm_data_internal.append({
            'Model': name,
            'TN': cm[0, 0],
            'FP': cm[0, 1],
            'FN': cm[1, 0],
            'TP': cm[1, 1]
        })

    pd.DataFrame(cm_data_internal).to_csv(
        os.path.join(results_dir, 'confusion_matrices_data_internal.csv'),
        index=False
    )

    # 外部验证集
    if external_validation_available:
        cm_data_external = []
        for name, result in external_results.items():
            cm = confusion_matrix(y_external, result['y_pred'])
            cm_data_external.append({
                'Model': name,
                'TN': cm[0, 0],
                'FP': cm[0, 1],
                'FN': cm[1, 0],
                'TP': cm[1, 1]
            })

        pd.DataFrame(cm_data_external).to_csv(
            os.path.join(results_dir, 'confusion_matrices_data_external.csv'),
            index=False
        )

    print("✅ 所有模型的混淆矩阵已生成")

# 6. 改进的DCA曲线 - 内部测试集
# 高亮显示最佳模型
highlight_models = [best_model_name]  # 可以添加其他想要突出的模型
fig = plot_dca_improved_v2(results, y[test_idx],
                           "Decision Curve Analysis - Internal Test Set",
                           highlight_models=highlight_models)
plt.savefig(os.path.join(results_dir, 'dca_curves_improved_v2_internal.pdf'),
            dpi=300, bbox_inches='tight')
plt.close()

# 外部验证集DCA曲线
if external_validation_available:
    fig = plot_dca_improved_v2(external_results, y_external,
                               "Decision Curve Analysis - External Validation Set",
                               highlight_models=highlight_models)
    plt.savefig(os.path.join(results_dir, 'dca_curves_improved_v2_external.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

# 添加校准曲线比较图
print("生成校准曲线比较图...")
fig = plot_calibration_curves_comparison(results, y[test_idx],
                                         "Calibration Analysis - Internal Test Set")
plt.savefig(os.path.join(results_dir, 'calibration_curves_comparison_internal.pdf'),
            dpi=300, bbox_inches='tight')
plt.close()

if external_validation_available:
    fig = plot_calibration_curves_comparison(external_results, y_external,
                                             "Calibration Analysis - External Validation Set")
    plt.savefig(os.path.join(results_dir, 'calibration_curves_comparison_external.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

# 7. 内部vs外部性能对比图
if external_validation_available:
    fig, axes = plt.subplots(4, 4, figsize=(32, 28))  # 保持大尺寸
    axes = axes.flatten()

    # 增加子图间距
    plt.subplots_adjust(hspace=0.5, wspace=0.35)  # 进一步增加间距

    model_names_list = list(results.keys())
    x_pos = np.arange(len(model_names_list))
    width = 0.35

    # 定义所有11个METRIC_WEIGHTS指标
    metrics_to_plot = [
        ('sensitivity', 'Sensitivity', 'darkred', 'lightcoral', '30%'),
        ('npv', 'NPV', 'darkgreen', 'lightgreen', '15%'),
        ('f3_score', 'F3-Score', 'darkviolet', 'plum', '10%'),
        ('auc_pr', 'AUC-PR', 'darkblue', 'lightblue', '12%'),
        ('detection_rate', 'Detection Rate', 'darkcyan', 'lightcyan', '8%'),
        ('mcc', 'MCC', 'darkorange', 'peachpuff', '5%'),
        ('g_mean', 'G-Mean', 'darkmagenta', 'orchid', '5%'),
        ('brier_score', 'Brier Score', 'brown', 'rosybrown', '5%'),
        ('specificity', 'Specificity', 'gray', 'lightgray', '5%'),
        ('ppv', 'PPV', 'navy', 'skyblue', '3%'),
        ('lr_negative', 'LR Negative', 'olive', 'yellowgreen', '2%')
    ]

    # 绘制11个指标对比图
    for idx, (metric, title, color_dark, color_light, weight) in enumerate(metrics_to_plot):
        ax = axes[idx]
        internal_values = [results[m][metric] for m in model_names_list]
        external_values = [external_results[m][metric] for m in model_names_list]

        bars1 = ax.bar(x_pos - width / 2, internal_values, width,
                       label='Internal', color=color_dark, alpha=0.8)
        bars2 = ax.bar(x_pos + width / 2, external_values, width,
                       label='External', color=color_light, alpha=0.8)

        # 添加数值标注
        for bar in bars1 + bars2:
            height = bar.get_height()
            if abs(height) > 0.01:  # 修改条件，适应负值
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=6)

        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison (Weight: {weight})', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # 重要修改：移除或修正y轴范围设置
        # 让matplotlib自动设置y轴范围，或根据数据动态设置
        if metric == 'brier_score':
            # Brier score通常在0-1之间，但让它自动调整
            pass  # 不设置固定范围
        elif metric == 'lr_negative':
            # LR negative可能有较大范围
            pass  # 不设置固定范围
        else:
            # 对其他指标，稍微扩展范围以显示完整柱子
            y_min = min(min(internal_values), min(external_values)) - 0.05
            y_max = max(max(internal_values), max(external_values)) + 0.05
            ax.set_ylim(max(0, y_min), min(1.1, y_max))

    # 12. 稳定性得分（独立的20%权重）
    ax12 = axes[11]
    if 'Stability_Score' in score_df.columns:
        stability_scores_plot = []
        for model in model_names_list:
            if model in score_df['Model'].values:
                score = score_df[score_df['Model'] == model]['Stability_Score'].values[0]
                stability_scores_plot.append(score / 100)  # 转换为0-1范围
            else:
                stability_scores_plot.append(0.5)  # 默认值

        bars = ax12.bar(x_pos, stability_scores_plot, color='purple', alpha=0.7, edgecolor='black')

        # 添加数值标注
        for bar, score in zip(bars, stability_scores_plot):
            ax12.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                      f'{score:.2f}', ha='center', va='bottom', fontsize=6)

        ax12.set_xlabel('Models', fontsize=10)
        ax12.set_ylabel('Stability Score')
        ax12.set_title('Stability Score (Weight: 20% - Independent)', fontsize=12, fontweight='bold')
        ax12.set_xticks(x_pos)
        ax12.set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=8)
        ax12.grid(True, alpha=0.3, axis='y')
        ax12.set_ylim(0, 1.05)

    # 13. 综合加权得分
    ax13 = axes[12]
    internal_scores = []
    external_scores = []

    for model in model_names_list:
        # 内部得分 - 使用11个指标的加权
        int_score = (
                results[model]['sensitivity'] * 0.30 +
                results[model]['npv'] * 0.15 +
                results[model]['f3_score'] * 0.10 +
                results[model]['auc_pr'] * 0.12 +
                results[model]['detection_rate'] * 0.08 +
                results[model]['mcc'] * 0.05 +
                results[model]['g_mean'] * 0.05 +
                (1 - results[model]['brier_score']) * 0.05 +  # 反转Brier Score
                results[model]['specificity'] * 0.05 +
                results[model]['ppv'] * 0.03 +
                (1 - results[model]['lr_negative']) * 0.02  # 反转LR Negative
        )
        internal_scores.append(int_score)

        # 外部得分
        ext_score = (
                external_results[model]['sensitivity'] * 0.30 +
                external_results[model]['npv'] * 0.15 +
                external_results[model]['f3_score'] * 0.10 +
                external_results[model]['auc_pr'] * 0.12 +
                external_results[model]['detection_rate'] * 0.08 +
                external_results[model]['mcc'] * 0.05 +
                external_results[model]['g_mean'] * 0.05 +
                (1 - external_results[model]['brier_score']) * 0.05 +
                external_results[model]['specificity'] * 0.05 +
                external_results[model]['ppv'] * 0.03 +
                (1 - external_results[model]['lr_negative']) * 0.02
        )
        external_scores.append(ext_score)

    bars19 = ax13.bar(x_pos - width / 2, internal_scores, width, label='Internal', color='purple', alpha=0.8)
    bars20 = ax13.bar(x_pos + width / 2, external_scores, width, label='External', color='mediumpurple', alpha=0.8)

    # 添加数值标注
    for bar in bars19 + bars20:
        height = bar.get_height()
        ax13.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                  f'{height:.2f}', ha='center', va='bottom', fontsize=6)

    ax13.set_xlabel('Models', fontsize=10)
    ax13.set_ylabel('Weighted Score')
    ax13.set_title(
        'Weighted Score (All 11 Metrics Combined)\n30% Sens + 15% NPV + 10% F3 + 12% AUC-PR + 8% DR + 5% MCC + 5% G-Mean + 5% Brier + 5% Spec + 3% PPV + 2% LR-',
        fontweight='bold', fontsize=11)
    ax13.set_xticks(x_pos)
    ax13.set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=8)
    ax13.legend(fontsize=8)
    ax13.grid(True, alpha=0.3, axis='y')
    ax13.set_ylim(0, 1.05)

    # 突出显示最佳模型
    best_model_idx = model_names_list.index(best_model_name)
    ax13.axvspan(best_model_idx - 0.5, best_model_idx + 0.5, alpha=0.3, color='gold')
    ax13.text(best_model_idx, max(max(internal_scores), max(external_scores)) * 1.05,
              '★ BEST', ha='center', fontweight='bold', fontsize=10)

    # 隐藏未使用的子图
    for i in range(13, 16):
        axes[i].set_visible(False)

    # 添加总标题
    fig.suptitle(
        'Comprehensive Performance Metrics Comparison: Internal vs External\n' +
        '11 METRIC_WEIGHTS Indicators + Stability Score (Independent 20%) + Weighted Total\n' +
        f'Best Model: {best_model_name}',
        fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, 'performance_metrics_comparison_internal_vs_external_v2.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

# 8. 类别不平衡影响分析
analyze_class_imbalance_impact(y[test_idx], predictions_dict)

# ===================== 10. 最佳模型的详细分析 =====================
print("\n9. 最佳模型的详细分析")
print("-" * 60)

# 获取最佳模型
best_model = models[best_model_name]
best_features = feature_combinations[best_model_name]['features']
best_feature_names = feature_combinations[best_model_name]['names']

# 准备SHAP分析数据
X_test_best = best_features[test_idx]

# 创建SHAP解释器
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_best)

# 1. SHAP摘要图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_best,
                  feature_names=best_feature_names,
                  show=False)
plt.title(f'SHAP Summary Plot - {best_model_name} Model', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'shap_summary_{best_model_name}.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# 2. SHAP重要性条形图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_best,
                  feature_names=best_feature_names,
                  plot_type="bar", show=False)
plt.title(f'SHAP Feature Importance - {best_model_name} Model', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'shap_importance_{best_model_name}.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# 3. 决策阈值分析
print("\n决策阈值分析:")

# 获取实际的最佳筛查阈值
actual_best_threshold = results[best_model_name]['optimal_threshold']
print(f"实际最佳筛查阈值: {actual_best_threshold:.3f}")

# 设计包含筛查阈值的分析范围
min_threshold = max(0.01, actual_best_threshold - 0.05)  # 确保不低于1%
max_threshold = 0.9
step_size = 0.01 if actual_best_threshold < 0.2 else 0.05  # 低阈值区域用更密集的步长

thresholds = np.arange(min_threshold, max_threshold, step_size)

# 确保实际的最佳阈值在分析范围内
if actual_best_threshold not in thresholds:
    thresholds = np.append(thresholds, actual_best_threshold)
    thresholds = np.sort(thresholds)

print(f"阈值分析范围: {thresholds.min():.3f} - {thresholds.max():.3f}")
print(f"步长: {step_size}")

threshold_metrics = []

for threshold in thresholds:
    y_pred_threshold = (results[best_model_name]['y_proba'] >= threshold).astype(int)

    sensitivity = recall_score(y[test_idx], y_pred_threshold)
    specificity = recall_score(y[test_idx], y_pred_threshold, pos_label=0)
    precision = precision_score(y[test_idx], y_pred_threshold, zero_division=0)
    f3 = fbeta_score(y[test_idx], y_pred_threshold, beta=3)

    # 计算NPV
    tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred_threshold).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    threshold_metrics.append({
        'Threshold': threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'NPV': npv,
        'F3_Score': f3,  # 改为F3
        'Youden_Index': sensitivity + specificity - 1,
        'Screening_Score': sensitivity * 0.5 + npv * 0.3 + specificity * 0.2,
        'Meets_Screening_Req': sensitivity >= 0.7 and npv >= 0.5
    })

threshold_df = pd.DataFrame(threshold_metrics)

# 使用与模型训练一致的筛查优化阈值
best_threshold = results[best_model_name]['optimal_threshold']

# 找到这个阈值在threshold_df中对应的行
best_threshold_idx = (threshold_df['Threshold'] - best_threshold).abs().idxmin()

print(f"\n最佳决策阈值（基于筛查要求优化）: {best_threshold:.2f}")
print(f"该阈值下的性能:")
print(f"  敏感性: {threshold_df.loc[best_threshold_idx, 'Sensitivity']:.3f}")
print(f"  特异性: {threshold_df.loc[best_threshold_idx, 'Specificity']:.3f}")
print(f"  F3-Score: {threshold_df.loc[best_threshold_idx, 'F3_Score']:.3f}")
print(f"  (注：此阈值基于筛查工具要求，非Youden指数)")

# 绘制阈值分析图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(threshold_df['Threshold'], threshold_df['Sensitivity'], 'b-', label='Sensitivity', lw=2)
ax.plot(threshold_df['Threshold'], threshold_df['Specificity'], 'r-', label='Specificity', lw=2)
ax.plot(threshold_df['Threshold'], threshold_df['F3_Score'], 'g-', label='F3-Score', lw=2)
ax.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')

ax.set_xlabel('Decision Threshold', fontsize=12)
ax.set_ylabel('Performance Metric', fontsize=12)
ax.set_title(f'Threshold Analysis - {best_model_name} Model\n(Best Threshold based on Screening Requirements)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'threshold_analysis.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# 保存阈值分析结果
threshold_df.to_csv(os.path.join(results_dir, 'threshold_analysis.csv'), index=False)

# 混淆矩阵 - 最佳模型
# 内部测试集
plt.figure(figsize=(8, 6))
cm_int = confusion_matrix(y[test_idx], results[best_model_name]['y_pred'])
sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues',
            xticklabels=['M0', 'M1'],
            yticklabels=['M0', 'M1'],
            annot_kws={'size': 16})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title(f'Confusion Matrix - {best_model_name} Model (Internal Test Set)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'confusion_matrix_{best_model_name}_internal.pdf'),
            dpi=300, bbox_inches='tight')
plt.close()

# 外部验证集
if external_validation_available:
    plt.figure(figsize=(8, 6))
    cm_ext = confusion_matrix(y_external, external_results[best_model_name]['y_pred'])
    sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['M0', 'M1'],
                yticklabels=['M0', 'M1'],
                annot_kws={'size': 16})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'Confusion Matrix - {best_model_name} Model (External Validation Set)',
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{best_model_name}_external.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()

# 在脚本的"10. 最佳模型的详细分析"部分后添加以下代码

# ===================== 保存最佳模型的特征信息 =====================
print("\n保存最佳模型特征信息...")

# 获取最佳模型的特征名称
best_model_features = results[best_model_name]['feature_names']

# 创建特征信息字典
feature_info = {
    'model_name': best_model_name,
    'total_features': len(best_model_features),
    'features': best_model_features,
    'feature_types': {}
}

# 识别每个特征的类型
for feature in best_model_features:
    if feature in clinical_names:
        feature_info['feature_types'][feature] = 'clinical'
    elif feature in radiomics_names:
        feature_info['feature_types'][feature] = 'radiomics'
    elif feature in iTED_names:
        feature_info['feature_types'][feature] = 'iTED'
    elif feature == '3D_ITHscore':
        feature_info['feature_types'][feature] = 'ITHscore'
    else:
        feature_info['feature_types'][feature] = 'unknown'

# 统计各类特征数量
feature_counts = {}
for feat_type in feature_info['feature_types'].values():
    feature_counts[feat_type] = feature_counts.get(feat_type, 0) + 1

feature_info['feature_counts'] = feature_counts

# 1. 保存为JSON格式（便于程序读取）
import json

with open(os.path.join(results_dir, f'best_model_features_{best_model_name}.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_info, f, indent=4, ensure_ascii=False)

# 2. 保存为CSV格式（便于查看）
features_df = pd.DataFrame({
    'Feature_Name': best_model_features,
    'Feature_Type': [feature_info['feature_types'][f] for f in best_model_features],
    'Feature_Index': range(len(best_model_features))
})
features_df.to_csv(os.path.join(results_dir, f'best_model_features_{best_model_name}.csv'), index=False)

# 3. 保存编码映射（用于新数据预测）
import json
with open(os.path.join(results_dir, 'encoding_mappings.json'), 'w', encoding='utf-8') as f:
    json.dump(encoding_mappings, f, ensure_ascii=False, indent=4)

# 4. 创建一个易读的特征报告
feature_report = f"""
    最佳模型特征列表
    ================
    模型名称: {best_model_name}
    特征总数: {len(best_model_features)}

    特征类型分布:
    """

for feat_type, count in feature_counts.items():
    feature_report += f"- {feat_type}: {count} 个\n"

feature_report += "\n详细特征列表:\n"
feature_report += "-" * 50 + "\n"

# 按类型分组显示特征
for feat_type in ['clinical', 'iTED', 'radiomics', 'ITHscore']:
    if feat_type in feature_counts and feature_counts[feat_type] > 0:
        feature_report += f"\n{feat_type.upper()} 特征:\n"
        for i, feature in enumerate(best_model_features):
            if feature_info['feature_types'][feature] == feat_type:
                feature_report += f"  {i + 1}. {feature}\n"

# 保存特征报告
with open(os.path.join(results_dir, f'best_model_features_report_{best_model_name}.txt'), 'w', encoding='utf-8') as f:
    f.write(feature_report)

print(f"✅ 特征信息已保存至:")
print(f"   - JSON文件: best_model_features_{best_model_name}.json")
print(f"   - CSV文件: best_model_features_{best_model_name}.csv")
print(f"   - 文本报告: best_model_features_report_{best_model_name}.txt")
print(f"   - 标签编码器: label_encoders.pkl")

# 显示特征摘要
print(f"\n特征摘要:")
print(f"总特征数: {len(best_model_features)}")
for feat_type, count in feature_counts.items():
    print(f"  {feat_type}: {count} 个")

# 如果是Clinical+iTED模型，分别列出特征
if best_model_name == "Clinical+iTED":
    print("\n临床特征:")
    clinical_in_model = [f for f in best_model_features if feature_info['feature_types'][f] == 'clinical']
    for i, feat in enumerate(clinical_in_model[:10]):  # 显示前10个
        print(f"  {i + 1}. {feat}")
    if len(clinical_in_model) > 10:
        print(f"  ... 共 {len(clinical_in_model)} 个")

    print("\niTED特征:")
    ited_in_model = [f for f in best_model_features if feature_info['feature_types'][f] == 'iTED']
    for i, feat in enumerate(ited_in_model[:10]):  # 显示前10个
        print(f"  {i + 1}. {feat}")
    if len(ited_in_model) > 10:
        print(f"  ... 共 {len(ited_in_model)} 个")


# ===================== 生成筛查工具专用报告 =====================
def generate_screening_tool_report(score_df, results, external_results=None, stability_details=None):
    """
    生成筛查工具专用的评估报告
    重点关注敏感性、NPV和稳定性
    """

    has_external = external_results is not None
    best_model = score_df.iloc[0]['Model']

    report = f"""
================================================================================
                    筛查工具模型评估报告（改进版）
================================================================================
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

一、评估体系说明
--------------------------------------------------------------------------------
本评估采用筛查工具优化的评分体系，重点关注：
1. 高敏感性（减少漏诊）
2. 高NPV（阴性结果可靠）
3. 内外部性能稳定性（非对称评分）

评分构成:
{'- 内部验证: 40%' if has_external else '- 内部验证: 100%'}
{'- 外部验证: 40%' if has_external else ''}
{'- 稳定性: 20%（轻微提升=优秀，大幅变化=需调查）' if has_external else ''}

二、最佳模型
--------------------------------------------------------------------------------
模型名称: {best_model}
综合得分: {score_df.iloc[0]['Total_Score']:.2f}/100
惩罚系数: {score_df.iloc[0]['Penalty_Factor']:.2f}

三、筛查性能指标
--------------------------------------------------------------------------------
"""

    # 获取最佳模型的指标
    if has_external:
        report += f"""
                    内部验证集      外部验证集      变化        稳定性评价
Sensitivity:        {score_df.iloc[0]['Internal_sensitivity_Raw']:.3f}          {score_df.iloc[0]['External_sensitivity_Raw']:.3f}      """

        # 计算变化和解释
        if stability_details and stability_details[0] and 'sensitivity' in stability_details[0]:
            change = stability_details[0]['sensitivity']['change'] * 100
            stability = stability_details[0]['sensitivity']['stability']
            interpretation = interpret_stability_score(stability, change, 'Sensitivity')
            report += f"{change:+.1f}%     {interpretation}"

        report += f"""
NPV:               {score_df.iloc[0]['Internal_npv_Raw']:.3f}          {score_df.iloc[0]['External_npv_Raw']:.3f}      """

        if stability_details and stability_details[0] and 'npv' in stability_details[0]:
            change = stability_details[0]['npv']['change'] * 100
            stability = stability_details[0]['npv']['stability']
            interpretation = interpret_stability_score(stability, change, 'NPV')
            report += f"{change:+.1f}%     {interpretation}"

        report += f"""
F3-Score:          {score_df.iloc[0]['Internal_f3_Raw']:.3f}          {score_df.iloc[0]['External_f3_Raw']:.3f}      """

        if stability_details and stability_details[0] and 'f3' in stability_details[0]:
            change = stability_details[0]['f3']['change'] * 100
            stability = stability_details[0]['f3']['stability']
            interpretation = interpret_stability_score(stability, change, 'F3-Score')
            report += f"{change:+.1f}%     {interpretation}"
    else:
        report += f"""
                    内部验证集
Sensitivity:        {score_df.iloc[0]['Internal_sensitivity_Raw']:.3f}
NPV:               {score_df.iloc[0]['Internal_npv_Raw']:.3f}
F3-Score:          {score_df.iloc[0]['Internal_f3_Raw']:.3f}"""

    report += f"""

四、筛查工具适用性评估
--------------------------------------------------------------------------------
"""

    # 评估是否满足筛查要求
    meets_requirements = True
    requirements_status = []

    # 检查内部验证集
    if score_df.iloc[0]['Internal_sensitivity_Raw'] >= 0.70:
        requirements_status.append("✓ 内部敏感性 ≥ 0.70")
    else:
        requirements_status.append("✗ 内部敏感性 < 0.70 (不满足)")
        meets_requirements = False

    if score_df.iloc[0]['Internal_npv_Raw'] >= 0.85:
        requirements_status.append("✓ 内部NPV ≥ 0.85")
    else:
        requirements_status.append("✗ 内部NPV < 0.85 (不满足)")
        meets_requirements = False

    # 检查外部验证集（如果有）
    if has_external:
        if score_df.iloc[0]['External_sensitivity_Raw'] >= 0.65:
            requirements_status.append("✓ 外部敏感性 ≥ 0.65")
        else:
            requirements_status.append("✗ 外部敏感性 < 0.65 (不满足)")
            meets_requirements = False

        if score_df.iloc[0]['External_npv_Raw'] >= 0.80:
            requirements_status.append("✓ 外部NPV ≥ 0.80")
        else:
            requirements_status.append("✗ 外部NPV < 0.80 (不满足)")
            meets_requirements = False

        # 稳定性检查
        sens_drop = score_df.iloc[0]['Internal_sensitivity_Raw'] - score_df.iloc[0]['External_sensitivity_Raw']
        if sens_drop <= 0.15:
            requirements_status.append(f"✓ 敏感性稳定 (变化{sens_drop:.3f})")
        else:
            requirements_status.append(f"✗ 敏感性不稳定 (下降{sens_drop:.3f} > 0.15)")
            meets_requirements = False

    for status in requirements_status:
        report += f"{status}\n"

    report += f"""
总体评估: {'适合作为筛查工具' if meets_requirements else '需要进一步优化'}

五、前5名模型对比
--------------------------------------------------------------------------------
排名  模型名称                    总分    内部Sens  内部NPV """

    if has_external:
        report += "  外部Sens  外部NPV  稳定性"

    report += "\n"

    for idx in range(min(5, len(score_df))):
        row = score_df.iloc[idx]
        report += f"{row['Rank']:2d}    {row['Model'][:25]:<25} {row['Total_Score']:6.2f}  "
        report += f"{row['Internal_sensitivity_Raw']:7.3f}  {row['Internal_npv_Raw']:7.3f}  "

        if has_external:
            report += f"{row['External_sensitivity_Raw']:7.3f}  {row['External_npv_Raw']:7.3f}  "
            if 'Stability_Score' in row:
                report += f"{row['Stability_Score']:6.1f}"
        report += "\n"

    report += """
六、临床应用建议
--------------------------------------------------------------------------------
1. 决策阈值: 建议使用针对高敏感性优化的阈值（约0.2-0.3）
2. 应用场景: 适合作为初步筛查工具，阴性结果可靠性高
3. 注意事项: 阳性结果需要进一步确认检查
4. 质量控制: 建议定期验证模型性能，特别是敏感性和NPV
5. 稳定性监测: 关注模型在不同数据集上的表现差异

七、模型特征
--------------------------------------------------------------------------------
特征数量: {len(results[best_model]['feature_names'])}
特征类型: 详见特征重要性分析文件

================================================================================
"""

    # 保存报告
    with open(os.path.join(results_dir, 'screening_tool_evaluation_report_improved.txt'),
              'w', encoding='utf-8') as f:
        f.write(report)

    print("\n✅ 筛查工具评估报告已保存: screening_tool_evaluation_report_improved.txt")

    return report


# 调用生成报告
screening_report = generate_screening_tool_report(
    score_df, results,
    external_results if external_validation_available else None,
    stability_details
)

# ===================== 11. 生成最终报告 =====================
print("\n10. 生成最终报告")
print("-" * 60)

# LightGBM保存模型的方式
best_model.booster_.save_model(os.path.join(results_dir, f'best_model_{best_model_name}_lgb.txt'))

# 生成报告
report = f"""
    多模态特征融合LightGBM模型分析报告（改进版）
    ==========================================
    生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

    1. 数据概况
    ------------------------------------------
    训练数据来源: 中六医院
    总样本数: {len(y)}
    训练集: {len(train_idx)} ({len(train_idx) / len(y) * 100:.1f}%)
    验证集: {len(val_idx)} ({len(val_idx) / len(y) * 100:.1f}%)
    测试集: {len(test_idx)} ({len(test_idx) / len(y) * 100:.1f}%)

    类别分布:
     M0 (无转移): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)
     M1 (有转移): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)
     不平衡比例: {imbalance_ratio:.1f}:1

    处理策略:
     - 使用SMOTE过采样: {'是' if use_smote else '否'}
     - 使用概率校准: {'是' if use_calibration else '否'}
     - 调整scale_pos_weight: {scale_pos_weight:.2f}
     - 重点评估指标: Sensitivity, AUC-PR, F3-Score, MCC.。。
    """

# 在这里添加SMOTE详细信息
if use_smote and best_model_name in smote_selection_results:
    smote_info = smote_selection_results[best_model_name]
    report += f"""

    SMOTE策略选择详情（{best_model_name}模型）:
     选定策略: {smote_info['best_strategy']}
     验证集召回率: {smote_info['best_recall']:.4f}
     原始训练样本: {smote_info['original_samples']} (M1: {smote_info['original_minority']})
     重采样后样本: {smote_info['final_samples']} (M1: {smote_info['final_minority']})
     少数类增加: {smote_info['final_minority'] - smote_info['original_minority']}样本
    """

if external_validation_available:
    report += f"""
    外部验证数据来源: 云大医院
    外部验证集样本数: {len(y_external)}
    外部验证集类别分布:
     M0 (无转移): {(y_external == 0).sum()} ({(y_external == 0).sum() / len(y_external) * 100:.1f}%)
     M1 (有转移): {(y_external == 1).sum()} ({(y_external == 1).sum() / len(y_external) * 100:.1f}%)
    """

# 从selected_features_info中获取筛选后的特征数
clinical_selected_count = sum(1 for name in selected_features_info.get('Clinical', {}).get('selected_names', []))
radiomics_selected_count = sum(1 for name in selected_features_info.get('Radiomics', {}).get('selected_names', []))
iTED_selected_count = sum(1 for name in selected_features_info.get('iTED', {}).get('selected_names', []))

# 定义实际使用的权重
actual_weights = {
    'sensitivity': 0.70,
    'auc_pr': 0.15,
    'f2_score': 0.10,
    'mcc': 0.05,
    'specificity': 0.00,
    'g_mean': 0.00
}

report += f"""
    2. 特征概况
    ------------------------------------------
    临床特征: {len(clinical_names)} → {clinical_selected_count} (筛选后)
    影像组学特征: {len(radiomics_names)} → {radiomics_selected_count} (筛选后)
    iTED特征: {len(iTED_names)} → {iTED_selected_count} (筛选后)
    3D_ITHscore: 1

    3. 最佳模型选择
    ------------------------------------------
    选择策略: 综合评分法（适应不平衡数据）
    权重分配:
    第一梯队（核心筛查指标 55%）:
     - Sensitivity: 30%
     - NPV: 15%  
     - F3-Score: 10%
     
    第二梯队（平衡指标 30%）:
     - AUC-PR: 12%
     - Detection Rate: 8%
     - MCC: 5%
     - G-Mean: 5%
    
    第三梯队（辅助指标 15%）:
     - Brier Score: 5%
     - Specificity: 5%
     - PPV: 3%
     - LR Negative: 2%
    
    独立评分（仅外部验证时）:
     - Stability: 20%

    最佳模型: {best_model_name}

    4. 最佳模型性能
    ------------------------------------------
    内部测试集:
    
    【第一梯队 - 核心筛查指标（55%权重）】
     Sensitivity (敏感性): {results[best_model_name]['sensitivity']:.4f} [权重30%]
     NPV (阴性预测值): {results[best_model_name]['npv']:.4f} [权重15%]
     F3-Score (更重视召回): {results[best_model_name]['f3_score']:.4f} [权重10%]
    
    【第二梯队 - 平衡指标（30%权重）】
     AUC-PR: {external_results[best_model_name]['auc_pr']:.4f} (95% CI: {external_results[best_model_name]['auc_pr_ci'][0]:.4f}-{external_results[best_model_name]['auc_pr_ci'][1]:.4f}) [权重12%]
     Detection Rate (检出率): {external_results[best_model_name]['detection_rate']:.4f} [权重8%]
     MCC (马修斯相关系数): {external_results[best_model_name]['mcc']:.4f} [权重5%]
     G-Mean (几何平均): {external_results[best_model_name]['g_mean']:.4f} [权重5%]
    
    【第三梯队 - 辅助指标（15%权重）】
     MCC (马修斯相关系数): {results[best_model_name]['mcc']:.4f} [权重5%]
     Brier Score (校准度): {results[best_model_name]['brier_score']:.4f} (95% CI: {results[best_model_name]['brier_ci'][0]:.4f}-{results[best_model_name]['brier_ci'][1]:.4f}) [权重5%]
     Specificity (特异性): {results[best_model_name]['specificity']:.4f} [权重3%]
     LR- (阴性似然比): {results[best_model_name]['lr_negative']:.4f} [权重2%]
     
    【稳定性分析（独立20%权重）】
     Stability Score: {score_df[score_df['Model'] == best_model_name]['Stability_Score'].values[0]:.1f}%
    
    【传统参考指标（不计入评分）】
     AUC-ROC: {results[best_model_name]['auc_roc']:.4f} (95% CI: {results[best_model_name]['auc_roc_ci'][0]:.4f}-{results[best_model_name]['auc_roc_ci'][1]:.4f})
     F1-Score: {results[best_model_name]['f1_score']:.4f}
     F2-Score: {results[best_model_name]['f2_score']:.4f}
     PPV (阳性预测值): {results[best_model_name]['ppv']:.4f}
     Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.4f}
     G-Mean: {results[best_model_name]['g_mean']:.4f}
    """

if external_validation_available:
    report += f"""
    外部验证集:
    
    【第一梯队 - 核心筛查指标（55%权重）】
     Sensitivity (敏感性): {external_results[best_model_name]['sensitivity']:.4f} [权重30%]
     NPV (阴性预测值): {external_results[best_model_name]['npv']:.4f} [权重15%]
     F3-Score (更重视召回): {external_results[best_model_name]['f3_score']:.4f} [权重10%]
    
    【第二梯队 - 平衡指标（30%权重）】
     AUC-PR: {external_results[best_model_name]['auc_pr']:.4f} (95% CI: {external_results[best_model_name]['auc_pr_ci'][0]:.4f}-{external_results[best_model_name]['auc_pr_ci'][1]:.4f}) [权重12%]
     Detection Rate (检出率): {external_results[best_model_name]['detection_rate']:.4f} [权重8%]
     MCC (马修斯相关系数): {external_results[best_model_name]['mcc']:.4f} [权重5%]
     G-Mean (几何平均): {external_results[best_model_name]['g_mean']:.4f} [权重5%]
    
    【第三梯队 - 辅助指标（15%权重）】
     Brier Score (校准度): {external_results[best_model_name]['brier_score']:.4f} (95% CI: {external_results[best_model_name]['brier_ci'][0]:.4f}-{external_results[best_model_name]['brier_ci'][1]:.4f}) [权重5%]
     Specificity (特异性): {external_results[best_model_name]['specificity']:.4f} [权重5%]
     PPV (阳性预测值): {external_results[best_model_name]['ppv']:.4f} [权重3%]
     LR- (阴性似然比): {external_results[best_model_name]['lr_negative']:.4f} [权重2%]
    
    【独立稳定性评分（20%权重）】
     Stability Score: {score_df[score_df['Model'] == best_model_name]['Stability_Score'].values[0] if 'Stability_Score' in score_df.columns else 'N/A':.1f}%
    
    【传统参考指标（不计入评分）】
     AUC-ROC: {external_results[best_model_name]['auc_roc']:.4f} (95% CI: {external_results[best_model_name]['auc_roc_ci'][0]:.4f}-{external_results[best_model_name]['auc_roc_ci'][1]:.4f})
     F1-Score: {external_results[best_model_name]['f1_score']:.4f}
     F2-Score: {external_results[best_model_name]['f2_score']:.4f}
     Balanced Accuracy: {external_results[best_model_name]['balanced_accuracy']:.4f}
    
    【稳定性分析】
     内外部敏感性差异: {abs(results[best_model_name]['sensitivity'] - external_results[best_model_name]['sensitivity']):.4f}
     内外部NPV差异: {abs(results[best_model_name]['npv'] - external_results[best_model_name]['npv']):.4f}
     内外部AUC-PR差异: {abs(results[best_model_name]['auc_pr'] - external_results[best_model_name]['auc_pr']):.4f}
     内外部MCC差异: {abs(results[best_model_name]['mcc'] - external_results[best_model_name]['mcc']):.4f}
    """

report += f"""
    决策阈值优化:
     推荐阈值: {best_threshold:.2f} (基于筛查工具要求：高敏感性优先)
     该阈值下敏感性: {threshold_df.loc[best_threshold_idx, 'Sensitivity']:.3f}
     该阈值下特异性: {threshold_df.loc[best_threshold_idx, 'Specificity']:.3f}
     阈值优化策略: 在满足最低敏感性要求前提下最大化综合得分

    特征数: {len(results[best_model_name]['feature_names'])}

    5. 生成的文件清单
    ------------------------------------------
    性能评估:
    - comprehensive_performance_internal.csv
    - comprehensive_performance_external.csv
    - model_scores_improved_comprehensive.csv
    - model_scoring_system_comparison.csv
    - model_scores_comparison_analysis.csv
    - class_imbalance_analysis.csv
    - threshold_analysis.csv
    - confusion_matrices_data_internal.csv
    - confusion_matrices_data_external.csv

    预测结果:
    - predictions_internal_[模型名].csv
    - predictions_external_[模型名].csv
    - all_patients_predictions_[最佳模型]_zlyy.csv
    - patients_risk_groups_for_protein_analysis_[最佳模型].csv

    可视化结果:
    - model_scores_visualization_enhanced.pdf
    - comprehensive_metrics_heatmap.pdf
    - stability_analysis_asymmetric_corrected.pdf
    - roc_curves_comparison_internal/external.pdf
    - pr_curves_comparison_internal/external.pdf
    - calibration_curves_improved_internal/external.pdf
    - calibration_curves_comparison_internal/external.pdf
    - dca_curves_improved_v2_internal/external.pdf
    - performance_metrics_comparison_internal_vs_external_v2.pdf
    - confusion_matrices_all_models_internal/external.pdf
    - risk_stratification_analysis.pdf
    
    最佳模型分析:
    - threshold_analysis.pdf
    - shap_summary_{best_model_name}.pdf
    - shap_importance_{best_model_name}.pdf
    - confusion_matrix_{best_model_name}_internal/external.pdf
    - best_model_features_{best_model_name}.json/csv/txt
    - best_model_features_report_{best_model_name}.txt
    
    模型文件:
    - best_model_{best_model_name}_lgb.txt
    - encoding_mappings.json
    
    报告文件:
    - multimodal_model_report_improved.txt
    - screening_tool_evaluation_report_improved.txt
    - risk_stratification_report.txt

"""

# 保存报告
with open(os.path.join(results_dir, 'multimodal_model_report_improved.txt'), 'w', encoding='utf-8') as f:
   f.write(report)

print(report)

print("\n" + "=" * 80)
print("多模态特征融合模型分析完成（改进版）！")
print(f"所有结果已保存至: {results_dir}")
print("=" * 80)

# ===================== 为所有训练集患者生成预测概率 =====================
print("\n" + "=" * 80)
print("为所有训练集患者生成预测概率和风险分组")
print("=" * 80)

# 获取最佳模型的特征
best_features_all = feature_combinations[best_model_name]['features']

# 使用最佳模型对所有患者进行预测
print(f"\n使用最佳模型 {best_model_name} 对所有患者进行预测...")

# 检查模型是否有校准器
if hasattr(best_model, 'calibrator'):
   print("使用校准后的概率...")
   all_predictions_uncalibrated = best_model.predict_proba(best_features_all)[:, 1]
   all_predictions = apply_calibration(all_predictions_uncalibrated, best_model.calibrator)
else:
   all_predictions = best_model.predict_proba(best_features_all)[:, 1]

# 创建包含所有信息的DataFrame
all_patients_predictions = pd.DataFrame({
   'PatientID': patient_ids,
   'True_M_stage': y,
   'Predicted_Probability': all_predictions,
   'Dataset_Split': 'Unknown'  # 先设置为未知
})

# 标记数据集划分
all_patients_predictions.loc[train_idx, 'Dataset_Split'] = 'Train'
all_patients_predictions.loc[val_idx, 'Dataset_Split'] = 'Validation'
all_patients_predictions.loc[test_idx, 'Dataset_Split'] = 'Test'

# 基于预测概率进行风险分组
# 方法1：基于分位数
tertiles = all_patients_predictions['Predicted_Probability'].quantile([0.33, 0.67])
all_patients_predictions['Risk_Group_Tertile'] = pd.cut(
   all_patients_predictions['Predicted_Probability'],
   bins=[0, tertiles[0.33], tertiles[0.67], 1],
   labels=['Low', 'Intermediate', 'High'],
   include_lowest=True
)

# 方法2：基于固定阈值（可以根据临床需求调整）
all_patients_predictions['Risk_Group_Fixed'] = pd.cut(
   all_patients_predictions['Predicted_Probability'],
   bins=[0, 0.2, 0.5, 1],
   labels=['Low', 'Intermediate', 'High'],
   include_lowest=True
)

# 方法3：基于最佳阈值 - 简化为高低风险二分组
all_patients_predictions['Risk_Binary'] = (all_patients_predictions['Predicted_Probability'] >= best_threshold).astype(int)

# 直接基于筛查阈值进行二分组
risk_groups_clinical = []
for prob in all_patients_predictions['Predicted_Probability']:
    if prob >= best_threshold:
        risk_groups_clinical.append('High')
    else:
        risk_groups_clinical.append('Low')

all_patients_predictions['Risk_Group_Clinical'] = risk_groups_clinical

# 添加额外的临床信息（如果需要）
# 从原始数据中提取一些关键临床特征
key_clinical_features = ['Age', 'Sex', 'T_stage', 'N_stage', 'BRAF_V600E']  # 根据实际情况调整
for feature in key_clinical_features:
   if feature in clinical_features.columns:
       all_patients_predictions[feature] = clinical_features[feature].values

# 统计各风险组的分布
print("\n风险分组统计:")
print("\n1. 基于三分位数的风险分组:")
print(all_patients_predictions['Risk_Group_Tertile'].value_counts().sort_index())
print("\n各组M1比例:")
for group in ['Low', 'Intermediate', 'High']:
   group_data = all_patients_predictions[all_patients_predictions['Risk_Group_Tertile'] == group]
   m1_rate = (group_data['True_M_stage'] == 1).mean() * 100
   print(f"  {group}: {m1_rate:.1f}% (n={len(group_data)})")

print("\n2. 基于固定阈值的风险分组 (0.2, 0.5):")
print(all_patients_predictions['Risk_Group_Fixed'].value_counts().sort_index())
print("\n各组M1比例:")
for group in ['Low', 'Intermediate', 'High']:
   group_data = all_patients_predictions[all_patients_predictions['Risk_Group_Fixed'] == group]
   m1_rate = (group_data['True_M_stage'] == 1).mean() * 100
   print(f"  {group}: {m1_rate:.1f}% (n={len(group_data)})")

print("\n3. 基于临床最佳阈值的风险分组（二分组）:")
print(all_patients_predictions['Risk_Group_Clinical'].value_counts().sort_index())
print("\n各组M1比例:")
for group in ['Low', 'High']:  # 改为两个组
   group_data = all_patients_predictions[all_patients_predictions['Risk_Group_Clinical'] == group]
   if len(group_data) > 0:  # 添加检查
       m1_rate = (group_data['True_M_stage'] == 1).mean() * 100
       print(f"  {group}: {m1_rate:.1f}% (n={len(group_data)})")

# 保存结果
output_filename = f'all_patients_predictions_{best_model_name}_zlyy.csv'
output_path = os.path.join(results_dir, output_filename)
all_patients_predictions.to_csv(output_path, index=False)
print(f"\n✅ 所有患者的预测结果已保存至: {output_filename}")

# 生成风险分组可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 预测概率分布直方图
ax1 = axes[0, 0]
counts, bins, _ = ax1.hist(all_patients_predictions['Predicted_Probability'],
                           bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# 在柱子上标记数量
for i, count in enumerate(counts):
    if count >= 5:  # 只显示数量>=5的柱子
        x_pos = (bins[i] + bins[i+1]) / 2
        ax1.text(x_pos, count + 5, f'{int(count)}',
                ha='center', va='bottom', fontsize=8)

ax1.axvline(x=best_threshold, color='red', linestyle='--',
           label=f'Best Threshold = {best_threshold:.2f}')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Predicted Probabilities (All Patients)')
ax1.legend()

# 2. 按真实标签分组的预测概率箱线图
ax2 = axes[0, 1]
data_to_plot = [all_patients_predictions[all_patients_predictions['True_M_stage'] == 0]['Predicted_Probability'],
               all_patients_predictions[all_patients_predictions['True_M_stage'] == 1]['Predicted_Probability']]

n_m0 = len(data_to_plot[0])
n_m1 = len(data_to_plot[1])

bp = ax2.boxplot(data_to_plot, labels=['M0', 'M1'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')

# 设置y轴范围
ax2.set_ylim(-0.1, 1.05)

# 标记数量
ax2.text(1, -0.12, f'n={n_m0}', ha='center', va='top', fontsize=10, fontweight='bold')
ax2.text(2, -0.12, f'n={n_m1}', ha='center', va='top', fontsize=10, fontweight='bold')

ax2.set_ylabel('Predicted Probability')
ax2.set_title('Predicted Probabilities by True M Stage')
ax2.axhline(y=best_threshold, color='red', linestyle='--', alpha=0.5)

# 3. 风险分组的M1率比较
ax3 = axes[1, 0]
risk_groups = ['Low', 'High']

m1_rates_clinical = []
group_sizes = []
m1_counts = []

for group in risk_groups:
    group_data = all_patients_predictions[all_patients_predictions['Risk_Group_Clinical'] == group]
    if len(group_data) > 0:
        m1_count = (group_data['True_M_stage'] == 1).sum()
        m1_rate = m1_count / len(group_data) * 100
        m1_rates_clinical.append(m1_rate)
        group_sizes.append(len(group_data))
        m1_counts.append(m1_count)
    else:
        m1_rates_clinical.append(0)
        group_sizes.append(0)
        m1_counts.append(0)

x = np.arange(len(risk_groups))
bars = ax3.bar(x, m1_rates_clinical, color='lightcoral')

# 标记数量
for i, (bar, m1_count, total) in enumerate(zip(bars, m1_counts, group_sizes)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
            f'{m1_count}/{total}\n({height:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('M1 Rate (%)')
ax3.set_xlabel('Risk Group')
ax3.set_title('M1 Rates by Risk Group (Clinical Threshold)')
ax3.set_xticks(x)
ax3.set_xticklabels(risk_groups)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(m1_rates_clinical) * 1.2 if m1_rates_clinical else 35)

# 4. 数据集划分中的风险分组分布
ax4 = axes[1, 1]
split_counts = all_patients_predictions.groupby(['Dataset_Split', 'Risk_Group_Clinical']).size().unstack(fill_value=0)

x = np.arange(len(split_counts.index))
width = 0.35

high_bars = ax4.bar(x - width/2, split_counts['High'], width, label='High', color='steelblue')
low_bars = ax4.bar(x + width/2, split_counts['Low'], width, label='Low', color='orange')

# 标记数量
for bars in [high_bars, low_bars]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, height + 2,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

ax4.set_xlabel('Dataset Split')
ax4.set_ylabel('Count')
ax4.set_title('Risk Group Distribution Across Dataset Splits')
ax4.set_xticks(x)
ax4.set_xticklabels(split_counts.index)
ax4.legend(title='Risk Group')
ax4.tick_params(axis='x', rotation=0)
ax4.grid(True, alpha=0.3, axis='y')

max_count = max(split_counts['High'].max(), split_counts['Low'].max())
ax4.set_ylim(0, max_count * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'risk_stratification_analysis.pdf'),
           dpi=300, bbox_inches='tight')
plt.close()

# 生成用于蛋白表达分析的简化文件
# 只包含必要的列
protein_analysis_df = all_patients_predictions[['PatientID', 'Predicted_Probability',
                                               'Risk_Group_Tertile', 'Risk_Group_Clinical',
                                               'True_M_stage']].copy()
protein_analysis_filename = f'patients_risk_groups_for_protein_analysis_{best_model_name}.csv'
protein_analysis_df.to_csv(os.path.join(results_dir, protein_analysis_filename), index=False)
print(f"\n✅ 用于蛋白表达分析的文件已保存至: {protein_analysis_filename}")

# 生成风险分组的详细报告
risk_report = f"""
风险分组分析报告
================
模型: {best_model_name}
总患者数: {len(all_patients_predictions)}

预测概率统计:
- 最小值: {all_patients_predictions['Predicted_Probability'].min():.4f}
- 25%分位数: {all_patients_predictions['Predicted_Probability'].quantile(0.25):.4f}
- 中位数: {all_patients_predictions['Predicted_Probability'].median():.4f}
- 75%分位数: {all_patients_predictions['Predicted_Probability'].quantile(0.75):.4f}
- 最大值: {all_patients_predictions['Predicted_Probability'].max():.4f}

风险分组方法比较:
"""

# 添加各种分组方法的详细统计
for method, column in [('三分位数', 'Risk_Group_Tertile'),
                       ('固定阈值', 'Risk_Group_Fixed'),
                       ('临床阈值', 'Risk_Group_Clinical')]:
    risk_report += f"\n{method}方法:\n"

    # 根据不同方法使用不同的组列表
    if column == 'Risk_Group_Clinical':
        groups = ['Low', 'High']  # 临床阈值只有两组
    else:
        groups = ['Low', 'Intermediate', 'High']  # 其他方法三组

    for group in groups:
        group_data = all_patients_predictions[all_patients_predictions[column] == group]
        if len(group_data) > 0:  # 添加检查
            n_patients = len(group_data)
            n_m1 = (group_data['True_M_stage'] == 1).sum()
            m1_rate = n_m1 / n_patients * 100
            prob_range = f"{group_data['Predicted_Probability'].min():.3f}-{group_data['Predicted_Probability'].max():.3f}"
            risk_report += f"  {group}: n={n_patients}, M1={n_m1} ({m1_rate:.1f}%), 概率范围: {prob_range}\n"

risk_report += f"""

"""

with open(os.path.join(results_dir, 'risk_stratification_report.txt'), 'w', encoding='utf-8') as f:
   f.write(risk_report)

print("\n" + "=" * 60)
print("风险分组分析完成！")
print("=" * 60)

# ===================== 输出单独模型的预测概率 =====================
print("\n" + "=" * 80)
print("生成单独模型（Radiomics、iTED）的预测概率")
print("=" * 80)

# 定义需要单独输出的模型
single_models = ['Radiomics', 'iTED']

for model_name in single_models:
    if model_name in models:
        print(f"\n处理 {model_name} 模型...")

        # 获取模型和特征
        model = models[model_name]
        features = feature_combinations[model_name]['features']

        # 对训练集所有患者进行预测
        if hasattr(model, 'calibrator'):
            predictions_uncalibrated = model.predict_proba(features)[:, 1]
            predictions_calibrated = apply_calibration(predictions_uncalibrated, model.calibrator)
        else:
            predictions_calibrated = model.predict_proba(features)[:, 1]

        # 创建DataFrame
        single_model_predictions = pd.DataFrame({
            'PatientID': patient_ids,
            'True_M_stage': y,
            'Predicted_Probability': predictions_calibrated,
            'Model': model_name
        })

        # 保存文件
        filename = f'{model_name}_predictions_all_patients_zlyy.csv'
        single_model_predictions.to_csv(os.path.join(results_dir, filename), index=False)
        print(f"✅ {model_name} 模型预测已保存至: {filename}")

        # 如果有外部验证集
        if external_validation_available:
            external_features = external_feature_combinations[model_name]['features']

            if hasattr(model, 'calibrator'):
                external_predictions_uncalibrated = model.predict_proba(external_features)[:, 1]
                external_predictions_calibrated = apply_calibration(external_predictions_uncalibrated, model.calibrator)
            else:
                external_predictions_calibrated = model.predict_proba(external_features)[:, 1]

            external_single_model_predictions = pd.DataFrame({
                'PatientID': patient_ids_external,
                'True_M_stage': y_external,
                'Predicted_Probability': external_predictions_calibrated,
                'Model': model_name
            })

            filename_external = f'{model_name}_predictions_all_patients_ydyy.csv'
            external_single_model_predictions.to_csv(os.path.join(results_dir, filename_external), index=False)
            print(f"✅ {model_name} 外部验证集预测已保存至: {filename_external}")

# ===================== 生成外部验证集的风险分组（如果有外部数据） =====================
if external_validation_available:
   print("\n" + "=" * 80)
   print("为外部验证集患者生成预测概率和风险分组")
   print("=" * 80)

   # 获取外部验证集的特征
   best_features_external = external_feature_combinations[best_model_name]['features']

   # 使用最佳模型对外部验证集进行预测
   print(f"\n使用最佳模型 {best_model_name} 对外部验证集进行预测...")

   # 检查模型是否有校准器
   if hasattr(best_model, 'calibrator'):
       print("使用校准后的概率...")
       external_predictions_uncalibrated = best_model.predict_proba(best_features_external)[:, 1]
       external_predictions = apply_calibration(external_predictions_uncalibrated, best_model.calibrator)
   else:
       external_predictions = best_model.predict_proba(best_features_external)[:, 1]

   # 创建外部验证集的预测DataFrame
   external_patients_predictions = pd.DataFrame({
       'PatientID': patient_ids_external,
       'True_M_stage': y_external,
       'Predicted_Probability': external_predictions,
       'Dataset': 'External_Validation'
   })

   # 应用相同的风险分组方法
   # 使用训练集的阈值
   # 应用相同的二分组方法
   risk_groups_clinical_external = []
   for prob in external_patients_predictions['Predicted_Probability']:
       if prob >= best_threshold:
           risk_groups_clinical_external.append('High')
       else:
           risk_groups_clinical_external.append('Low')

   external_patients_predictions['Risk_Group_Clinical'] = risk_groups_clinical_external

   # 显示外部验证集的风险分组统计
   print("\n外部验证集风险分组统计:")
   print(external_patients_predictions['Risk_Group_Clinical'].value_counts().sort_index())
   print("\n各组M1比例:")
   for group in ['Low', 'High']:  # 改为两组
       group_data = external_patients_predictions[external_patients_predictions['Risk_Group_Clinical'] == group]
       if len(group_data) > 0:
           m1_rate = (group_data['True_M_stage'] == 1).mean() * 100
           print(f"  {group}: {m1_rate:.1f}% (n={len(group_data)})")

print("\n" + "=" * 80)
print("所有分析完成！")
print(f"结果已保存至: {results_dir}")
print("=" * 80)

# ===================== 筛查工具性能分析 =====================
print("\n" + "=" * 80)
print("筛查工具性能分析（高敏感性模型）")
print("=" * 80)

# 找出敏感性最高的前3个模型
sensitivity_ranking = sorted(results.items(),
                             key=lambda x: x[1]['sensitivity'],
                             reverse=True)[:3]

print("\n敏感性最高的模型:")
for rank, (model_name, result) in enumerate(sensitivity_ranking, 1):
    print(f"\n{rank}. {model_name}")
    print(f"   敏感性: {result['sensitivity']:.3f}")
    print(f"   特异性: {result['specificity']:.3f}")
    print(f"   检出M1数: {int(result['sensitivity'] * 24)}/24")
    print(f"   假阳性数: {int((1 - result['specificity']) * 153)}/153")

    # 计算不同阈值下的性能
    y_proba = result['y_proba']
    print(f"\n   不同阈值下的性能:")
    for threshold in [0.1, 0.2, 0.3]:
        y_pred_new = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred_new).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        print(f"   阈值{threshold:.1f}: 敏感性={sens:.3f}, 特异性={spec:.3f}")


def validate_score_df(score_df):
    """验证score_df中的列是否完整"""
    required_columns = [
        'Model', 'Rank', 'Total_Score',
        'Sensitivity_Score',      # 1
        'NPV_Score',              # 2
        'F3_Score',               # 3
        'AUC_PR_Score',           # 4
        'Detection_Rate_Score',   # 5
        'MCC_Score',              # 6
        'G_Mean_Score',           # 7 - 添加
        'Brier_Score',            # 8
        'Specificity_Score',      # 9
        'PPV_Score',              # 10 - 添加
        'LR_Negative_Score',      # 11
        'Stability_Score'         # 12
    ]

    missing_columns = [col for col in required_columns if col not in score_df.columns]
    if missing_columns:
        print(f"警告：缺少以下列：{missing_columns}")
        print(f"实际列：{score_df.columns.tolist()}")
    else:
        print("✅ 所有必需的列都存在")

    return len(missing_columns) == 0


# 在calculate_comprehensive_model_scores函数返回前调用
if not validate_score_df(score_df):
    print("警告：score_df缺少某些列，可能影响后续分析")
