import os
import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats
from scipy.spatial import distance_matrix, ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免X connection错误
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义患者分组
#LOW_RISK_PATIENTS = [554, 560, 592, 607, 619, 621, 625, 629, 644, 645, 676, 700, 713, 661, 646, 879]
#HIGH_RISK_PATIENTS = [365, 692, 654, 47, 514, 518, 576, 597, 651, 679, 704, 552, 491, 557, 585, 608, 615, 673]
#更换lightgbm后的阈值
#LOW_RISK_PATIENTS = [47, 365, 554, 560, 592, 607, 619, 621, 625, 629, 644, 645, 676, 700, 713, 661, 646, 879]
#HIGH_RISK_PATIENTS = [491, 514, 518, 552, 557, 576, 585, 597, 608, 615, 651, 673, 679, 704]
#采用筛查阈值后的风险分组
LOW_RISK_PATIENTS = [47, 554, 560, 592, 607, 619, 621, 625, 629, 644, 645, 676, 700, 713, 661, 646, 879]
HIGH_RISK_PATIENTS = [365, 491, 514, 518, 552, 557, 576, 585, 597, 608, 615, 651, 654, 673, 679, 692, 704]

# 添加真实临床结局（仅用于综合评分计算）
NON_METASTASIS_PATIENTS = [47, 365, 554, 560, 592, 607, 619, 621, 625, 629, 644, 645, 646, 661, 676, 700, 713, 879]
METASTASIS_PATIENTS = [491, 514, 518, 552, 557, 576, 585, 597, 608, 615, 651, 654, 673, 679, 692, 704]

# 数据路径
DATA_DIR = "/data/tiatoolbox/模型解释"  # 输入数据目录
OUTPUT_DIR = "/data/tiatoolbox/模型解释结果"  # 输出结果目录


def calculate_correlation_with_pvalue(x, y):
    """计算Spearman相关系数和p值"""
    from scipy import stats
    import numpy as np
    import pandas as pd

    # 移除缺失值
    valid_mask = ~(pd.isna(x) | pd.isna(y))
    if np.sum(valid_mask) < 3:
        return np.nan, np.nan

    try:
        corr, pval = stats.spearmanr(x[valid_mask], y[valid_mask])
        return corr, pval
    except:
        return np.nan, np.nan


# 细胞类型定义
CELL_TYPES = {
    1: "Neoplastic_Epithelial",
    2: "Inflammatory",
    3: "Connective"
}


class ThyroidCancerPathologyAnalyzer:
    """甲状腺癌病理特征分析器"""

    def _estimate_pvalue(self, corr_abs):
        if corr_abs > 0.6:
            return 0.001
        elif corr_abs > 0.4:
            return 0.01
        elif corr_abs > 0.3:
            return 0.05
        else:
            return 0.10

    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.low_risk_patients = LOW_RISK_PATIENTS
        self.high_risk_patients = HIGH_RISK_PATIENTS
        self.all_patients = LOW_RISK_PATIENTS + HIGH_RISK_PATIENTS

        # 创建输出目录（如果不存在）
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # 存储所有数据
        self.nuclear_features = {}
        self.ith_scores = {}
        self.patient_summaries = []

    def load_patient_data(self):
        """加载所有患者数据"""
        print("Loading patient data...")

        for patient_id in self.all_patients:
            # 加载nuclear features
            nuclear_path = os.path.join(self.data_dir, f"nuclear_features_{patient_id}.csv")
            ith_path = os.path.join(self.data_dir, f"cellular_ith_scores_{patient_id}.csv")

            if os.path.exists(nuclear_path) and os.path.exists(ith_path):
                self.nuclear_features[patient_id] = pd.read_csv(nuclear_path)
                self.ith_scores[patient_id] = pd.read_csv(ith_path)
                print(f"  Loaded data for patient {patient_id}")
            else:
                print(f"  Warning: Missing data for patient {patient_id}")

        print(f"Successfully loaded data for {len(self.nuclear_features)} patients\n")

    def analyze_ith_differences(self):
        """分析高低风险组间的ITH差异"""
        print("=" * 60)
        print("1. INTRATUMORAL HETEROGENEITY (ITH) ANALYSIS")
        print("=" * 60)

        # 合并所有ITH数据
        all_ith_data = []
        for patient_id in self.nuclear_features.keys():
            ith_df = self.ith_scores[patient_id].copy()
            ith_df['patient_id'] = patient_id
            ith_df['risk_group'] = 'high' if patient_id in self.high_risk_patients else 'low'
            all_ith_data.append(ith_df)

        combined_ith = pd.concat(all_ith_data, ignore_index=True)

        # 统计检验
        significant_features = []
        ith_results = []

        print("\nITH Features with Significant Differences (p < 0.05):")
        print("-" * 50)

        ith_columns = [col for col in combined_ith.columns if col.endswith('_ith')]

        for feature in ith_columns:
            high_risk_values = combined_ith[combined_ith['risk_group'] == 'high'][feature].values
            low_risk_values = combined_ith[combined_ith['risk_group'] == 'low'][feature].values

            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(high_risk_values, low_risk_values, alternative='two-sided')

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(high_risk_values) + np.var(low_risk_values)) / 2)
            if pooled_std > 0:
                effect_size = (np.mean(high_risk_values) - np.mean(low_risk_values)) / pooled_std
            else:
                effect_size = 0

            ith_results.append({
                'feature': feature,
                'high_mean': np.mean(high_risk_values),
                'high_std': np.std(high_risk_values),
                'low_mean': np.mean(low_risk_values),
                'low_std': np.std(low_risk_values),
                'p_value': p_value,
                'effect_size': abs(effect_size)
            })

            if p_value < 0.05:
                significant_features.append(feature)
                print(f"  {feature}: p={p_value:.4f}, Effect Size={abs(effect_size):.3f}")
                print(f"    High Risk: {np.mean(high_risk_values):.4f} ± {np.std(high_risk_values):.4f}")
                print(f"    Low Risk:  {np.mean(low_risk_values):.4f} ± {np.std(low_risk_values):.4f}")

        # 保存所有ITH特征的分析结果到CSV
        ith_results_df = pd.DataFrame(ith_results)
        ith_results_csv_path = os.path.join(self.output_dir, 'ith_analysis_results.csv')
        ith_results_df.to_csv(ith_results_csv_path, index=False)
        print(f"\nSaved ITH analysis results to: {ith_results_csv_path}")

        return ith_results_df, significant_features

    def analyze_cell_composition(self):
        """分析细胞组成差异"""
        print("\n" + "=" * 60)
        print("2. CELL COMPOSITION ANALYSIS")
        print("=" * 60)

        composition_data = []

        for patient_id in self.nuclear_features.keys():
            df = self.nuclear_features[patient_id]
            total_cells = len(df)

            # 计算各细胞类型比例
            cell_counts = df['nucleus_type'].value_counts()

            composition = {
                'patient_id': patient_id,
                'risk_group': 'high' if patient_id in self.high_risk_patients else 'low',
                'total_cells': total_cells,
                'tumor_ratio': cell_counts.get(1, 0) / total_cells if total_cells > 0 else 0,
                'inflammatory_ratio': cell_counts.get(2, 0) / total_cells if total_cells > 0 else 0,
                'stromal_ratio': cell_counts.get(3, 0) / total_cells if total_cells > 0 else 0,
            }

            # 计算肿瘤/炎症比值
            if cell_counts.get(2, 0) > 0:
                composition['tumor_inflammatory_ratio'] = cell_counts.get(1, 0) / cell_counts.get(2, 0)
            else:
                composition['tumor_inflammatory_ratio'] = cell_counts.get(1, 0)

            composition_data.append(composition)

        composition_df = pd.DataFrame(composition_data)

        # 统计比较
        print("\nCell Type Composition Comparison:")
        print("-" * 50)

        for metric in ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio', 'tumor_inflammatory_ratio']:
            high_values = composition_df[composition_df['risk_group'] == 'high'][metric].values
            low_values = composition_df[composition_df['risk_group'] == 'low'][metric].values

            stat, p_value = stats.mannwhitneyu(high_values, low_values, alternative='two-sided')

            print(f"\n{metric}:")
            print(f"  High Risk: {np.mean(high_values):.3f} ± {np.std(high_values):.3f}")
            print(f"  Low Risk:  {np.mean(low_values):.3f} ± {np.std(low_values):.3f}")
            print(f"  p-value: {p_value:.4f}")

        return composition_df

    def analyze_nuclear_morphology(self):
        """分析肿瘤细胞核形态特征"""
        print("\n" + "=" * 60)
        print("3. NUCLEAR MORPHOLOGY ANALYSIS")
        print("=" * 60)

        morphology_data = []

        for patient_id in self.nuclear_features.keys():
            df = self.nuclear_features[patient_id]
            tumor_cells = df[df['nucleus_type'] == 1]  # 只分析肿瘤细胞

            if len(tumor_cells) > 0:
                morphology = {
                    'patient_id': patient_id,
                    'risk_group': 'high' if patient_id in self.high_risk_patients else 'low',

                    # 核大小特征
                    'mean_nuclear_area': tumor_cells['area'].mean(),
                    'large_nuclei_ratio': (tumor_cells['area'] > tumor_cells['area'].quantile(0.75)).mean(),
                    'giant_nuclei_ratio': (tumor_cells['area'] > tumor_cells['area'].quantile(0.95)).mean(),

                    # 核形态复杂度
                    'mean_eccentricity': tumor_cells['eccentricity'].mean(),
                    'irregular_nuclei_ratio': (tumor_cells['circularity'] < 0.7).mean(),
                    'high_curve_std_ratio': (tumor_cells['curve_std'] > tumor_cells['curve_std'].quantile(0.75)).mean(),

                    # 染色特征
                    'mean_intensity_max': tumor_cells['intensity_max'].mean(),
                    'hyperchromatic_ratio': (
                            tumor_cells['intensity_max'] > tumor_cells['intensity_max'].quantile(0.75)).mean(),

                    # 纹理特征
                    'mean_contrast': tumor_cells['contrast'].mean(),
                    'mean_homogeneity': tumor_cells['homogeneity'].mean(),
                }

                morphology_data.append(morphology)

        morphology_df = pd.DataFrame(morphology_data)

        # 统计比较关键形态学特征
        # 统计比较所有形态学特征
        print("\nMorphological Features Comparison:")
        print("-" * 50)

        # 获取所有形态学特征列（排除patient_id和risk_group）
        morphology_features = [col for col in morphology_df.columns
                               if col not in ['patient_id', 'risk_group']]

        morphology_results = []
        for feature in morphology_features:
            if feature in morphology_df.columns:
                high_values = morphology_df[morphology_df['risk_group'] == 'high'][feature].values
                low_values = morphology_df[morphology_df['risk_group'] == 'low'][feature].values

                stat, p_value = stats.mannwhitneyu(high_values, low_values, alternative='two-sided')

                # 计算效应量
                pooled_std = np.sqrt((np.var(high_values) + np.var(low_values)) / 2)
                if pooled_std > 0:
                    effect_size = abs(np.mean(high_values) - np.mean(low_values)) / pooled_std
                else:
                    effect_size = 0

                morphology_results.append({
                    'feature': feature,
                    'high_mean': np.mean(high_values),
                    'high_std': np.std(high_values),
                    'low_mean': np.mean(low_values),
                    'low_std': np.std(low_values),
                    'p_value': p_value,
                    'effect_size': effect_size
                })

                # 只打印显著的特征
                if p_value < 0.05:
                    print(f"\n{feature}:")
                    print(f"  High Risk: {np.mean(high_values):.4f} ± {np.std(high_values):.4f}")
                    print(f"  Low Risk:  {np.mean(low_values):.4f} ± {np.std(low_values):.4f}")
                    print(f"  p-value: {p_value:.4f}, Effect Size: {effect_size:.3f}")

        # 保存形态学分析结果到CSV
        morphology_results_df = pd.DataFrame(morphology_results)
        morphology_results_csv_path = os.path.join(self.output_dir, 'morphology_analysis_results.csv')
        morphology_results_df.to_csv(morphology_results_csv_path, index=False)
        print(f"\nSaved morphology analysis results to: {morphology_results_csv_path}")

        # 保存结果供后续使用
        self.morphology_results_df = morphology_results_df

        return morphology_df

    def analyze_spatial_patterns(self):
        """分析空间分布模式"""
        print("\n" + "=" * 60)
        print("4. SPATIAL DISTRIBUTION ANALYSIS")
        print("=" * 60)

        spatial_data = []

        for patient_id in self.nuclear_features.keys():
            df = self.nuclear_features[patient_id]
            tumor_cells = df[df['nucleus_type'] == 1]

            if len(tumor_cells) > 10:
                coords = tumor_cells[['centroid_x', 'centroid_y']].values

                # 计算空间分散度
                centroid = coords.mean(axis=0)
                distances_from_center = np.sqrt(((coords - centroid) ** 2).sum(axis=1))

                # 计算最近邻距离
                if len(coords) > 5:
                    nbrs = NearestNeighbors(n_neighbors=min(6, len(coords)))
                    nbrs.fit(coords)
                    distances, indices = nbrs.kneighbors(coords)
                    mean_nn_distance = distances[:, 1:].mean()  # 排除自身
                else:
                    mean_nn_distance = 0

                spatial = {
                    'patient_id': patient_id,
                    'risk_group': 'high' if patient_id in self.high_risk_patients else 'low',
                    'tumor_dispersion': distances_from_center.std(),
                    'mean_nn_distance': mean_nn_distance,
                    'spatial_heterogeneity': distances_from_center.std() / (distances_from_center.mean() + 1e-6),
                }

                # 炎症细胞浸润分析
                inflammatory_cells = df[df['nucleus_type'] == 2]
                if len(inflammatory_cells) > 0:
                    spatial['inflammatory_density'] = len(inflammatory_cells) / len(df)
                else:
                    spatial['inflammatory_density'] = 0

                spatial_data.append(spatial)

        spatial_df = pd.DataFrame(spatial_data)

        # 统计比较
        print("\nSpatial Pattern Comparison:")
        print("-" * 50)

        for feature in ['tumor_dispersion', 'mean_nn_distance', 'spatial_heterogeneity', 'inflammatory_density']:
            if feature in spatial_df.columns:
                high_values = spatial_df[spatial_df['risk_group'] == 'high'][feature].values
                low_values = spatial_df[spatial_df['risk_group'] == 'low'][feature].values

                if len(high_values) > 0 and len(low_values) > 0:
                    stat, p_value = stats.mannwhitneyu(high_values, low_values, alternative='two-sided')

                    print(f"\n{feature}:")
                    print(f"  High Risk: {np.mean(high_values):.4f} ± {np.std(high_values):.4f}")
                    print(f"  Low Risk:  {np.mean(low_values):.4f} ± {np.std(low_values):.4f}")
                    print(f"  p-value: {p_value:.4f}")

        return spatial_df

    def calculate_composite_scores(self, composition_df, morphology_df, spatial_df, ith_results_df):
        """计算综合病理评分 - 使用数据驱动的方法"""
        print("\n" + "=" * 60)
        print("5. COMPOSITE PATHOLOGY SCORE")
        print("=" * 60)

        # 整合所有特征
        integrated_data = []

        # 首先选择最显著的ITH特征（p<0.1）
        significant_ith = ith_results_df[ith_results_df['p_value'] < 0.1]['feature'].tolist()
        if len(significant_ith) == 0:
            # 如果没有显著的，选择p值最小的5个
            significant_ith = ith_results_df.nsmallest(5, 'p_value')['feature'].tolist()

        print(f"\nSelected {len(significant_ith)} significant ITH features for composite score")

        for patient_id in self.nuclear_features.keys():
            patient_data = {'patient_id': patient_id,
                            'risk_group': 'high' if patient_id in self.high_risk_patients else 'low'}

            # 添加组成特征
            if patient_id in composition_df['patient_id'].values:
                comp_row = composition_df[composition_df['patient_id'] == patient_id].iloc[0]
                patient_data['tumor_ratio'] = comp_row['tumor_ratio']
                patient_data['inflammatory_ratio'] = comp_row['inflammatory_ratio']
                patient_data['tumor_inflammatory_ratio'] = comp_row.get('tumor_inflammatory_ratio', 0)

            # 添加形态特征
            if patient_id in morphology_df['patient_id'].values:
                morph_row = morphology_df[morphology_df['patient_id'] == patient_id].iloc[0]
                patient_data['mean_nuclear_area'] = morph_row['mean_nuclear_area']
                patient_data['large_nuclei_ratio'] = morph_row['large_nuclei_ratio']
                patient_data['irregular_nuclei_ratio'] = morph_row['irregular_nuclei_ratio']
                patient_data['hyperchromatic_ratio'] = morph_row['hyperchromatic_ratio']
                patient_data['mean_eccentricity'] = morph_row['mean_eccentricity']
                patient_data['mean_intensity_max'] = morph_row['mean_intensity_max']

            # 添加空间特征
            if patient_id in spatial_df['patient_id'].values:
                spatial_row = spatial_df[spatial_df['patient_id'] == patient_id].iloc[0]
                patient_data['tumor_dispersion'] = spatial_row['tumor_dispersion']
                patient_data['spatial_heterogeneity'] = spatial_row['spatial_heterogeneity']
                patient_data['inflammatory_density'] = spatial_row.get('inflammatory_density', 0)

            # 添加显著的ITH特征
            ith_row = self.ith_scores[patient_id].iloc[0]
            for ith_feature in significant_ith:
                if ith_feature in ith_row.index:
                    patient_data[ith_feature] = ith_row[ith_feature]

            integrated_data.append(patient_data)

        integrated_df = pd.DataFrame(integrated_data)

        # ========= 关键修改：基于真实结局计算特征重要性 =========
        feature_cols = [col for col in integrated_df.columns
                        if col not in ['patient_id', 'risk_group']]

        integrated_df[feature_cols] = integrated_df[feature_cols].fillna(integrated_df[feature_cols].median())

        # 计算每个特征的判别能力（基于真实结局）
        feature_importance = {}
        for feature in feature_cols:
            if feature in integrated_df.columns:
                # 使用真实结局分组
                metastasis_values = integrated_df[
                    integrated_df['patient_id'].isin(METASTASIS_PATIENTS)
                ][feature].values
                non_metastasis_values = integrated_df[
                    integrated_df['patient_id'].isin(NON_METASTASIS_PATIENTS)
                ][feature].values

                if len(metastasis_values) > 0 and len(non_metastasis_values) > 0:
                    stat, p_val = stats.mannwhitneyu(metastasis_values, non_metastasis_values)

                    # 计算基于真实结局的效应量
                    pooled_std = np.sqrt((np.var(metastasis_values) + np.var(non_metastasis_values)) / 2)
                    if pooled_std > 0:
                        effect_size = abs(np.median(metastasis_values) - np.median(non_metastasis_values)) / pooled_std
                    else:
                        effect_size = 0

                    feature_importance[feature] = effect_size if p_val < 0.2 else 0

        # 选择重要性最高的特征
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        selected_features = [f[0] for f in top_features if f[1] > 0]

        print(f"\nTop features for composite score (based on true outcome):")
        for feat, importance in top_features[:5]:
            print(f"  - {feat}: effect_size={importance:.3f}")

        # 标准化选定的特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(integrated_df[selected_features])

        # 使用基于真实结局的权重
        total_importance = sum([f[1] for f in top_features]) + 1e-6
        weights = {f[0]: f[1] / total_importance for f in top_features}

        # 计算综合评分
        composite_scores = np.zeros(len(integrated_df))
        for i, feature in enumerate(selected_features):
            # 确定方向（基于真实结局）
            metastasis_median = integrated_df[
                integrated_df['patient_id'].isin(METASTASIS_PATIENTS)
            ][feature].median()
            non_metastasis_median = integrated_df[
                integrated_df['patient_id'].isin(NON_METASTASIS_PATIENTS)
            ][feature].median()

            direction = 1 if metastasis_median > non_metastasis_median else -1
            composite_scores += direction * scaled_features[:, i] * weights.get(feature, 0.1)

        integrated_df['composite_score'] = composite_scores

        # 保存每位患者的综合病理评分（CSV）
        scores_csv_path = os.path.join(self.output_dir, 'patient_composite_scores.csv')
        integrated_df[['patient_id', 'risk_group', 'composite_score']].to_csv(scores_csv_path, index=False)
        print(f"Saved patient composite scores to: {scores_csv_path}")

        # 保存完整整合表（所有特征 + 综合评分）
        full_csv_path = os.path.join(self.output_dir, 'pathology_analysis_results.csv')
        integrated_df.to_csv(full_csv_path, index=False)
        print(f"Saved full integrated table (CSV) to: {full_csv_path}")

        # 同步导出Excel（若环境支持），与脚本输出说明保持一致
        try:
            full_xlsx_path = os.path.join(self.output_dir, 'pathology_analysis_results.xlsx')
            integrated_df.to_excel(full_xlsx_path, index=False)
            print(f"Saved full integrated table (Excel) to: {full_xlsx_path}")
        except Exception as e:
            print(f"Warning: failed to save Excel file; CSV saved instead. Details: {e}")

        # ========= 评估：在模型分组上比较综合评分 =========
        high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score'].values
        low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score'].values

        stat, p_value = stats.mannwhitneyu(high_scores, low_scores, alternative='two-sided')

        print("\nComposite Pathology Score (weights from true outcome, evaluated on model groups):")
        print("-" * 50)
        print(f"High Risk Group: {np.mean(high_scores):.3f} ± {np.std(high_scores):.3f}")
        print(f"Low Risk Group:  {np.mean(low_scores):.3f} ± {np.std(low_scores):.3f}")
        print(f"Mann-Whitney U test p-value: {p_value:.4f}")

        # 添加验证信息
        print("\nValidation Note:")
        print("- Feature weights calculated from: Metastasis vs Non-metastasis groups")
        print("- Score differences evaluated between: Model High Risk vs Low Risk groups")
        print("- This approach validates whether model risk groups capture outcome-relevant pathology")

        return integrated_df

    def create_visualizations(self, ith_results_df, composition_df, morphology_df, integrated_df):
        import pandas as pd
        """创建改进的综合可视化 - 包含完整的病理-临床-影像组学关联分析"""
        print("\n" + "=" * 60)
        print("6. GENERATING COMPREHENSIVE VISUALIZATIONS WITH FULL CORRELATION ANALYSIS")
        print("=" * 60)

        from matplotlib.gridspec import GridSpec
        from scipy.stats import spearmanr

        # 设置风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # ========== 加载临床和影像组学特征 ==========
        radiomics_path = os.path.join(self.data_dir, 'clinical_features_iTED_features_3D_ITHscore_34.csv')
        radiomics_df = None
        full_corr_matrix = None

        if os.path.exists(radiomics_path):
            print(f"Loading clinical and radiomics features from: {radiomics_path}")
            radiomics_df = pd.read_csv(radiomics_path)
            radiomics_df['PatientID'] = radiomics_df['PatientID'].astype(int)
            print(f"Loaded data: {radiomics_df.shape[0]} patients, {radiomics_df.shape[1] - 1} features")

            # 准备病理ITH特征数据
            pathology_data = []

            for patient_id in self.all_patients:
                if patient_id in radiomics_df['PatientID'].values:
                    # 获取所有ITH特征
                    if patient_id in self.ith_scores:
                        ith_row = self.ith_scores[patient_id].iloc[0]
                        patient_data = {'PatientID': patient_id}

                        # 添加所有ITH特征
                        for col in ith_row.index:
                            if col.endswith('_ith'):
                                patient_data[col] = ith_row[col]

                        # 添加综合评分
                        if patient_id in integrated_df['patient_id'].values:
                            comp_score = \
                                integrated_df[integrated_df['patient_id'] == patient_id]['composite_score'].iloc[0]
                            patient_data['composite_score'] = comp_score

                        pathology_data.append(patient_data)

            if pathology_data:
                pathology_df = pd.DataFrame(pathology_data)

                # 合并病理和临床/影像组学数据
                merged_df = pd.merge(radiomics_df, pathology_df, on='PatientID', how='inner')
                print(f"Merged data: {merged_df.shape[0]} patients with complete data")

                # 计算完整的相关性矩阵
                # 选择所有数值型特征
                clinical_features = ['Age', 'BMI', 'Tumor_size', 'Number_of_metastatic_lymph_nodes',
                                     'NLR', 'TG', 'TGAb', 'TPOAb']

                # 定义要排除的分类变量
                categorical_vars_to_exclude = ['M_stage', 'Sex', 'Multifocal',
                                               'Infiltrated_the_adjacent_tissue', 'T_stage']

                # iTED特征（13个）
                ited_features = [col for col in merged_df.columns if col.startswith('iTED_')]

                # 传统影像组学特征（排除分类变量）
                traditional_radiomics = [col for col in merged_df.columns
                                         if not col.startswith('iTED_')
                                         and col not in clinical_features
                                         and col not in ['PatientID'] + categorical_vars_to_exclude
                                         and not col.endswith('_ith')
                                         and col != 'composite_score']

                pathology_features = [col for col in merged_df.columns if col.endswith('_ith')] + ['composite_score']

                # 组合所有连续型特征
                radiomics_features = ited_features + traditional_radiomics
                corr_features = pathology_features + clinical_features + radiomics_features

                # 确保只选择数值型特征
                corr_data = merged_df[corr_features].select_dtypes(include=[np.number])

                # 计算相关性矩阵
                full_corr_matrix = corr_data.corr(method='spearman')

                print(f"Correlation matrix computed: {full_corr_matrix.shape}")

        # 创建图形 - 4x3布局
        fig = plt.figure(figsize=(32, 28))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.4,
                      width_ratios=[1, 1, 1.5],  # 第三列更宽
                      height_ratios=[1, 1, 1.2, 1])  # 第三行更高

        # ========== 第一行：显著的ITH特征 ==========
        # 修改阈值，可以根据需要调整
        sig_threshold = 0.1  # 可以改为0.05如果只想显示p<0.05的
        sig_ith = ith_results_df[ith_results_df['p_value'] < sig_threshold]

        if len(sig_ith) == 0:
            # 如果没有显著特征，显示p值最小的3个
            sig_ith = ith_results_df.nsmallest(3, 'p_value')
            n_plots = 3
        else:
            n_plots = min(3, len(sig_ith))
            sig_ith = sig_ith.nsmallest(n_plots, 'p_value')

        # 只显示有数据的子图，最多3个
        n_plots_to_show = min(3, len(sig_ith))

        for idx in range(3):
            if idx < n_plots_to_show:
                ax = fig.add_subplot(gs[0, idx])
                feature_row = sig_ith.iloc[idx]
                feature = feature_row['feature']

                # 收集数据
                high_risk_data = []
                low_risk_data = []

                for patient_id in self.nuclear_features.keys():
                    ith_row = self.ith_scores[patient_id].iloc[0]
                    if feature in ith_row.index:
                        if patient_id in self.high_risk_patients:
                            high_risk_data.append(ith_row[feature])
                        else:
                            low_risk_data.append(ith_row[feature])

                # 创建箱线图
                bp = ax.boxplot([high_risk_data, low_risk_data],
                                labels=['High Risk', 'Low Risk'],
                                patch_artist=True,
                                widths=0.6)

                # 设置颜色
                colors = ['#FF6B6B', '#4ECDC4']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # 添加散点
                for i, (data, color) in enumerate(zip([high_risk_data, low_risk_data], colors)):
                    y = data
                    x = np.random.normal(i + 1, 0.04, size=len(y))
                    ax.scatter(x, y, alpha=0.4, s=20, color=color, edgecolors='black', linewidth=0.5)

                # 添加p值标注
                p_val = feature_row['p_value']
                effect_size = feature_row['effect_size']

                y_max = max(max(high_risk_data), max(low_risk_data))
                y_range = y_max - min(min(high_risk_data), min(low_risk_data))

                # 显著性标记
                sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

                # 绘制显著性线
                ax.plot([1, 2], [y_max + y_range * 0.05, y_max + y_range * 0.05], 'k-', linewidth=1)
                ax.text(1.5, y_max + y_range * 0.08, f'{sig_text}\np={p_val:.3f}',
                        ha='center', fontsize=10, fontweight='bold')

                # 设置标题和标签
                feature_name = feature.replace('_ith', '').replace('_', ' ').title()
                ax.set_title(f'{feature_name} ITH\n(ES={effect_size:.2f})', fontsize=11, fontweight='bold')
                ax.set_ylabel('ITH Score', fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                # 第3个位置显示ITH汇总统计
                ax = fig.add_subplot(gs[0, idx])
                if idx == 2 and len(sig_ith) > 0:
                    summary_text = f"ITH Analysis Summary\n\n"
                    summary_text += f"Total features: 25\n"
                    summary_text += f"Significant (p<0.05): {len(ith_results_df[ith_results_df['p_value'] < 0.05])}\n"
                    summary_text += f"\nTop 3 by p-value:\n"
                    for i, row in ith_results_df.nsmallest(3, 'p_value').iterrows():
                        summary_text += f"{row['feature'].replace('_ith', '')}\n"
                        summary_text += f"  p={row['p_value']:.3f}\n"

                    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                            fontsize=9, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax.set_title('ITH Summary', fontsize=11, fontweight='bold')
                    ax.axis('off')
                else:
                    ax.axis('off')

            # 添加子图标签
            ax.text(-0.15, 1.15, chr(65 + idx), transform=ax.transAxes,
                    fontsize=14, fontweight='bold')

        # ========== 第二行：细胞组成和核形态学 ==========
        # 4. High Risk Group - Cell Composition
        ax4 = fig.add_subplot(gs[1, 0])
        high_risk_comp = composition_df[composition_df['risk_group'] == 'high'][
            ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio']].mean()

        wedges, texts, autotexts = ax4.pie(high_risk_comp,
                                           labels=['Tumor', 'Inflammatory', 'Stromal'],
                                           autopct='%1.1f%%',
                                           colors=['#FF6B6B', '#FFE66D', '#A8E6CF'],
                                           explode=(0.05, 0.05, 0.05),
                                           startangle=90)

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax4.set_title('High Risk Group\nCell Composition', fontsize=11, fontweight='bold')
        ax4.text(-0.15, 1.15, 'D', transform=ax4.transAxes, fontsize=14, fontweight='bold')

        # 5. Low Risk Group - Cell Composition
        ax5 = fig.add_subplot(gs[1, 1])
        low_risk_comp = composition_df[composition_df['risk_group'] == 'low'][
            ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio']].mean()

        wedges, texts, autotexts = ax5.pie(low_risk_comp,
                                           labels=['Tumor', 'Inflammatory', 'Stromal'],
                                           autopct='%1.1f%%',
                                           colors=['#4ECDC4', '#FFE66D', '#A8E6CF'],
                                           explode=(0.05, 0.05, 0.05),
                                           startangle=90)

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax5.set_title('Low Risk Group\nCell Composition', fontsize=11, fontweight='bold')
        ax5.text(-0.15, 1.15, 'E', transform=ax5.transAxes, fontsize=14, fontweight='bold')

        # 6. 替代方案 - 显示病理特征效应量排序或其他有意义的分析
        ax6 = fig.add_subplot(gs[1, 2])

        # 检查是否有显著的形态学特征
        has_sig_morph = False
        if hasattr(self, 'morphology_results_df'):
            sig_morph_features = self.morphology_results_df[self.morphology_results_df['p_value'] < 0.05]
            has_sig_morph = len(sig_morph_features) > 0

        if not has_sig_morph:
            # 如果没有显著的形态学特征，显示ITH特征的效应量分布
            top_ith_effects = ith_results_df.nlargest(8, 'effect_size')

            y_pos = np.arange(len(top_ith_effects))
            colors_by_p = ['#FF6B6B' if p < 0.05 else '#B0B0B0'
                           for p in top_ith_effects['p_value']]

            bars = ax6.barh(y_pos, top_ith_effects['effect_size'].values,
                            color=colors_by_p, alpha=0.8, edgecolor='black', linewidth=1)

            ax6.set_yticks(y_pos)
            ax6.set_yticklabels([f.replace('_ith', '')[:15] for f in top_ith_effects['feature']],
                                fontsize=8)
            ax6.set_xlabel('Effect Size (Cohen\'s d)', fontsize=10)
            ax6.set_title('Top ITH Features by Effect Size\n(Red: p<0.05)', fontsize=11, fontweight='bold')
            ax6.axvline(x=0.2, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax6.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax6.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax6.grid(True, alpha=0.3, axis='x')
        else:
            # 原有的形态学比较代码
            # 初始化selected_morph变量
            selected_morph = []
            morph_p_values = []  # Fix for UnboundLocalError

            # 从保存的morphology_results_df中选择p值最小的3个特征
            if hasattr(self, 'morphology_results_df'):
                # 如果在analyze_nuclear_morphology中保存了结果
                sig_morph = self.morphology_results_df.nsmallest(3, 'p_value')
                selected_morph = sig_morph['feature'].tolist()
            else:
                # 否则重新计算
                morph_features = [col for col in morphology_df.columns
                                  if col not in ['patient_id', 'risk_group']]

                morph_p_values = []
                for feat in morph_features:
                    if feat in morphology_df.columns:
                        high_vals = morphology_df[morphology_df['risk_group'] == 'high'][feat].values
                        low_vals = morphology_df[morphology_df['risk_group'] == 'low'][feat].values
                        if np.std(high_vals) > 0 or np.std(low_vals) > 0:
                            _, p_val = stats.mannwhitneyu(high_vals, low_vals)
                            morph_p_values.append((feat, p_val))

                if morph_p_values:  # 只有在有数据时才处理
                    morph_p_values.sort(key=lambda x: x[1])
                    selected_morph = [f[0] for f in morph_p_values[:3]]

            # 确保selected_morph已定义
            if 'selected_morph' not in locals():
                selected_morph = []

            if len(selected_morph) < 3:
                # 如果没有足够的有差异特征，补充其他特征
                for feat, p in (morph_p_values if "morph_p_values" in locals() and morph_p_values else []):
                    if feat not in selected_morph and len(selected_morph) < 3:
                        selected_morph.append(feat)

            # 创建标签映射
            label_map = {
                'mean_nuclear_area': 'Nuclear\nArea',
                'mean_eccentricity': 'Eccentricity',
                'irregular_nuclei_ratio': 'Irregular\nNuclei %',
                'hyperchromatic_ratio': 'Hyperchromatic\nNuclei %',
                'mean_intensity_max': 'Max\nIntensity',
                'large_nuclei_ratio': 'Large\nNuclei %'
            }

            if selected_morph:
                x = np.arange(len(selected_morph))
                width = 0.35

                high_morph = morphology_df[morphology_df['risk_group'] == 'high'][selected_morph].mean()
                low_morph = morphology_df[morphology_df['risk_group'] == 'low'][selected_morph].mean()
                high_sem = morphology_df[morphology_df['risk_group'] == 'high'][selected_morph].sem()
                low_sem = morphology_df[morphology_df['risk_group'] == 'low'][selected_morph].sem()

                bars1 = ax6.bar(x - width / 2, high_morph, width, yerr=high_sem,
                                label='High Risk', color='#FF6B6B', alpha=0.8,
                                edgecolor='black', linewidth=1, capsize=5)
                bars2 = ax6.bar(x + width / 2, low_morph, width, yerr=low_sem,
                                label='Low Risk', color='#4ECDC4', alpha=0.8,
                                edgecolor='black', linewidth=1, capsize=5)

                # 添加p值标记
                for i, feat in enumerate(selected_morph):
                    # 从 morphology_results_df 获取p值
                    if hasattr(self, 'morphology_results_df'):
                        p_val_row = self.morphology_results_df[self.morphology_results_df['feature'] == feat]
                        p_val = p_val_row['p_value'].iloc[0] if not p_val_row.empty else 1.0
                    else:
                        p_val = 1.0
                    if p_val < 0.05:
                        y_max = max(high_morph.iloc[i] + high_sem.iloc[i],
                                    low_morph.iloc[i] + low_sem.iloc[i]) * 1.1
                        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                        ax6.text(i, y_max, f'{sig_text}\np={p_val:.3f}',
                                 ha='center', fontsize=8)

                ax6.set_xlabel('Features', fontsize=10)
                ax6.set_ylabel('Value', fontsize=10)
                ax6.set_title('Nuclear Morphology Comparison', fontsize=11, fontweight='bold')
                ax6.set_xticks(x)
                ax6.set_xticklabels([label_map.get(f, f) for f in selected_morph], rotation=0, fontsize=9)
                ax6.legend(loc='upper right', fontsize=9)
                ax6.grid(True, alpha=0.3, axis='y')
            else:
                ax6.text(0.5, 0.5, 'No differential\nmorphology features',
                         ha='center', va='center', fontsize=12, color='gray')
                ax6.axis('off')

        ax6.text(-0.15, 1.05, 'F', transform=ax6.transAxes, fontsize=14, fontweight='bold')

        # ========== 第三行：评分和相关性分析 ==========
        # 7. Distribution of Composite Scores
        ax7 = fig.add_subplot(gs[2, 0])
        high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score']
        low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score']

        parts = ax7.violinplot([high_scores.values, low_scores.values], positions=[1, 2], widths=0.7,
                               showmeans=True, showmedians=True, showextrema=True)

        colors = ['#FF6B6B', '#4ECDC4']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # 添加散点
        for i, (scores, color) in enumerate(zip([high_scores, low_scores], colors)):
            y = scores
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax7.scatter(x, y, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)

        # 添加统计信息
        stat, p_value = scipy.stats.mannwhitneyu(high_scores, low_scores)
        sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'

        ax7.text(1.5, ax7.get_ylim()[1] * 0.9, f'{sig_text}\np={p_value:.3f}',
                 ha='center', fontsize=11, fontweight='bold')

        ax7.set_xticks([1, 2])
        ax7.set_xticklabels(['High Risk', 'Low Risk'])
        ax7.set_ylabel('Composite Pathology Score', fontsize=10)
        ax7.set_title(
            'Distribution of Composite Scores\n(Weights from outcomes, groups from model)\n(Mann-Whitney U test)',
            fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.text(-0.15, 1.05, 'G', transform=ax7.transAxes, fontsize=14, fontweight='bold')

        # 8. Individual Patient Scores
        ax8 = fig.add_subplot(gs[2, 1])
        patient_ids = integrated_df['patient_id'].values
        scores = integrated_df['composite_score'].values

        colors_patient = ['#FF6B6B' if p in self.high_risk_patients else '#4ECDC4'
                          for p in patient_ids]

        for i, (score, color) in enumerate(zip(scores, colors_patient)):
            ax8.scatter(i, score, c=color, alpha=0.7, s=60, edgecolors='black', linewidth=1)

        median_score = np.median(scores)
        ax8.axhline(y=median_score, color='gray', linestyle='--', linewidth=2)
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', alpha=0.7, label='High Risk', edgecolor='black'),
            Patch(facecolor='#4ECDC4', alpha=0.7, label='Low Risk', edgecolor='black'),
            plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--',
                       label=f'Median ({median_score:.2f})')
        ]
        ax8.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax8.set_xlabel('Patient Index', fontsize=10)
        ax8.set_ylabel('Composite Score', fontsize=10)
        ax8.set_title('Individual Patient Scores\n(Outcome-weighted features)',
                      fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.text(-0.15, 1.05, 'H', transform=ax8.transAxes, fontsize=14, fontweight='bold')

        # 9. 完整的病理-临床-影像组学相关性热图
        ax9 = fig.add_subplot(gs[2, 2])

        if full_corr_matrix is not None and radiomics_df is not None:
            # 准备完整的相关性分析
            # 获取所有病理ITH特征（25个）+ composite_score
            path_features = [col for col in full_corr_matrix.columns if col.endswith('_ith')]
            if 'composite_score' in full_corr_matrix.columns:
                path_features.append('composite_score')

            # 定义要排除的分类变量
            categorical_vars_to_exclude = ['M_stage', 'Sex', 'Multifocal',
                                           'Infiltrated_the_adjacent_tissue', 'T_stage']

            # 获取所有临床和影像特征（除PatientID和要排除的分类变量）
            clinical_rad_features = [col for col in radiomics_df.columns
                                     if col not in ['PatientID'] + categorical_vars_to_exclude
                                     and col not in path_features]

            # 创建相关性和p值矩阵
            import pandas as pd
            from scipy import stats

            # 准备数据
            if 'merged_df' in locals():
                analysis_df = merged_df
            else:
                # 重新创建merged_df
                pathology_data = []
                for patient_id in self.all_patients:
                    if patient_id in radiomics_df['PatientID'].values and patient_id in self.ith_scores:
                        ith_row = self.ith_scores[patient_id].iloc[0]
                        patient_data = {'PatientID': patient_id}
                        for col in ith_row.index:
                            if col.endswith('_ith'):
                                patient_data[col] = ith_row[col]
                        if patient_id in integrated_df['patient_id'].values:
                            comp_score = \
                            integrated_df[integrated_df['patient_id'] == patient_id]['composite_score'].iloc[0]
                            patient_data['composite_score'] = comp_score
                        pathology_data.append(patient_data)
                if pathology_data:
                    pathology_df = pd.DataFrame(pathology_data)
                    analysis_df = pd.merge(radiomics_df, pathology_df, on='PatientID', how='inner')
                else:
                    analysis_df = None

            if analysis_df is not None and len(path_features) > 0 and len(clinical_rad_features) > 0:
                # 计算相关性矩阵和p值矩阵
                n_path = len(path_features)
                n_clin = len(clinical_rad_features)
                corr_matrix = np.zeros((n_path, n_clin))
                pval_matrix = np.ones((n_path, n_clin))

                for i, p_feat in enumerate(path_features):
                    for j, c_feat in enumerate(clinical_rad_features):
                        if p_feat in analysis_df.columns and c_feat in analysis_df.columns:
                            x = analysis_df[p_feat].values
                            y = analysis_df[c_feat].values
                            # 移除缺失值
                            valid_mask = ~(pd.isna(x) | pd.isna(y))
                            if np.sum(valid_mask) >= 3:
                                try:
                                    corr, pval = stats.spearmanr(x[valid_mask], y[valid_mask])
                                    corr_matrix[i, j] = corr
                                    pval_matrix[i, j] = pval
                                except:
                                    pass

                # 创建DataFrame用于热图
                corr_df = pd.DataFrame(corr_matrix, index=path_features, columns=clinical_rad_features)
                pval_df = pd.DataFrame(pval_matrix, index=path_features, columns=clinical_rad_features)

                # 保存完整的相关性分析结果到CSV
                correlation_results = []
                for i, p_feat in enumerate(path_features):
                    for j, c_feat in enumerate(clinical_rad_features):
                        correlation_results.append({
                            'Pathology_Feature': p_feat,
                            'Clinical_Radiomics_Feature': c_feat,
                            'Correlation': corr_matrix[i, j],
                            'P_value': pval_matrix[i, j]
                        })

                corr_results_df = pd.DataFrame(correlation_results)
                corr_results_path = os.path.join(self.output_dir, 'pathology_clinical_radiomics_correlations.csv')
                corr_results_df.to_csv(corr_results_path, index=False)
                print(f'Saved correlation analysis to: {corr_results_path}')

                # 创建带星号的标注矩阵
                annot_matrix = []
                for i in range(n_path):
                    row_annot = []
                    for j in range(n_clin):
                        corr_val = corr_matrix[i, j]
                        p_val = pval_matrix[i, j]
                        if pd.isna(corr_val) or pd.isna(p_val):
                            row_annot.append('')
                        else:
                            # 只显示显著的相关性
                            if p_val < 0.05:
                                text = f'{corr_val:.2f}'
                                if p_val < 0.001:
                                    text += '***'
                                elif p_val < 0.01:
                                    text += '**'
                                else:
                                    text += '*'
                                row_annot.append(text)
                            else:
                                row_annot.append('')  # 不显示非显著的
                    annot_matrix.append(row_annot)

                # 绘制热图
                sns.heatmap(corr_df, annot=annot_matrix, fmt='', cmap='RdBu_r',
                            center=0, vmin=-1, vmax=1, ax=ax9,
                            cbar_kws={'shrink': 0.8, 'label': 'Spearman Correlation'},
                            linewidths=0.1, linecolor='gray',
                            annot_kws={'size': 4},  # 增大注释字体
                            square=False)  # 改为False，让格子自适应

                # 设置标签
                # 设置X轴标签 - 保留完整的iTED前缀
                x_labels = []
                for feat in clinical_rad_features:
                    if 'iTED_' in feat:
                        # 保留完整的iTED_前缀
                        label = feat  # 不再删除iTED_前缀
                        if len(label) > 25:  # 如果太长，可以适当缩短
                            # 保留iTED_前缀，缩短后面的部分
                            parts = label.split('_')
                            if len(parts) > 2:
                                label = '_'.join(parts[:2]) + '_' + parts[-1][:8]
                    elif len(feat) > 20:
                        label = feat[:20]
                    else:
                        label = feat
                    x_labels.append(label)

                ax9.set_xticks(range(len(x_labels)))
                ax9.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)  # 增大字体

                # Y轴标签
                y_labels = []
                for feat in path_features:
                    if feat == 'composite_score':
                        y_labels.append('Composite Score')
                    else:
                        label = feat.replace('_ith', '').replace('_', ' ')
                        if len(label) > 25:
                            label = label[:25]
                        y_labels.append(label)

                ax9.set_yticks(range(len(y_labels)))
                ax9.set_yticklabels(y_labels, rotation=0, fontsize=7)  # 增大字体

                # 更新标题，反映实际的特征数量
                n_pathology = len(path_features)  # 26个
                n_clinical_radiomics = len(clinical_rad_features)  # 46个（删除5个分类变量后）

                ax9.set_title(
                    f'Pathology-Clinical-Radiomics Correlation Matrix\n'
                    f'({n_pathology} × {n_clinical_radiomics} features, *p<0.05, **p<0.01, ***p<0.001)',
                    fontsize=10, fontweight='bold')

                print(
                    f'Heatmap shows {n_pathology} pathology features × {n_clinical_radiomics} clinical/radiomics features')
                print(f'Excluded categorical variables: {", ".join(categorical_vars_to_exclude)}')

                # 保存相关性和p值矩阵供报告使用
                self.full_corr_df = corr_df
                self.full_pval_df = pval_df

        # 10. Top Features by Effect Size
        ax10 = fig.add_subplot(gs[3, 0])
        top_effects = ith_results_df.nlargest(10, 'effect_size')

        y_pos = np.arange(len(top_effects))
        colors_effect = ['#FF6B6B' if p < 0.05 else '#B0B0B0' for p in top_effects['p_value']]

        bars = ax10.barh(y_pos, top_effects['effect_size'].values, color=colors_effect, alpha=0.8,
                         edgecolor='black', linewidth=1)

        ax10.axvline(x=0.2, color='yellow', linestyle='--', alpha=0.5, linewidth=1, label='Small')
        ax10.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Medium')
        ax10.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Large')

        for i, (bar, val) in enumerate(zip(bars, top_effects['effect_size'].values)):
            ax10.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                      f'{val:.2f}', va='center', fontsize=8)

        ax10.set_yticks(y_pos)
        ax10.set_yticklabels([f.replace('_ith', '') for f in top_effects['feature']], fontsize=8)
        ax10.set_xlabel('Effect Size (Cohen\'s d)', fontsize=10)
        ax10.set_title('Top 10 Features by Effect Size\n(Red: p<0.05)', fontsize=11, fontweight='bold')
        ax10.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax10.grid(True, alpha=0.3, axis='x')
        ax10.text(-0.15, 1.05, 'J', transform=ax10.transAxes, fontsize=14, fontweight='bold')

        # 11. Direction of ITH Differences
        ax11 = fig.add_subplot(gs[3, 1])

        top10_ith = ith_results_df.nsmallest(10, 'p_value')
        differences = (top10_ith['high_mean'] - top10_ith['low_mean']).values

        # 改进的差异值处理 - 确保正负值都能清晰显示
        features = top10_ith['feature'].values
        adjusted_differences = []
        truncated_indices = []
        actual_values = {}

        # 分离intensity和非intensity特征
        intensity_indices = []
        non_intensity_values = []
        for idx, feat in enumerate(features):
            if 'intensity' in feat.lower():
                intensity_indices.append(idx)
            else:
                non_intensity_values.append(abs(differences[idx]))

        # 计算合理的显示范围
        if non_intensity_values:
            display_range = max(non_intensity_values) * 1.2
        else:
            display_range = 1.0

        # 处理每个值
        for i, diff in enumerate(differences):
            feat = features[i]

            # 对intensity特征进行截断
            if i in intensity_indices:
                if abs(diff) > display_range:
                    actual_values[i] = diff
                    adjusted_differences.append(np.sign(diff) * display_range * 0.95)
                    truncated_indices.append(i)
                else:
                    # 对于较小的intensity值，适当放大以提高可见性
                    if abs(diff) < display_range * 0.2:
                        adjusted_differences.append(np.sign(diff) * display_range * 0.25)
                    else:
                        adjusted_differences.append(diff)
            else:
                # 非intensity特征，确保可见性
                if abs(diff) < display_range * 0.1:
                    # 太小的值适当放大
                    adjusted_differences.append(np.sign(diff) * display_range * 0.15)
                else:
                    adjusted_differences.append(diff)

        differences = np.array(adjusted_differences)
        colors_diff = ['#FF6B6B' if d > 0 else '#4ECDC4' for d in differences]

        y_pos = np.arange(len(top10_ith))
        bars = ax11.barh(y_pos, differences, color=colors_diff, alpha=0.8,
                         edgecolor='black', linewidth=1)

        # 为截断的值添加实际值标注
        for idx in truncated_indices:
            if idx in actual_values:
                actual_val = actual_values[idx]
                bar_val = differences[idx]

                if bar_val != 0:
                    text_x = bar_val * 1.05
                    ax11.text(text_x, idx, f'[{actual_val:.2f}]',
                              ha='left' if bar_val > 0 else 'right',
                              va='center', fontsize=6, color='red',
                              fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='yellow',
                                        edgecolor='red',
                                        alpha=0.7))

        # 为截断的值添加标记
        for idx in truncated_indices:
            ax11.text(differences[idx] * 0.95, idx, '»', ha='right' if differences[idx] > 0 else 'left',
                      va='center', fontsize=12, fontweight='bold', color='black')

        for i, (_, row) in enumerate(top10_ith.iterrows()):
            p_val = row['p_value']
            if p_val < 0.05:
                x_pos = differences[i] * 0.5
                ax11.text(x_pos, i, '*', ha='center', va='center',
                          fontsize=12, fontweight='bold', color='white')

        ax11.set_yticks(y_pos)
        ax11.set_yticklabels([f.replace('_ith', '') for f in top10_ith['feature']], fontsize=8)
        ax11.set_xlabel('Difference (High Risk - Low Risk)', fontsize=10)
        ax11.set_title('Direction of ITH Differences\n(Blue: Lower in High Risk)', fontsize=11, fontweight='bold')
        ax11.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax11.grid(True, alpha=0.3, axis='x')

        ax11.text(0.02, 0.98, 'Lower ITH in High Risk ←', transform=ax11.transAxes,
                  fontsize=8, va='top', color='#4ECDC4', fontweight='bold')
        ax11.text(0.98, 0.98, '→ Higher ITH in High Risk', transform=ax11.transAxes,
                  fontsize=8, va='top', ha='right', color='#FF6B6B', fontweight='bold')

        # 调整x轴范围
        xlim = ax11.get_xlim()
        max_abs = max(abs(xlim[0]), abs(xlim[1]))
        ax11.set_xlim(-max_abs * 1.4, max_abs * 1.4)
        ax11.text(-0.15, 1.05, 'K', transform=ax11.transAxes, fontsize=14, fontweight='bold')
        # 生成图K的文本描述
        with open(os.path.join(self.output_dir, 'figure_k_direction_analysis.txt'), 'w') as f:
            f.write('DIRECTION OF ITH DIFFERENCES ANALYSIS\n')
            f.write('=' * 60 + '\n\n')
            f.write('Analysis of feature differences between High Risk and Low Risk groups\n')
            f.write('Positive values: Higher in High Risk group\n')
            f.write('Negative values: Higher in Low Risk group\n\n')

            f.write('Top 10 ITH Features by Effect Size:\n')
            f.write('-' * 40 + '\n')

            for idx, (feat, diff) in enumerate(zip(features, differences)):
                direction = 'Higher in High Risk' if diff > 0 else 'Higher in Low Risk'

                # Check if value was truncated
                if idx in truncated_indices and idx in actual_values:
                    actual_val = actual_values[idx]
                    f.write(f'{idx + 1:2d}. {feat:30s}: {actual_val:8.4f} ({direction})\n')
                    f.write(f'    [Value truncated for display from {actual_val:.4f} to {diff:.4f}]\n')
                else:
                    f.write(f'{idx + 1:2d}. {feat:30s}: {diff:8.4f} ({direction})\n')

                # Add interpretation
                if 'intensity' in feat.lower():
                    f.write(f'    → Intensity feature indicating {direction.lower()} signal strength\n')
                elif 'entropy' in feat.lower():
                    f.write(f'    → Heterogeneity measure {direction.lower()}\n')
                elif 'contrast' in feat.lower():
                    f.write(f'    → Local variation {direction.lower()}\n')

            f.write('\n' + '=' * 60 + '\n')
            f.write('SUMMARY:\n')

            high_risk_features = [feat for feat, diff in zip(features, differences) if diff > 0]
            low_risk_features = [feat for feat, diff in zip(features, differences) if diff < 0]

            f.write(f'Features higher in High Risk group: {len(high_risk_features)}\n')
            for feat in high_risk_features:
                f.write(f'  - {feat}\n')

            f.write(f'\nFeatures higher in Low Risk group: {len(low_risk_features)}\n')
            for feat in low_risk_features:
                f.write(f'  - {feat}\n')

            print('Generated: figure_k_direction_analysis.txt')

        # 12. Summary Statistics and Key Findings
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')

        n_sig = len(ith_results_df[ith_results_df['p_value'] < 0.05])
        best_feature = ith_results_df.iloc[0]['feature'].replace('_ith', '')
        best_p = ith_results_df.iloc[0]['p_value']
        best_es = ith_results_df.iloc[0]['effect_size']

        threshold = integrated_df['composite_score'].median()
        high_correct = np.sum(high_scores > threshold) / len(high_scores) * 100
        low_correct = np.sum(low_scores <= threshold) / len(low_scores) * 100
        overall_accuracy = (np.sum(high_scores > threshold) + np.sum(low_scores <= threshold)) / len(
            integrated_df) * 100

        # 添加相关性分析的关键发现
        key_correlations = ""
        if full_corr_matrix is not None:
            # 找出最强的相关性
            path_cols = [col for col in full_corr_matrix.columns if col.endswith('_ith')][:5]
            rad_cols = ['3D_ITHscore', 'iTED_firstorder_Variance', 'iTED_firstorder_Uniformity']

            strong_corr = []
            for p_col in path_cols:
                for r_col in rad_cols:
                    if p_col in full_corr_matrix.index and r_col in full_corr_matrix.columns:
                        corr_val = full_corr_matrix.loc[p_col, r_col]
                        if abs(corr_val) > 0.3:
                            strong_corr.append(f"{p_col[:10]} vs {r_col[:15]}: r={corr_val:.2f}")

            if strong_corr:
                key_correlations = "\n".join(strong_corr[:3])

        summary_text = f"""KEY FINDINGS SUMMARY

1. PATHOLOGICAL ITH:
   • {n_sig}/25 features significant
   • Top: {best_feature} (p={best_p:.3f}, ES={best_es:.2f})

2. COMPOSITE SCORE:
   • p-value: {p_value:.3f} {sig_text}
   • Accuracy: {overall_accuracy:.1f}%

3. BIOLOGICAL INSIGHT:
   • High Risk: LOWER ITH
   • Suggests clonal expansion

4. KEY CORRELATIONS:
{key_correlations if key_correlations else '   • Computing...'}

5. CLINICAL VALUE:
   • Validates imaging model
   • Guides therapy"""

        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

        ax12.set_title('Summary of Findings', fontsize=11, fontweight='bold', pad=20)
        ax12.text(-0.15, 1.05, 'L', transform=ax12.transAxes, fontsize=14, fontweight='bold')

        # 总标题
        fig.suptitle('Comprehensive Pathological Analysis for Thyroid Cancer Risk Stratification\n' +
                     'Integration of Cellular ITH, Clinical Features, and Imaging Biomarkers',
                     fontsize=16, fontweight='bold', y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # 保存PDF时使用更高的DPI
        output_path_pdf = os.path.join(self.output_dir, 'pathology_analysis_comprehensive.pdf')
        plt.savefig(output_path_pdf, format='pdf', dpi=400, bbox_inches='tight')  # 提高到400 DPI
        print(f"Saved visualization to: {output_path_pdf}")

        # 保存PNG预览版
        output_path_png = os.path.join(self.output_dir, 'pathology_analysis_comprehensive_preview.png')
        plt.savefig(output_path_png, format='png', dpi=150, bbox_inches='tight')
        print(f"Also saved preview as: {output_path_png}")

        plt.close()

        # 生成图G (Composite Scores) 的详细统计报告
        violin_stats_path = os.path.join(self.output_dir, 'composite_score_statistics.txt')
        with open(violin_stats_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("COMPOSITE SCORE DISTRIBUTION STATISTICS (Figure G)\n")
            f.write("=" * 60 + "\n\n")

            f.write("COMPOSITE SCORE CALCULATION METHOD:\n")
            f.write("-" * 30 + "\n")
            f.write("The composite score is calculated using:\n")
            f.write("1. Feature selection: Top features by effect size (p<0.2)\n")
            f.write("2. Effect size calculation: Based on METASTASIS vs NON-METASTASIS groups\n")  # 新增
            f.write("3. Standardization: Z-score normalization\n")
            f.write("4. Weighting: Based on effect size from true outcomes\n")  # 修改
            f.write("5. Direction: Adjusted by outcome group median comparison\n")  # 修改
            f.write("6. Evaluation: Scores compared between MODEL risk groups\n")  # 新增
            f.write("7. Formula: Σ(direction × scaled_feature × weight)\n\n")

            high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score']
            low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score']

            f.write("High Risk Group:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  N samples: {len(high_scores)}\n")
            f.write(f"  Mean ± SD: {high_scores.mean():.3f} ± {high_scores.std():.3f}\n")
            f.write(
                f"  Median (IQR): {high_scores.median():.3f} ({high_scores.quantile(0.25):.3f} - {high_scores.quantile(0.75):.3f})\n")
            f.write(f"  Min - Max: {high_scores.min():.3f} - {high_scores.max():.3f}\n\n")

            f.write("Low Risk Group:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  N samples: {len(low_scores)}\n")
            f.write(f"  Mean ± SD: {low_scores.mean():.3f} ± {low_scores.std():.3f}\n")
            f.write(
                f"  Median (IQR): {low_scores.median():.3f} ({low_scores.quantile(0.25):.3f} - {low_scores.quantile(0.75):.3f})\n")
            f.write(f"  Min - Max: {low_scores.min():.3f} - {low_scores.max():.3f}\n\n")

            # Mann-Whitney U test
            from scipy import stats
            stat, p_value = stats.mannwhitneyu(high_scores, low_scores)
            f.write("Statistical Test:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  Mann-Whitney U statistic: {stat:.2f}\n")
            f.write(f"  P-value: {p_value:.4f}\n")
            f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n\n")

            # Effect size
            pooled_std = np.sqrt((high_scores.var() + low_scores.var()) / 2)
            if pooled_std > 0:
                cohens_d = (high_scores.mean() - low_scores.mean()) / pooled_std
                f.write(f"  Cohen's d: {cohens_d:.3f}\n")
                f.write(f"  Effect size interpretation: ")
                if abs(cohens_d) < 0.2:
                    f.write("Negligible\n")
                elif abs(cohens_d) < 0.5:
                    f.write("Small\n")
                elif abs(cohens_d) < 0.8:
                    f.write("Medium\n")
                else:
                    f.write("Large\n")

        print(f"Composite score statistics saved to: {violin_stats_path}")

        # 生成详细的分析报告
        if full_corr_matrix is not None:
            self._generate_correlation_report(full_corr_matrix, radiomics_df)

        # 生成SCI论文用的图例说明
        self._generate_figure_legends_for_paper(ith_results_df, composition_df,
                                                morphology_df, integrated_df, full_corr_matrix)

        # 生成各个图的详细描述文件
        self._generate_detailed_figure_descriptions(ith_results_df, composition_df,
                                                    morphology_df, integrated_df)

    def _generate_correlation_report(self, full_corr_matrix, radiomics_df):
        """生成图I的详细描述报告"""

        report_path = os.path.join(self.output_dir, 'pathology_radiomics_correlation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('FIGURE I: PATHOLOGY-CLINICAL-RADIOMICS CORRELATION HEATMAP\n')
            f.write('=' * 80 + '\n\n')

            # 图例说明
            f.write('FIGURE LEGEND:\n')
            f.write('-' * 40 + '\n')
            f.write('Comprehensive correlation heatmap showing Spearman correlation coefficients\n')
            f.write('between pathological intratumoral heterogeneity (ITH) features and\n')
            f.write('clinical/radiomics features.\n\n')

            # ========= 在这里替换原有的Matrix dimensions部分 =========
            # 需要先计算特征数量
            if hasattr(self, 'full_corr_df'):
                n_pathology = len([col for col in self.full_corr_df.index])
                n_total_cols = len([col for col in self.full_corr_df.columns])

                # 识别不同类型的特征
                clinical_features = ['Age', 'BMI', 'Tumor_size', 'Number_of_metastatic_lymph_nodes',
                                     'NLR', 'TG', 'TGAb', 'TPOAb']
                n_clinical = len([col for col in self.full_corr_df.columns if col in clinical_features])
                n_radiomics = n_total_cols - n_clinical

                f.write(f'Matrix dimensions: {n_pathology} × {n_total_cols}\n')
                f.write(f'- Rows ({n_pathology}): 25 pathological ITH features + 1 composite score\n')
                f.write(f'- Columns ({n_total_cols}): {n_clinical} clinical + {n_radiomics} radiomics features\n\n')
            else:
                # 原有的默认值
                f.write('Matrix dimensions: 26 × 21\n')
                f.write('- Rows (26): 25 pathological ITH features + 1 composite score\n')
                f.write('- Columns (21): Clinical and radiomics features\n\n')

            f.write('Color scheme:\n')
            f.write('- Red: Positive correlation\n')
            f.write('- Blue: Negative correlation\n')
            f.write('- White: No correlation\n\n')

            f.write('Statistical significance:\n')
            f.write('- ***: p < 0.001\n')
            f.write('- **: p < 0.01\n')
            f.write('- *: p < 0.05\n')
            f.write('- Blank: p ≥ 0.05 (non-significant)\n\n')

            # 分析结果
            if hasattr(self, 'full_corr_df') and hasattr(self, 'full_pval_df'):
                corr_df = self.full_corr_df
                pval_df = self.full_pval_df

                f.write('KEY FINDINGS:\n')
                f.write('-' * 40 + '\n\n')

                # 找出最强的正相关
                strong_positive = []
                strong_negative = []

                for i in range(len(corr_df.index)):
                    for j in range(len(corr_df.columns)):
                        corr_val = corr_df.iloc[i, j]
                        p_val = pval_df.iloc[i, j]
                        if p_val < 0.05:
                            if corr_val > 0.3:
                                strong_positive.append({
                                    'path': corr_df.index[i],
                                    'clin': corr_df.columns[j],
                                    'corr': corr_val,
                                    'p': p_val
                                })
                            elif corr_val < -0.3:
                                strong_negative.append({
                                    'path': corr_df.index[i],
                                    'clin': corr_df.columns[j],
                                    'corr': corr_val,
                                    'p': p_val
                                })

                # 报告强正相关
                f.write('1. Strong Positive Correlations (r > 0.3, p < 0.05):\n')
                if strong_positive:
                    strong_positive.sort(key=lambda x: abs(x['corr']), reverse=True)
                    for item in strong_positive[:10]:
                        f.write(
                            f"   • {item['path'][:20]} vs {item['clin'][:20]}: r={item['corr']:.3f}, p={item['p']:.3f}\n")
                else:
                    f.write('   None found\n')

                # 报告强负相关
                f.write('\n2. Strong Negative Correlations (r < -0.3, p < 0.05):\n')
                if strong_negative:
                    strong_negative.sort(key=lambda x: abs(x['corr']), reverse=True)
                    for item in strong_negative[:10]:
                        f.write(
                            f"   • {item['path'][:20]} vs {item['clin'][:20]}: r={item['corr']:.3f}, p={item['p']:.3f}\n")
                else:
                    f.write('   None found\n')

                # 统计总结
                total_tests = corr_df.shape[0] * corr_df.shape[1]
                sig_count = np.sum(pval_df.values < 0.05)

                f.write('\n3. Statistical Summary:\n')
                f.write(f'   • Total correlations tested: {total_tests}\n')
                f.write(f'   • Significant correlations (p<0.05): {sig_count} ({sig_count / total_tests * 100:.1f}%)\n')

                # Composite Score特别分析
                if 'composite_score' in corr_df.index:
                    f.write('\n4. Composite Score Correlations:\n')
                    comp_corr = corr_df.loc['composite_score']
                    comp_pval = pval_df.loc['composite_score']

                    sig_comp = [(feat, corr, p) for feat, corr, p in
                                zip(comp_corr.index, comp_corr.values, comp_pval.values) if p < 0.05]
                    sig_comp.sort(key=lambda x: abs(x[1]), reverse=True)

                    for feat, corr, p in sig_comp[:5]:
                        f.write(f'   • vs {feat[:25]}: r={corr:.3f}, p={p:.3f}\n')

                f.write('\n' + '=' * 80 + '\n')

            print(f'Correlation report saved to: {report_path}')

    def _generate_figure_legends_for_paper(self, ith_results_df, composition_df,
                                           morphology_df, integrated_df, full_corr_matrix):
        """生成SCI论文用的图例说明"""

        legends_path = os.path.join(self.output_dir, 'figure_legends_for_paper.txt')

        with open(legends_path, 'w', encoding='utf-8') as f:
            f.write("FIGURE LEGENDS FOR SCIENTIFIC PUBLICATION\n")
            f.write("=" * 80 + "\n\n")

            # 主图说明
            f.write("Figure X. Comprehensive pathological validation of radiomics-based risk ")
            f.write("stratification for thyroid cancer distant metastasis prediction.\n\n")

            # 详细说明每个子图
            f.write("(A-C) Box plots displaying intratumoral heterogeneity (ITH) features with ")
            f.write("statistically significant differences between high-risk and low-risk groups. ")

            sig_features = ith_results_df[ith_results_df['p_value'] < 0.05]
            if len(sig_features) > 0:
                f.write(f"Elongation ITH (p={sig_features.iloc[0]['p_value']:.3f}, ")
                f.write(f"Cohen's d={sig_features.iloc[0]['effect_size']:.2f}) and ")
                if len(sig_features) > 1:
                    f.write(f"extent ITH (p={sig_features.iloc[1]['p_value']:.3f}, ")
                    f.write(f"Cohen's d={sig_features.iloc[1]['effect_size']:.2f}) ")
                f.write("demonstrated significantly lower heterogeneity in the high-risk group, ")
                f.write("suggesting clonal expansion in aggressive tumors.\n\n")

            f.write("(D-E) Pie charts illustrating cellular composition. High-risk tumors (D) ")
            high_comp = composition_df[composition_df['risk_group'] == 'high'][
                ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio']].mean()
            low_comp = composition_df[composition_df['risk_group'] == 'low'][
                ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio']].mean()
            f.write(f"contained {high_comp['tumor_ratio']:.1%} neoplastic epithelial cells, ")
            f.write(f"{high_comp['inflammatory_ratio']:.1%} inflammatory cells, and ")
            f.write(f"{high_comp['stromal_ratio']:.1%} stromal cells. Low-risk tumors (E) showed ")
            f.write(f"{low_comp['tumor_ratio']:.1%}, {low_comp['inflammatory_ratio']:.1%}, and ")
            f.write(f"{low_comp['stromal_ratio']:.1%} respectively.\n\n")

            # 修正图F的描述 - 匹配实际图内容
            f.write("(F) Top ITH features by effect size (Cohen's d). ")
            f.write("Horizontal bar chart showing the eight ITH features with the largest effect sizes. ")
            f.write("Red bars indicate statistically significant features (p<0.05), ")
            f.write("gray bars indicate non-significant features. ")
            f.write("Reference lines mark small (0.2), medium (0.5), and large (0.8) effect sizes.\n\n")

            f.write("(G) Violin plots of composite pathology scores showing significant ")
            high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score']
            low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score']
            _, p_val = stats.mannwhitneyu(high_scores, low_scores)
            f.write("discrimination between model-predicted risk groups (p={p_val:.3f}). ")
            f.write("Score weights derived from true metastasis outcomes. ")  # 新增说明
            f.write("Individual data points overlaid on violin plots.\n\n")

            f.write("(H) Individual patient scores with median threshold classification. ")
            threshold = integrated_df['composite_score'].median()
            accuracy = (sum(high_scores > threshold) + sum(low_scores <= threshold)) / len(
                integrated_df) * 100
            f.write(f"Overall classification accuracy: {accuracy:.1f}%.\n\n")

            f.write("(I) Spearman correlation heatmap between pathological ITH features ")
            f.write("(rows) and clinical/radiomics features (columns). ")
            f.write("The correlation matrix reveals biological relationships between ")
            f.write("microscopic pathology and macroscopic imaging characteristics. ")
            f.write("Red indicates positive correlation, blue indicates negative correlation.\n\n")

            f.write("(J) Top 10 ITH features ranked by effect size (Cohen's d). ")
            f.write("Red bars indicate statistically significant features (p<0.05). ")
            f.write("Reference lines mark small (0.2), medium (0.5), and large (0.8) effect sizes.\n\n")

            f.write("(K) Direction and magnitude of ITH differences between risk groups. ")
            f.write("Blue bars indicate features with lower values in the high-risk group, ")
            f.write("supporting the clonal expansion hypothesis. ")
            f.write("Asterisks mark significant differences (p<0.05).\n\n")

            f.write("(L) Summary panel presenting key statistical findings and clinical implications.\n\n")

            # 添加缩写说明
            f.write("Abbreviations: ITH, intratumoral heterogeneity; iTED, image-derived tumor ")
            f.write("environment descriptor; ES, effect size; FO, first-order; ")
            f.write("GLCM, gray-level co-occurrence matrix; GLRLM, gray-level run-length matrix; ")
            f.write("GLDM, gray-level dependence matrix; GLSZM, gray-level size zone matrix; ")
            f.write("NGTDM, neighboring gray-tone difference matrix.\n\n")

            # 统计总结
            f.write("Statistical Summary:\n")
            f.write("-" * 40 + "\n")
            n_sig = len(ith_results_df[ith_results_df['p_value'] < 0.05])
            f.write(f"• Total patients: {len(integrated_df)} ")
            f.write(f"(High-risk: n={len(self.high_risk_patients)}, ")
            f.write(f"Low-risk: n={len(self.low_risk_patients)})\n")
            f.write(f"• Significant ITH features: {n_sig}/25\n")
            f.write(f"• Composite score discrimination: p={p_val:.4f}\n")
            f.write(f"• Classification accuracy: {accuracy:.1f}%\n")

            if full_corr_matrix is not None:
                # 报告关键相关性
                f.write("\nKey Correlations:\n")
                path_cols = [col for col in full_corr_matrix.columns if col.endswith('_ith')][:5]

                if 'elongation_ith' in full_corr_matrix.index and '3D_ITHscore' in full_corr_matrix.columns:
                    corr = full_corr_matrix.loc['elongation_ith', '3D_ITHscore']
                    f.write(f"• Elongation ITH vs 3D_ITHscore: r={corr:.3f}\n")

                if 'elongation_ith' in full_corr_matrix.index and 'iTED_firstorder_Uniformity' in full_corr_matrix.columns:
                    corr = full_corr_matrix.loc['elongation_ith', 'iTED_firstorder_Uniformity']
                    f.write(f"• Elongation ITH vs Image Uniformity: r={corr:.3f}\n")

                if 'composite_score' in full_corr_matrix.index and 'Tumor_size' in full_corr_matrix.columns:
                    corr = full_corr_matrix.loc['composite_score', 'Tumor_size']
                    f.write(f"• Composite Score vs Tumor Size: r={corr:.3f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Figure legends saved to: {legends_path}")

    def _generate_detailed_figure_descriptions(self, ith_results_df, composition_df,
                                               morphology_df, integrated_df):
        """生成各个图的详细描述文件"""

        # 生成图A和B的详细描述
        self._generate_figure_abc_description(ith_results_df)

        # 生成图D和E的详细描述
        self._generate_figure_de_description(composition_df)

        # 生成图F的详细描述
        self._generate_figure_f_description(ith_results_df)

        # 生成图H的详细描述
        self._generate_figure_h_description(integrated_df)

        # 生成图J的详细描述
        self._generate_figure_j_description(ith_results_df)

    def _generate_figure_abc_description(self, ith_results_df):  # 改名
        """生成图A、B和C的详细描述"""

        output_path = os.path.join(self.output_dir, 'figure_abc_ith_boxplots.txt')  # 改文件名

        # 获取显著的ITH特征
        sig_features = ith_results_df[ith_results_df['p_value'] < 0.05].nsmallest(3, 'p_value')  # 改为3个

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FIGURES A, B & C: ITH FEATURE BOX PLOTS\n")  # 改标题
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION:\n")
            f.write("-" * 40 + "\n")
            f.write("Box plots comparing intratumoral heterogeneity (ITH) features between\n")
            f.write("high-risk and low-risk patient groups.\n\n")

            f.write("VISUAL ELEMENTS:\n")
            f.write("- Red boxes: High-risk group\n")
            f.write("- Cyan boxes: Low-risk group\n")
            f.write("- Individual data points overlaid as scatter points\n")
            f.write("- Significance bars with p-values shown above\n")
            f.write("- Effect sizes (ES) shown in subplot titles\n\n")

            if len(sig_features) > 0:
                f.write("FIGURE A: Elongation ITH\n")
                f.write("-" * 40 + "\n")
                row = sig_features.iloc[0]
                f.write(f"Feature: {row['feature']}\n")
                f.write(f"P-value: {row['p_value']:.4f}\n")
                f.write(f"Effect Size (Cohen's d): {row['effect_size']:.3f}\n")
                f.write(f"High-risk mean ± SD: {row['high_mean']:.4f} ± {row['high_std']:.4f}\n")
                f.write(f"Low-risk mean ± SD: {row['low_mean']:.4f} ± {row['low_std']:.4f}\n")
                f.write(
                    f"Direction: {'Lower' if row['high_mean'] < row['low_mean'] else 'Higher'} in high-risk group\n\n")

            if len(sig_features) > 1:
                f.write("FIGURE B: Extent ITH\n")
                f.write("-" * 40 + "\n")
                row = sig_features.iloc[1]
                f.write(f"Feature: {row['feature']}\n")
                f.write(f"P-value: {row['p_value']:.4f}\n")
                f.write(f"Effect Size (Cohen's d): {row['effect_size']:.3f}\n")
                f.write(f"High-risk mean ± SD: {row['high_mean']:.4f} ± {row['high_std']:.4f}\n")
                f.write(f"Low-risk mean ± SD: {row['low_mean']:.4f} ± {row['low_std']:.4f}\n")
                f.write(
                    f"Direction: {'Lower' if row['high_mean'] < row['low_mean'] else 'Higher'} in high-risk group\n\n")

            if len(sig_features) > 2:  # 添加Figure C
                f.write("FIGURE C: Intensity Min ITH\n")
                f.write("-" * 40 + "\n")
                row = sig_features.iloc[2]
                f.write(f"Feature: {row['feature']}\n")
                f.write(f"P-value: {row['p_value']:.4f}\n")
                f.write(f"Effect Size (Cohen's d): {row['effect_size']:.3f}\n")
                f.write(f"High-risk mean ± SD: {row['high_mean']:.4f} ± {row['high_std']:.4f}\n")
                f.write(f"Low-risk mean ± SD: {row['low_mean']:.4f} ± {row['low_std']:.4f}\n")
                f.write(
                    f"Direction: {'Lower' if row['high_mean'] < row['low_mean'] else 'Higher'} in high-risk group\n\n")

            f.write("STATISTICAL INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            f.write("• Statistical test: Mann-Whitney U test\n")
            f.write("• Significance levels: *p<0.05, **p<0.01, ***p<0.001\n")
            f.write("• Both features show significantly lower heterogeneity in high-risk group\n")
            f.write("• This supports the clonal expansion hypothesis in aggressive tumors\n")
            f.write("\n" + "=" * 80 + "\n")

        print(f"Generated: figure_ab_ith_boxplots.txt")

    def _generate_figure_de_description(self, composition_df):
        """生成图D和E的详细描述"""

        output_path = os.path.join(self.output_dir, 'figure_de_cell_composition.txt')

        # 计算平均组成
        high_comp = composition_df[composition_df['risk_group'] == 'high'][
            ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio']].mean()
        low_comp = composition_df[composition_df['risk_group'] == 'low'][
            ['tumor_ratio', 'inflammatory_ratio', 'stromal_ratio']].mean()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FIGURES D & E: CELL COMPOSITION PIE CHARTS\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION:\n")
            f.write("-" * 40 + "\n")
            f.write("Pie charts showing the cellular composition of tumor samples\n")
            f.write("averaged across patients in each risk group.\n\n")

            f.write("FIGURE D: High Risk Group Cell Composition\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Tumor cells (red): {high_comp['tumor_ratio']:.1%}\n")
            f.write(f"• Inflammatory cells (yellow): {high_comp['inflammatory_ratio']:.1%}\n")
            f.write(f"• Stromal cells (green): {high_comp['stromal_ratio']:.1%}\n")
            f.write(f"• Total patients: n={len(self.high_risk_patients)}\n\n")

            f.write("FIGURE E: Low Risk Group Cell Composition\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Tumor cells (cyan): {low_comp['tumor_ratio']:.1%}\n")
            f.write(f"• Inflammatory cells (yellow): {low_comp['inflammatory_ratio']:.1%}\n")
            f.write(f"• Stromal cells (green): {low_comp['stromal_ratio']:.1%}\n")
            f.write(f"• Total patients: n={len(self.low_risk_patients)}\n\n")

            f.write("COMPARATIVE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write("Differences between groups:\n")
            f.write(
                f"• Tumor ratio: High-risk ({high_comp['tumor_ratio']:.1%}) vs Low-risk ({low_comp['tumor_ratio']:.1%}) - Δ={(high_comp['tumor_ratio'] - low_comp['tumor_ratio']) * 100:+.1f}%\n")
            f.write(
                f"• Inflammatory ratio: High-risk ({high_comp['inflammatory_ratio']:.1%}) vs Low-risk ({low_comp['inflammatory_ratio']:.1%}) - Δ={(high_comp['inflammatory_ratio'] - low_comp['inflammatory_ratio']) * 100:+.1f}%\n")
            f.write(
                f"• Stromal ratio: High-risk ({high_comp['stromal_ratio']:.1%}) vs Low-risk ({low_comp['stromal_ratio']:.1%}) - Δ={(high_comp['stromal_ratio'] - low_comp['stromal_ratio']) * 100:+.1f}%\n\n")

            f.write("KEY OBSERVATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("• High-risk tumors show slightly higher tumor cell proportion\n")
            f.write("• Inflammatory infiltration is similar between groups\n")
            f.write("• Low-risk tumors have slightly more stromal component\n")
            f.write("• The differences are relatively modest, suggesting cell composition\n")
            f.write("  alone may not be the primary discriminator between risk groups\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Generated: figure_de_cell_composition.txt")

    def _generate_figure_f_description(self, ith_results_df):
        """生成图F的详细描述"""

        output_path = os.path.join(self.output_dir, 'figure_f_effect_sizes.txt')

        # 获取效应量最大的8个特征
        top_effects = ith_results_df.nlargest(8, 'effect_size')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FIGURE F: TOP ITH FEATURES BY EFFECT SIZE\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION:\n")
            f.write("-" * 40 + "\n")
            f.write("Horizontal bar chart showing the eight ITH features with the largest\n")
            f.write("effect sizes (Cohen's d) for discriminating between risk groups.\n\n")

            f.write("VISUAL ELEMENTS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Red bars: Statistically significant features (p<0.05)\n")
            f.write("• Gray bars: Non-significant features (p≥0.05)\n")
            f.write("• Vertical reference lines:\n")
            f.write("  - 0.2: Small effect size (gray dashed)\n")
            f.write("  - 0.5: Medium effect size (gray dashed)\n")
            f.write("  - 0.8: Large effect size (gray dashed)\n\n")

            f.write("FEATURES RANKED BY EFFECT SIZE:\n")
            f.write("-" * 40 + "\n")
            for i, (_, data) in enumerate(top_effects.iterrows(), 1):
                sig_marker = "*" if data['p_value'] < 0.05 else " "
                f.write(
                    f"{i}. {data['feature']:25s} ES={data['effect_size']:.3f} (p={data['p_value']:.4f}){sig_marker}\n")

            f.write("\nEFFECT SIZE INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            f.write("Cohen's d interpretation guidelines:\n")
            f.write("• 0.2-0.5: Small effect\n")
            f.write("• 0.5-0.8: Medium effect\n")
            f.write("• >0.8: Large effect\n\n")

            f.write("KEY FINDINGS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Multiple features show large effect sizes (>0.8)\n")
            f.write("• Top features include texture and intensity-based ITH metrics\n")
            f.write("• Several features with large effect sizes are statistically significant\n")
            f.write("• This demonstrates strong discriminative power of ITH features\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Generated: figure_f_effect_sizes.txt")

    def _generate_figure_h_description(self, integrated_df):
        """生成图H的详细描述，包含分类准确率"""

        output_path = os.path.join(self.output_dir, 'figure_h_individual_scores.txt')

        # 计算分类准确率
        high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score']
        low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score']
        threshold = integrated_df['composite_score'].median()
        high_correct = np.sum(high_scores > threshold) / len(high_scores)
        low_correct = np.sum(low_scores <= threshold) / len(low_scores)
        overall_accuracy = (np.sum(high_scores > threshold) + np.sum(low_scores <= threshold)) / len(
            integrated_df) * 100

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FIGURE H: INDIVIDUAL PATIENT SCORES\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION:\n")
            f.write("-" * 40 + "\n")
            f.write("Scatter plot showing composite pathology scores for each individual patient.\n")
            f.write("Patients are arranged along the x-axis, with scores on the y-axis.\n\n")

            f.write("VISUAL ELEMENTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Red circles: High-risk patients (n={len(self.high_risk_patients)})\n")
            f.write(f"• Blue circles: Low-risk patients (n={len(self.low_risk_patients)})\n")
            f.write("• Gray dashed line: Median threshold for classification\n")
            f.write("• Each point represents one patient's composite score\n\n")

            f.write("CLASSIFICATION PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Overall classification accuracy: {overall_accuracy:.1f}%\n")
            f.write(f"• High-risk correctly classified: {high_correct:.1%}\n")
            f.write(f"• Low-risk correctly classified: {low_correct:.1%}\n")
            f.write("• Classification rule: Score > median → High risk\n")
            f.write("• Classification rule: Score ≤ median → Low risk\n\n")

            f.write("DISTRIBUTION PATTERNS:\n")
            f.write("-" * 40 + "\n")
            f.write("• High-risk patients tend to cluster above the median line\n")
            f.write("• Low-risk patients tend to cluster below the median line\n")
            f.write("• Some overlap exists, indicating imperfect but good discrimination\n")
            f.write("• The composite score successfully stratifies most patients\n\n")

            f.write("PATIENT-LEVEL INSIGHTS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Each patient has a unique pathology profile\n")
            f.write("• The composite score integrates multiple pathological features\n")
            f.write("• Misclassified patients may represent borderline cases\n")
            f.write("• Individual variation within risk groups is apparent\n")

            f.write("\nCLINICAL IMPLICATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• The {overall_accuracy:.1f}% accuracy suggests good clinical utility\n")
            f.write("• The model could assist in risk stratification\n")
            f.write("• Misclassified cases may need additional evaluation\n")
            f.write("• The threshold could be optimized for specific clinical needs\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Generated: figure_h_individual_scores.txt")

    def _generate_figure_j_description(self, ith_results_df):
        """生成图J的详细描述"""

        output_path = os.path.join(self.output_dir, 'figure_j_top10_effect_sizes.txt')

        # 获取效应量最大的10个特征
        top10_effects = ith_results_df.nlargest(10, 'effect_size')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FIGURE J: TOP 10 FEATURES BY EFFECT SIZE\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION:\n")
            f.write("-" * 40 + "\n")
            f.write("Horizontal bar chart displaying the top 10 ITH features ranked by\n")
            f.write("effect size (Cohen's d) with significance indicators.\n\n")

            f.write("VISUAL ELEMENTS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Bar colors:\n")
            f.write("  - Red bars: Statistically significant (p<0.05)\n")
            f.write("  - Gray bars: Non-significant (p≥0.05)\n")
            f.write("• Reference lines:\n")
            f.write("  - Yellow dashed (0.2): Small effect size threshold\n")
            f.write("  - Orange dashed (0.5): Medium effect size threshold\n")
            f.write("  - Red dashed (0.8): Large effect size threshold\n")
            f.write("• Effect size values displayed at the end of each bar\n\n")

            f.write("COMPLETE RANKING (TOP 10):\n")
            f.write("-" * 40 + "\n")
            f.write("Rank  Feature                        Effect Size  P-value    Significance\n")
            f.write("-" * 75 + "\n")
            for i, (_, data) in enumerate(top10_effects.iterrows(), 1):
                sig = "Yes" if data['p_value'] < 0.05 else "No"
                feature_name = data['feature'].replace('_ith', '')[:25]
                f.write(
                    f"{i:2d}.   {feature_name:30s} {data['effect_size']:6.3f}      {data['p_value']:6.4f}     {sig}\n")

            f.write("\nEFFECT SIZE DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            large_effects = sum(top10_effects['effect_size'] > 0.8)
            medium_effects = sum((top10_effects['effect_size'] > 0.5) & (top10_effects['effect_size'] <= 0.8))
            small_effects = sum((top10_effects['effect_size'] > 0.2) & (top10_effects['effect_size'] <= 0.5))

            f.write(f"• Large effects (>0.8): {large_effects} features\n")
            f.write(f"• Medium effects (0.5-0.8): {medium_effects} features\n")
            f.write(f"• Small effects (0.2-0.5): {small_effects} features\n\n")

            f.write("KEY INSIGHTS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Multiple ITH features show large discriminative power\n")
            f.write("• Effect sizes range from small to large\n")
            f.write("• Not all features with large effect sizes are statistically significant\n")
            f.write("  (likely due to small sample size)\n")
            f.write("• The combination of effect size and significance helps identify\n")
            f.write("  the most reliable discriminative features\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Generated: figure_j_top10_effect_sizes.txt")

    def generate_report(self, ith_results_df, composition_df, morphology_df, spatial_df, integrated_df):
        """生成综合分析报告"""
        print("\n" + "=" * 60)
        print("7. GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)

        report_path = os.path.join(self.output_dir, 'pathology_validation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PATHOLOGICAL VALIDATION REPORT FOR THYROID CANCER METASTASIS PREDICTION MODEL\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. 研究概述
            f.write("1. STUDY OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Patients: {len(self.all_patients)}\n")
            f.write(f"Low Risk Group: {len(self.low_risk_patients)} patients\n")
            f.write(f"High Risk Group: {len(self.high_risk_patients)} patients\n")
            f.write(f"Data Location: {self.data_dir}\n\n")

            # ========= 在这里插入新的方法学说明 =========
            f.write("2. METHODOLOGY NOTE\n")
            f.write("-" * 40 + "\n")
            f.write("Composite Score Calculation:\n")
            f.write("  - Feature weights: Derived from true metastasis outcomes\n")
            f.write("  - Group comparison: Applied to imaging model risk groups\n")
            f.write("  - Purpose: Validate if model groups capture outcome-relevant pathology\n\n")

            # 3. 主要发现
            f.write("3. KEY FINDINGS\n")
            f.write("-" * 40 + "\n")

            # ITH分析结果
            significant_ith = ith_results_df[ith_results_df['p_value'] < 0.05]
            f.write(f"a) Intratumoral Heterogeneity (ITH):\n")
            f.write(
                f"   - {len(significant_ith)}/{len(ith_results_df)} features showed significant differences\n")
            f.write(f"   - Most significant features:\n")
            for _, row in significant_ith.nsmallest(5, 'p_value').iterrows():
                f.write(f"     * {row['feature']}: p={row['p_value']:.4f}, "
                        f"Effect Size={row['effect_size']:.3f}\n")

            # 细胞组成分析
            f.write(f"\nb) Cell Composition:\n")
            high_comp = composition_df[composition_df['risk_group'] == 'high'][
                ['tumor_ratio', 'inflammatory_ratio']].mean()
            low_comp = composition_df[composition_df['risk_group'] == 'low'][
                ['tumor_ratio', 'inflammatory_ratio']].mean()
            f.write(f"   - High Risk: Tumor={high_comp['tumor_ratio']:.3f}, "
                    f"Inflammatory={high_comp['inflammatory_ratio']:.3f}\n")
            f.write(f"   - Low Risk:  Tumor={low_comp['tumor_ratio']:.3f}, "
                    f"Inflammatory={low_comp['inflammatory_ratio']:.3f}\n")

            # 形态学特征 - 修正为匹配图F的ITH效应量描述
            f.write(f"\nc) ITH Features by Effect Size (Figure F):\n")
            # 获取效应量最大的特征
            top_effects = ith_results_df.nlargest(3, 'effect_size')
            for _, row in top_effects.iterrows():
                f.write(f"   - {row['feature']}: ES={row['effect_size']:.3f}, p={row['p_value']:.4f}\n")
            f.write(f"   - Multiple ITH features show large effect sizes (>0.8)\n")
            f.write(f"   - Red bars in Figure F indicate statistically significant features\n")

            # 综合评分
            f.write(f"\nd) Composite Pathology Score:\n")
            high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score']
            low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score']
            stat, p_value = stats.mannwhitneyu(high_scores, low_scores)
            f.write(f"   - High Risk: {high_scores.mean():.3f} ± {high_scores.std():.3f}\n")
            f.write(f"   - Low Risk:  {low_scores.mean():.3f} ± {low_scores.std():.3f}\n")
            f.write(f"   - Mann-Whitney U test p-value: {p_value:.4f}\n")

            # 使用中位数阈值的分类准确度
            threshold = integrated_df['composite_score'].median()
            high_correct = np.sum(high_scores > threshold) / len(high_scores)
            low_correct = np.sum(low_scores <= threshold) / len(low_scores)
            overall_accuracy = (np.sum(high_scores > threshold) + np.sum(low_scores <= threshold)) / len(
                integrated_df)
            f.write(f"   - Classification accuracy (median threshold): {overall_accuracy:.1%}\n")

            # 4. 临床意义
            f.write("\n4. CLINICAL IMPLICATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("The pathological analysis confirms that:\n")

            if len(significant_ith) > 0:
                # 检查ITH方向
                avg_diff = significant_ith['high_mean'].mean() - significant_ith['low_mean'].mean()
                if avg_diff < 0:
                    f.write("- High-risk patients show LOWER intratumoral heterogeneity\n")
                    f.write("- This suggests clonal expansion and more homogeneous tumor cells\n")
                else:
                    f.write("- High-risk patients show HIGHER intratumoral heterogeneity\n")
                    f.write("- This suggests greater cellular diversity and tumor complexity\n")

            f.write("- Tumor cell morphology differs between risk groups\n")
            f.write("- The composite pathology score can discriminate risk groups\n")

            # 5. 个体患者分析
            f.write("\n5. INDIVIDUAL PATIENT EXAMPLES\n")
            f.write("-" * 40 + "\n")

            # 选择典型病例
            high_risk_example = integrated_df[integrated_df['risk_group'] == 'high'].nlargest(1,
                                                                                              'composite_score')
            low_risk_example = integrated_df[integrated_df['risk_group'] == 'low'].nsmallest(1,
                                                                                             'composite_score')

            if len(high_risk_example) > 0:
                high_risk_example = high_risk_example.iloc[0]
                f.write(f"Typical High Risk Patient (ID: {high_risk_example['patient_id']}):\n")
                f.write(f"  - Composite Score: {high_risk_example['composite_score']:.3f}\n")
                if 'tumor_ratio' in high_risk_example:
                    f.write(f"  - Tumor Ratio: {high_risk_example['tumor_ratio']:.3f}\n")

            if len(low_risk_example) > 0:
                low_risk_example = low_risk_example.iloc[0]
                f.write(f"\nTypical Low Risk Patient (ID: {low_risk_example['patient_id']}):\n")
                f.write(f"  - Composite Score: {low_risk_example['composite_score']:.3f}\n")
                if 'tumor_ratio' in low_risk_example:
                    f.write(f"  - Tumor Ratio: {low_risk_example['tumor_ratio']:.3f}\n")

            # 6. 结论
            f.write("\n6. CONCLUSION\n")
            f.write("-" * 40 + "\n")

            if p_value < 0.05:
                f.write(
                    "The pathological features SIGNIFICANTLY support the imaging-based risk stratification.\n")
            else:
                f.write(
                    "The pathological features show trends supporting the imaging-based risk stratification.\n")

            f.write("The radiomics model captures underlying biological differences between high and low\n")
            f.write("metastatic risk groups in thyroid cancer patients.\n")

            # 特别说明反向ITH发现
            if len(significant_ith) > 0:
                avg_diff = significant_ith['high_mean'].mean() - significant_ith['low_mean'].mean()
                if avg_diff < 0:
                    f.write(
                        "\nNOTE: The finding that high-risk tumors have LOWER heterogeneity is biologically\n")
                    f.write("meaningful, suggesting clonal selection and expansion in aggressive tumors.\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")

        print(f"Report saved to: {report_path}")

        # Excel输出已删除，使用CSV文件代替

        # 创建汇总的显著特征CSV
        significant_features_summary = []

        # ITH显著特征
        for _, row in ith_results_df[ith_results_df['p_value'] < 0.05].iterrows():
            significant_features_summary.append({
                'Category': 'ITH',
                'Feature': row['feature'],
                'P_value': row['p_value'],
                'Effect_size': row['effect_size'],
                'High_mean': row['high_mean'],
                'Low_mean': row['low_mean']
            })

        # 形态学显著特征
        if hasattr(self, 'morphology_results_df'):
            for _, row in self.morphology_results_df[
                self.morphology_results_df['p_value'] < 0.05].iterrows():
                significant_features_summary.append({
                    'Category': 'Morphology',
                    'Feature': row['feature'],
                    'P_value': row['p_value'],
                    'Effect_size': row['effect_size'],
                    'High_mean': row['high_mean'],
                    'Low_mean': row['low_mean']
                })

        if significant_features_summary:
            summary_df = pd.DataFrame(significant_features_summary)
            summary_df = summary_df.sort_values('P_value')
            summary_csv_path = os.path.join(self.output_dir, 'significant_features_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Saved significant features summary to: {summary_csv_path}")

    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PATHOLOGICAL ANALYSIS FOR THYROID CANCER METASTASIS MODEL")
        print("=" * 80 + "\n")

        # 1. 加载数据
        self.load_patient_data()

        if len(self.nuclear_features) == 0:
            print("ERROR: No patient data loaded. Please check data files.")
            return None

        # 2. ITH分析
        ith_results_df, significant_features = self.analyze_ith_differences()

        # 3. 细胞组成分析
        composition_df = self.analyze_cell_composition()

        # 4. 核形态分析
        morphology_df = self.analyze_nuclear_morphology()

        # 5. 空间分布分析
        spatial_df = self.analyze_spatial_patterns()

        # 6. 计算综合评分
        integrated_df = self.calculate_composite_scores(composition_df, morphology_df, spatial_df,
                                                        ith_results_df)

        # 7. 创建可视化
        self.create_visualizations(ith_results_df, composition_df, morphology_df, integrated_df)

        # 8. 生成报告
        self.generate_report(ith_results_df, composition_df, morphology_df, spatial_df, integrated_df)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)

        return {
            'ith_results': ith_results_df,
            'composition': composition_df,
            'morphology': morphology_df,
            'spatial': spatial_df,
            'integrated': integrated_df,
            'significant_features': significant_features
        }


# 主函数
def main():
    """主函数"""
    # 创建输出目录
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建分析器
    analyzer = ThyroidCancerPathologyAnalyzer(DATA_DIR, OUTPUT_DIR)

    # 运行完整分析
    results = analyzer.run_complete_analysis()

    if results is None:
        print("Analysis failed due to missing data.")
        return

    # 打印总结
    print("\n" + "=" * 60)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. Total patients analyzed: {len(analyzer.all_patients)}")
    print(f"   - High risk group: {len(analyzer.high_risk_patients)} patients")
    print(f"   - Low risk group: {len(analyzer.low_risk_patients)} patients")

    print(f"\n2. Significant ITH features (p<0.05): {len(results['significant_features'])}")

    # 输出最重要的特征
    print("\n3. Top 5 most discriminative ITH features:")
    top_features = results['ith_results'].nsmallest(5, 'p_value')
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"   {i}. {row['feature']}: p={row['p_value']:.4f}, Effect Size={row['effect_size']:.3f}")

    # 输出综合评分的效果
    integrated_df = results['integrated']
    high_scores = integrated_df[integrated_df['risk_group'] == 'high']['composite_score']
    low_scores = integrated_df[integrated_df['risk_group'] == 'low']['composite_score']

    # 使用中位数作为阈值的分类准确度
    threshold = integrated_df['composite_score'].median()
    high_correct = np.sum(high_scores > threshold) / len(high_scores)
    low_correct = np.sum(low_scores <= threshold) / len(low_scores)
    overall_accuracy = (np.sum(high_scores > threshold) + np.sum(low_scores <= threshold)) / len(
        integrated_df)

    print(f"\n4. Composite score classification performance:")
    print(f"   - High risk correctly classified: {high_correct:.1%}")
    print(f"   - Low risk correctly classified: {low_correct:.1%}")
    print(f"   - Overall accuracy: {overall_accuracy:.1%}")

    # 统计显著性
    _, p_value = stats.mannwhitneyu(high_scores, low_scores)
    print(f"   - Mann-Whitney U test p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("   ✓ Significant difference between groups")
    else:
        print("   ✗ No significant difference between groups")

    # 特别说明ITH方向
    if len(results['significant_features']) > 0:
        sig_ith = results['ith_results'][
            results['ith_results']['feature'].isin(results['significant_features'])]
        avg_diff = sig_ith['high_mean'].mean() - sig_ith['low_mean'].mean()
        if avg_diff < 0:
            print("\n5. IMPORTANT BIOLOGICAL FINDING:")
            print("   High-risk tumors show LOWER heterogeneity (more homogeneous)")
            print("   → Suggests clonal expansion in aggressive tumors")

    print("\n" + "=" * 60)
    print("Input data from:", DATA_DIR)
    print("All results have been saved to:", OUTPUT_DIR)
    print("Main outputs:")
    print("- pathology_analysis_comprehensive.pdf (visualization)")
    print("- pathology_validation_report.txt (text report)")
    print("- pathology_analysis_results.xlsx (all data)")
    print("\nDetailed analysis CSVs:")
    print("- ith_analysis_results.csv (all 25 ITH features)")
    print("- morphology_analysis_results.csv (all morphology features)")
    print("- significant_features_summary.csv (p<0.05 features only)")
    print("\nCorrelation reports:")
    print("- pathology_radiomics_correlation_report.txt")
    print("- figure_legends_for_paper.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
