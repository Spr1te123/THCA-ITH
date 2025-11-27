import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.measure import label, regionprops
import warnings
import time
import logging
import pickle
import shutil
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
import math
from scipy.ndimage import binary_dilation, generate_binary_structure

# --- 配置日志记录 ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')
logging.getLogger('radiomics').setLevel(logging.ERROR)

warnings.filterwarnings('ignore')

# --- 输出文件夹设置 ---
main_output_folder = "/results/Habitat_radiomics/habitat_pipeline_3D_ITHscore_output_zlyy"
ited_features_folder = os.path.join(main_output_folder, "1_iTED_Features")
ithscore_folder = os.path.join(main_output_folder, "2_ITHscore_3D")
summary_folder = os.path.join(main_output_folder, "3_Summary_Stats")
subregion_segmentation_folder = os.path.join(main_output_folder, "4_Intermediate_Subregion_Segmentations")
subregion_features_folder = os.path.join(main_output_folder, "5_Intermediate_Subregion_Features")
visualization_folder = os.path.join(main_output_folder, "6_Visualizations")
code_folder = os.path.join(main_output_folder, "7_Code_And_Settings")
qc_folder = os.path.join(main_output_folder, "8_Quality_Control")

for folder in [main_output_folder, ited_features_folder, ithscore_folder, summary_folder,
               subregion_segmentation_folder, subregion_features_folder,
               visualization_folder, code_folder, qc_folder]:
    os.makedirs(folder, exist_ok=True)

# --- 参数设置 ---
PARAMS = {
    # Pyradiomics settings
    'binCount': 32,
    'interpolator': 'sitkBSpline', 'resamplePixelSpacing': None,
    'normalize': True, 'normalizeScale': 100, 'geometryTolerance': 1e-5,
    'correctMask': True, 'verbose': False,
    # SLIC settings
    'MIN_SUBREGION_VOXELS': 10,
    'MIN_DESIRED_SUBREGIONS': 50,
    'TARGET_VOXELS_PER_SUBREGION': 100,
    'slic_compactness': 0.1,
    'slic_sigma': 1,
    # iTED calculation settings
    'MIN_VALID_SUBREGIONS_FOR_ITED': 5, 'MAX_GMM_COMPONENTS_ITED': 4,
    'gmm_covariance_type': 'diag', 'gmm_n_init': 3,
    # *** 3D ITHscore Settings ***
    'ENABLE_3D_ITHSCORE': True,
    'MAX_CLUSTERS_FOR_ITHSCORE': 8,
    'CONNECTIVITY_3D': 26,
    # 修改：使用不同的聚类数选择策略
    'USE_ELBOW_METHOD': True,  # 使用肘部法则而不是轮廓系数
    'MIN_CLUSTERS_FOR_ITHSCORE': 3,  # 最小聚类数
    # Mask Expansion Settings
    'ENABLE_MASK_EXPANSION': True,
    'EXPANSION_DISTANCE_MM': 3.0,
    'MAX_EXPANSION_ATTEMPTS': 3,
    'USE_THYROID_CONSTRAINT': True,
    'ADAPTIVE_EXPANSION': True,
    # File saving options
    'SAVE_SUBREGION_SEGMENTATION': True, 'SAVE_SUBREGION_FEATURES': True,
    'SAVE_VISUALIZATION': True,
    # 新增：可视化患者列表
    'VISUALIZATION_PATIENT_LIST': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
    # Multiprocessing settings
    'NUM_PROCESSES': max(1, cpu_count() - 2),
    # *** 新增：统计和质控参数 ***
    'ENABLE_STATISTICAL_POWER_CHECK': True,
    'MIN_POWER_THRESHOLD': 0.7,
    'ENABLE_CONFIDENCE_INTERVALS': True,
    'N_BOOTSTRAP': 1000,
    'ENABLE_QUALITY_CONTROL': True,
    'MIN_SNR': 5.0,
    'MIN_CONTRAST': 10.0,
    'MIN_TUMOR_VOLUME_MM3': 50.0,
    # *** 新增：SLIC稳定性检查参数 ***
    'ENABLE_SLIC_STABILITY_CHECK': True,
    'SLIC_STABILITY_ITERATIONS': 5,
    'SLIC_STABILITY_DICE_THRESHOLD': 0.7,
    # *** 新增：3D ITHscore置信区间参数 ***
    'ENABLE_3D_ITHSCORE_CI': True,
    'ITHSCORE_BOOTSTRAP_ITERATIONS': 100,
    # 新增：实时保存选项
    'SAVE_RESULTS_IMMEDIATELY': True,  # 分析完每个患者立即保存结果
}

# --- 特征提取器设置 ---
ENABLED_FEATURE_CLASSES = [
    'firstorder', 'glcm', 'glrlm', 'gldm', 'glszm', 'ngtdm'
]


# --- 保持原有的辅助函数（质量控制、统计功效等）---
def perform_quality_control(image_sitk, mask_sitk, patient_id):
    """执行图像质量控制"""
    qc_results = {
        'PatientID': patient_id,
        'QC_DateTime': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        image_np = sitk.GetArrayFromImage(image_sitk)
        mask_np = sitk.GetArrayFromImage(mask_sitk)
        spacing = image_sitk.GetSpacing()

        # 1. 计算信噪比（SNR）
        tumor_values = image_np[mask_np > 0]
        background_mask = (mask_np == 0)

        # 找到远离肿瘤的背景区域
        distance_map = ndimage.distance_transform_edt(~mask_np.astype(bool))
        far_background = background_mask & (distance_map > 20)

        if np.sum(far_background) > 100:
            background_values = image_np[far_background]
            signal = np.mean(tumor_values)
            noise = np.std(background_values)
            snr = signal / noise if noise > 0 else 0
        else:
            snr = 0

        qc_results['SNR'] = snr
        qc_results['SNR_Pass'] = snr >= PARAMS['MIN_SNR']

        # 2. 计算对比度
        if len(background_values) > 0:
            contrast = abs(np.mean(tumor_values) - np.mean(background_values))
        else:
            contrast = 0

        qc_results['Contrast_HU'] = contrast
        qc_results['Contrast_Pass'] = contrast >= PARAMS['MIN_CONTRAST']

        # 3. 肿瘤体积检查
        voxel_volume = np.prod(spacing)
        tumor_volume = np.sum(mask_np > 0) * voxel_volume
        qc_results['TumorVolume_mm3'] = tumor_volume
        qc_results['Volume_Pass'] = tumor_volume >= PARAMS['MIN_TUMOR_VOLUME_MM3']

        # 4. 强度分布检查（检测异常值）
        tumor_std = np.std(tumor_values)
        tumor_iqr = np.percentile(tumor_values, 75) - np.percentile(tumor_values, 25)

        qc_results['Intensity_STD'] = tumor_std
        qc_results['Intensity_IQR'] = tumor_iqr
        qc_results['Intensity_Pass'] = tumor_std > 0 and tumor_iqr > 5

        # 5. 边缘锐利度检查
        edge_mask = ndimage.binary_dilation(mask_np.astype(bool)) & ~mask_np.astype(bool)
        if np.sum(edge_mask) > 0:
            edge_gradient = np.mean(np.abs(np.gradient(image_np)[0])[edge_mask])
        else:
            edge_gradient = 0

        qc_results['EdgeGradient'] = edge_gradient
        qc_results['Edge_Pass'] = edge_gradient > 10

        # 综合判断
        qc_results['Overall_QC_Pass'] = all([
            qc_results['SNR_Pass'],
            qc_results['Contrast_Pass'],
            qc_results['Volume_Pass'],
            qc_results['Intensity_Pass']
        ])

        # 质量评分（0-100）
        quality_score = 0
        if qc_results['SNR_Pass']: quality_score += 25
        if qc_results['Contrast_Pass']: quality_score += 25
        if qc_results['Volume_Pass']: quality_score += 25
        if qc_results['Intensity_Pass']: quality_score += 25

        qc_results['Quality_Score'] = quality_score

        # 生成建议
        if quality_score >= 75:
            qc_results['Recommendation'] = "High quality - proceed with full analysis"
        elif quality_score >= 50:
            qc_results['Recommendation'] = "Medium quality - results should be interpreted with caution"
        else:
            qc_results['Recommendation'] = "Low quality - consider re-scanning or excluding from analysis"

    except Exception as e:
        logging.error(f"质量控制失败 - 患者 {patient_id}: {e}")
        qc_results['QC_Error'] = str(e)
        qc_results['Overall_QC_Pass'] = False
        qc_results['Quality_Score'] = 0

    return qc_results, qc_results.get('Overall_QC_Pass', False)


def check_statistical_power(n_subregions, effect_size=0.5, alpha=0.05):
    """检查统计功效"""
    if n_subregions < 5:
        power = 0.2
        recommendation = "极低功效 - 结果不可靠，建议扩展或放弃分析"
    elif n_subregions < 10:
        power = 0.4
        recommendation = "低功效 - 结果仅供参考"
    elif n_subregions < 20:
        power = 0.6
        recommendation = "中等功效 - 结果需谨慎解释"
    elif n_subregions < 30:
        power = 0.8
        recommendation = "良好功效 - 结果较可靠"
    else:
        power = 0.9
        recommendation = "优秀功效 - 结果可靠"

    return power, recommendation


def calculate_ited_with_confidence_interval(feature_values, feature_name, params, n_bootstrap=None):
    """计算iTED值及其置信区间"""
    if n_bootstrap is None:
        n_bootstrap = params.get('N_BOOTSTRAP', 1000)

    results = {
        'feature_name': feature_name,
        'n_samples': len(feature_values),
        'iTED': 0.0,
        'CI_lower': 0.0,
        'CI_upper': 0.0,
        'CI_width': 0.0,
        'selected_k': 1
    }

    if len(feature_values) < 2:
        return results

    # 原始iTED计算
    X_feature_values = feature_values.reshape(-1, 1)
    best_k = 1
    lowest_bic = np.inf
    best_gmm = None

    for k in range(1, min(params['MAX_GMM_COMPONENTS_ITED'] + 1, len(X_feature_values))):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42,
                                  covariance_type=params['gmm_covariance_type'],
                                  n_init=params['gmm_n_init'])
            gmm.fit(X_feature_values)
            bic_val = gmm.bic(X_feature_values)
            if np.isfinite(bic_val) and bic_val < lowest_bic:
                lowest_bic = bic_val
                best_k = k
                best_gmm = gmm
        except:
            continue

    # 计算原始iTED
    if best_k > 1 and best_gmm is not None:
        means = best_gmm.means_.flatten()
        if len(means) > 1 and np.all(np.isfinite(means)):
            original_ited = np.std(means)
        else:
            original_ited = 0.0
    else:
        original_ited = 0.0

    results['iTED'] = original_ited
    results['selected_k'] = best_k

    # Bootstrap置信区间
    if params.get('ENABLE_CONFIDENCE_INTERVALS', True) and len(feature_values) >= 10:
        bootstrap_iteds = []

        for i in range(n_bootstrap):
            indices = np.random.choice(len(feature_values), size=len(feature_values), replace=True)
            bootstrap_sample = feature_values[indices].reshape(-1, 1)

            if best_k > 1:
                try:
                    gmm_boot = GaussianMixture(n_components=best_k, random_state=42 + i,
                                               covariance_type=params['gmm_covariance_type'],
                                               n_init=1)
                    gmm_boot.fit(bootstrap_sample)
                    means_boot = gmm_boot.means_.flatten()
                    if len(means_boot) > 1 and np.all(np.isfinite(means_boot)):
                        ited_boot = np.std(means_boot)
                    else:
                        ited_boot = 0.0
                except:
                    ited_boot = original_ited
            else:
                ited_boot = 0.0

            bootstrap_iteds.append(ited_boot)

        results['CI_lower'] = np.percentile(bootstrap_iteds, 2.5)
        results['CI_upper'] = np.percentile(bootstrap_iteds, 97.5)
        results['CI_width'] = results['CI_upper'] - results['CI_lower']
        results['bootstrap_std'] = np.std(bootstrap_iteds)

    return results


def assess_slic_stability(image_np, tumor_mask_np, n_segments, params, n_iterations=5):
    """评估SLIC分割的稳定性"""
    try:
        all_segmentations = []
        n_subregions_list = []

        for i in range(n_iterations):
            labels = slic(
                image_np,
                n_segments=n_segments,
                compactness=params['slic_compactness'],
                sigma=params['slic_sigma'],
                start_label=1,
                mask=tumor_mask_np,
                enforce_connectivity=True,
                channel_axis=None
            )
            labels[tumor_mask_np == 0] = 0

            all_segmentations.append(labels)
            valid_labels = np.unique(labels[labels > 0])
            n_subregions_list.append(len(valid_labels))

        dice_scores = []
        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                dice = calculate_segmentation_dice(all_segmentations[i], all_segmentations[j])
                dice_scores.append(dice)

        boundary_uncertainty = np.zeros_like(tumor_mask_np, dtype=float)
        for seg in all_segmentations:
            from scipy import ndimage
            dilated = ndimage.binary_dilation(seg > 0)
            eroded = ndimage.binary_erosion(seg > 0)
            boundaries = dilated & ~eroded
            boundary_uncertainty[boundaries] += 1
        boundary_uncertainty /= n_iterations

        stability_metrics = {
            'mean_dice': np.mean(dice_scores) if dice_scores else 0,
            'std_dice': np.std(dice_scores) if dice_scores else 0,
            'min_dice': np.min(dice_scores) if dice_scores else 0,
            'n_subregions_mean': np.mean(n_subregions_list),
            'n_subregions_std': np.std(n_subregions_list),
            'n_subregions_cv': np.std(n_subregions_list) / (np.mean(n_subregions_list) + 1e-7),
            'boundary_uncertainty_mean': np.mean(boundary_uncertainty[tumor_mask_np > 0]),
            'is_stable': np.mean(dice_scores) > params.get('SLIC_STABILITY_DICE_THRESHOLD',
                                                           0.7) if dice_scores else False
        }

        return stability_metrics

    except Exception as e:
        logging.error(f"SLIC稳定性检查失败: {e}")
        return {
            'mean_dice': -1,
            'is_stable': False,
            'error': str(e)
        }


def calculate_segmentation_dice(seg1, seg2):
    """计算两个分割结果之间的Dice系数"""
    mask1 = seg1 > 0
    mask2 = seg2 > 0

    intersection = np.sum(mask1 & mask2)
    dice = 2.0 * intersection / (np.sum(mask1) + np.sum(mask2))

    return dice


def smart_mask_expansion(tumor_mask_sitk, thyroid_mask_sitk, expansion_distance_mm, max_iterations=10):
    """智能掩膜扩展，确保不超出甲状腺边界"""
    try:
        tumor_mask_np = sitk.GetArrayFromImage(tumor_mask_sitk)
        thyroid_mask_np = sitk.GetArrayFromImage(thyroid_mask_sitk) if thyroid_mask_sitk else None

        spacing = tumor_mask_sitk.GetSpacing()
        expansion_voxels = [int(math.ceil(expansion_distance_mm / s)) for s in spacing]
        original_volume = np.sum(tumor_mask_np)

        if thyroid_mask_np is None:
            dilater = sitk.BinaryDilateImageFilter()
            dilater.SetKernelType(sitk.sitkBall)
            dilater.SetKernelRadius(expansion_voxels[::-1])
            dilater.SetForegroundValue(1)
            expanded_mask_sitk = dilater.Execute(tumor_mask_sitk)

            return expanded_mask_sitk, {
                'method': 'isotropic',
                'original_volume': original_volume,
                'expanded_volume': np.sum(sitk.GetArrayFromImage(expanded_mask_sitk)),
                'constrained': False
            }

        expanded_mask_np = tumor_mask_np.copy()
        current_expansion = [0, 0, 0]
        iteration = 0

        while iteration < max_iterations:
            if all(current_expansion[i] >= expansion_voxels[i] for i in range(3)):
                break

            temp_mask = expanded_mask_np.copy()
            struct_3d = generate_binary_structure(3, 1)

            can_expand = [False, False, False]

            for axis in range(3):
                if current_expansion[axis] < expansion_voxels[axis]:
                    directional_struct = np.zeros((3, 3, 3))
                    directional_struct[1, 1, 1] = 1

                    if axis == 0:
                        directional_struct[0, 1, 1] = 1
                        directional_struct[2, 1, 1] = 1
                    elif axis == 1:
                        directional_struct[1, 0, 1] = 1
                        directional_struct[1, 2, 1] = 1
                    else:
                        directional_struct[1, 1, 0] = 1
                        directional_struct[1, 1, 2] = 1

                    test_expanded = binary_dilation(temp_mask, directional_struct)

                    if np.all(test_expanded[thyroid_mask_np > 0] == test_expanded[thyroid_mask_np > 0]):
                        can_expand[axis] = True

            if not any(can_expand):
                available_space = (thyroid_mask_np > 0) & (expanded_mask_np == 0)
                if np.sum(available_space) > 0:
                    from scipy.ndimage import distance_transform_edt
                    dist_from_tumor = distance_transform_edt(~expanded_mask_np.astype(bool))
                    dist_from_tumor[thyroid_mask_np == 0] = np.inf

                    threshold = np.min(dist_from_tumor[available_space])
                    if threshold < np.inf:
                        new_voxels = (dist_from_tumor <= threshold + 0.5) & available_space
                        expanded_mask_np[new_voxels] = 1
                else:
                    break
            else:
                for axis in range(3):
                    if can_expand[axis] and current_expansion[axis] < expansion_voxels[axis]:
                        directional_struct = np.zeros((3, 3, 3))
                        directional_struct[1, 1, 1] = 1

                        if axis == 0:
                            directional_struct[0, 1, 1] = 1
                            directional_struct[2, 1, 1] = 1
                        elif axis == 1:
                            directional_struct[1, 0, 1] = 1
                            directional_struct[1, 2, 1] = 1
                        else:
                            directional_struct[1, 1, 0] = 1
                            directional_struct[1, 1, 2] = 1

                        expanded_mask_np = binary_dilation(expanded_mask_np, directional_struct)
                        expanded_mask_np = expanded_mask_np & (thyroid_mask_np > 0)
                        current_expansion[axis] += 1

            iteration += 1

        labeled_array, num_features = ndimage.label(expanded_mask_np)
        if num_features > 1:
            sizes = ndimage.sum(expanded_mask_np, labeled_array, range(num_features + 1))
            max_label = np.argmax(sizes)
            expanded_mask_np = (labeled_array == max_label).astype(np.uint8)

        expanded_mask_sitk = sitk.GetImageFromArray(expanded_mask_np.astype(np.uint8))
        expanded_mask_sitk.CopyInformation(tumor_mask_sitk)

        return expanded_mask_sitk, {
            'method': 'smart_constrained',
            'original_volume': original_volume,
            'expanded_volume': np.sum(expanded_mask_np),
            'constrained': True,
            'iterations': iteration,
            'final_expansion_voxels': current_expansion
        }

    except Exception as e:
        logging.error(f"智能掩膜扩展失败: {e}\n{traceback.format_exc()}")
        return tumor_mask_sitk, {
            'method': 'failed',
            'error': str(e),
            'original_volume': np.sum(sitk.GetArrayFromImage(tumor_mask_sitk))
        }


# --- 修改：改进的3D ITHscore计算函数 ---
def calculate_3d_ithscore(subregion_labels_np, subregion_features_df, tumor_mask_np, params):
    """
    基于3D空间分析计算ITHscore（修正版）
    """
    try:
        # 1. 准备特征矩阵用于聚类
        feature_cols = [col for col in subregion_features_df.columns
                        if col not in ['SubregionID', 'SubregionVoxelCount']]

        if not feature_cols:
            return {'3D_ITHscore': 0.0, 'error': 'No features for clustering'}

        # 提取特征并标准化
        X_features = subregion_features_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)

        # 2. 确定最佳聚类数
        min_k = params.get('MIN_CLUSTERS_FOR_ITHSCORE', 3)
        max_k = min(params['MAX_CLUSTERS_FOR_ITHSCORE'], len(X_features) - 1)
        if max_k < min_k:
            max_k = min_k

        if max_k < 2:
            return {'3D_ITHscore': 0.0, 'n_clusters': 1, 'error': 'Too few subregions for clustering'}

        # 使用肘部法则或轮廓系数
        if params.get('USE_ELBOW_METHOD', True):
            # 肘部法则
            inertias = []
            K_range = range(min_k, max_k + 1)

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)

            # 计算肘部点（简化版：选择斜率变化最大的点）
            if len(inertias) > 2:
                # 计算二阶差分
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                # 找到二阶差分最大的点
                elbow_idx = np.argmax(diffs2) + 1  # +1因为diff减少了一个元素
                best_k = K_range[min(elbow_idx, len(K_range) - 1)]
            else:
                best_k = min_k
        else:
            # 使用轮廓系数
            from sklearn.metrics import silhouette_score
            silhouette_scores = []
            K_range = range(min_k, max_k + 1)

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)

            if silhouette_scores:
                best_k = K_range[np.argmax(silhouette_scores)]
            else:
                best_k = min_k

        # 3. 执行最终聚类
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_cluster_labels = final_kmeans.fit_predict(X_scaled)

        # 4. 将聚类标签映射回3D空间
        cluster_map_3d = np.zeros_like(subregion_labels_np, dtype=np.int32) - 1

        for idx, (_, row) in enumerate(subregion_features_df.iterrows()):
            subregion_id = int(row['SubregionID'])
            cluster_label = final_cluster_labels[idx]
            cluster_map_3d[subregion_labels_np == subregion_id] = cluster_label

        # 只考虑肿瘤区域内的体素
        cluster_map_3d[tumor_mask_np == 0] = -1

        # 5. 计算每个聚类的3D连通区域
        total_volume = np.sum(tumor_mask_np)
        ith_components = []

        connectivity_type = params.get('CONNECTIVITY_3D', 26)
        if connectivity_type == 6:
            struct = ndimage.generate_binary_structure(3, 1)  # 6连通
        elif connectivity_type == 18:
            struct = ndimage.generate_binary_structure(3, 2)  # 18连通
        else:
            struct = ndimage.generate_binary_structure(3, 3)  # 26连通

        for cluster_id in range(best_k):
            # 获取当前聚类的二值掩码
            cluster_mask = (cluster_map_3d == cluster_id).astype(np.uint8)

            if np.sum(cluster_mask) == 0:
                continue

            # 3D连通区域分析
            labeled_array, n_regions = ndimage.label(cluster_mask, structure=struct)

            # 计算每个连通区域的体积
            region_volumes = []
            for region_id in range(1, n_regions + 1):
                region_volume = np.sum(labeled_array == region_id)
                region_volumes.append(region_volume)

            if region_volumes:
                max_volume = max(region_volumes)
                total_cluster_volume = sum(region_volumes)

                ith_components.append({
                    'cluster_id': cluster_id,
                    'n_regions': n_regions,
                    'max_volume': max_volume,
                    'total_volume': total_cluster_volume,
                    'mean_region_volume': total_cluster_volume / n_regions
                })

        # 6. 计算3D ITHscore（修正的公式）
        if ith_components:
            # 修正的ITH计算方法
            # ITH = 1 - (最大连通区域占比的加权平均)
            weighted_coherence = 0

            for comp in ith_components:
                # 每个聚类的权重（基于其占总体积的比例）
                cluster_weight = comp['total_volume'] / total_volume
                # 该聚类中最大连通区域的比例
                max_region_ratio = comp['max_volume'] / comp['total_volume']
                # 加权贡献
                weighted_coherence += cluster_weight * max_region_ratio

            # ITHscore: 1减去加权的连通性（连通性越高，异质性越低）
            ith_score_3d = 1 - weighted_coherence
            ith_score_3d = np.clip(ith_score_3d, 0, 1)

            # 计算额外的异质性指标
            # Shannon熵（基于聚类体积分布）
            cluster_volumes = [comp['total_volume'] for comp in ith_components]
            cluster_probs = np.array(cluster_volumes) / np.sum(cluster_volumes)
            shannon_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

            # 空间碎片化指数
            fragmentation_index = np.mean([comp['n_regions'] for comp in ith_components])

        else:
            ith_score_3d = 0.0
            shannon_entropy = 0.0
            fragmentation_index = 0.0

        return {
            '3D_ITHscore': ith_score_3d,
            'n_clusters': best_k,
            'cluster_map': cluster_map_3d,
            'ith_components': ith_components,
            'shannon_entropy': shannon_entropy,
            'fragmentation_index': fragmentation_index,
            'total_volume': total_volume
        }

    except Exception as e:
        logging.error(f"计算3D ITHscore时出错: {e}\n{traceback.format_exc()}")
        return {'3D_ITHscore': 0.0, 'error': str(e)}


def calculate_3d_ithscore_with_ci(subregion_labels_np, subregion_features_df, tumor_mask_np, params):
    """基于3D空间分析计算ITHscore，包含置信区间"""
    try:
        # 首先计算原始的3D ITHscore
        original_results = calculate_3d_ithscore(
            subregion_labels_np, subregion_features_df, tumor_mask_np, params
        )

        # 如果不需要置信区间或样本太少，直接返回
        if not params.get('ENABLE_3D_ITHSCORE_CI', True) or len(subregion_features_df) < 10:
            return original_results

        # Bootstrap计算置信区间
        n_bootstrap = params.get('ITHSCORE_BOOTSTRAP_ITERATIONS', 100)
        bootstrap_scores = []
        bootstrap_n_clusters = []

        feature_cols = [col for col in subregion_features_df.columns
                        if col not in ['SubregionID', 'SubregionVoxelCount']]

        for i in range(n_bootstrap):
            # 对亚区进行重采样
            n_subregions = len(subregion_features_df)
            bootstrap_indices = np.random.choice(n_subregions, size=n_subregions, replace=True)
            bootstrap_features_df = subregion_features_df.iloc[bootstrap_indices].copy()
            bootstrap_features_df.reset_index(drop=True, inplace=True)

            try:
                # 重新计算3D ITHscore
                bootstrap_result = calculate_3d_ithscore(
                    subregion_labels_np,
                    bootstrap_features_df,
                    tumor_mask_np,
                    params
                )

                bootstrap_scores.append(bootstrap_result['3D_ITHscore'])
                bootstrap_n_clusters.append(bootstrap_result['n_clusters'])

            except Exception as e:
                continue

        # 添加置信区间到结果
        if len(bootstrap_scores) > 10:
            original_results['3D_ITHscore_CI_lower'] = np.percentile(bootstrap_scores, 2.5)
            original_results['3D_ITHscore_CI_upper'] = np.percentile(bootstrap_scores, 97.5)
            original_results['3D_ITHscore_CI_width'] = (
                    original_results['3D_ITHscore_CI_upper'] -
                    original_results['3D_ITHscore_CI_lower']
            )
            original_results['3D_ITHscore_std'] = np.std(bootstrap_scores)

            # 聚类数的稳定性
            if bootstrap_n_clusters:
                original_results['n_clusters_most_frequent'] = max(
                    set(bootstrap_n_clusters),
                    key=bootstrap_n_clusters.count
                )
                original_results['n_clusters_stability'] = (
                        bootstrap_n_clusters.count(original_results['n_clusters']) /
                        len(bootstrap_n_clusters)
                )

        return original_results

    except Exception as e:
        logging.error(f"计算3D ITHscore置信区间时出错: {e}\n{traceback.format_exc()}")
        return calculate_3d_ithscore(subregion_labels_np, subregion_features_df, tumor_mask_np, params)


# --- 新增：实时保存结果的函数 ---
def save_patient_results_immediately(patient_id, patient_ited_features, patient_ithscore_results,
                                     ited_output_file, ithscore_output_file,
                                     ited_no_ci_file, ithscore_no_ci_file):
    """分析完一个患者后立即保存结果"""
    try:
        # 准备iTED数据
        ited_df_new = pd.DataFrame([patient_ited_features])

        # 准备无CI版本的iTED数据
        ited_no_ci_data = {k: v for k, v in patient_ited_features.items()
                           if not any(ci_key in k for ci_key in ['_CI_lower', '_CI_upper', '_CI_width'])}
        ited_no_ci_df_new = pd.DataFrame([ited_no_ci_data])

        # 准备ITHscore数据（不包含大型数组）
        ithscore_save_data = {k: v for k, v in patient_ithscore_results.items()
                              if k not in ['cluster_map', 'ith_components']}
        ithscore_df_new = pd.DataFrame([ithscore_save_data])

        # 准备无CI版本的ITHscore数据
        ithscore_no_ci_data = {k: v for k, v in ithscore_save_data.items()
                               if not any(ci_key in k for ci_key in ['_CI_lower', '_CI_upper', '_CI_width', '_std'])}
        ithscore_no_ci_df_new = pd.DataFrame([ithscore_no_ci_data])

        # 保存iTED结果（带CI）
        if os.path.exists(ited_output_file):
            existing_df = pd.read_csv(ited_output_file)
            combined_df = pd.concat([existing_df, ited_df_new], ignore_index=True)
            combined_df.to_csv(ited_output_file, index=False, na_rep='NaN')
        else:
            ited_df_new.to_csv(ited_output_file, index=False, na_rep='NaN')

        # 保存iTED结果（无CI）
        if os.path.exists(ited_no_ci_file):
            existing_df = pd.read_csv(ited_no_ci_file)
            combined_df = pd.concat([existing_df, ited_no_ci_df_new], ignore_index=True)
            combined_df.to_csv(ited_no_ci_file, index=False, na_rep='NaN')
        else:
            ited_no_ci_df_new.to_csv(ited_no_ci_file, index=False, na_rep='NaN')

        # 保存ITHscore结果（带CI）
        if os.path.exists(ithscore_output_file):
            existing_df = pd.read_csv(ithscore_output_file)
            combined_df = pd.concat([existing_df, ithscore_df_new], ignore_index=True)
            combined_df.to_csv(ithscore_output_file, index=False, na_rep='NaN')
        else:
            ithscore_df_new.to_csv(ithscore_output_file, index=False, na_rep='NaN')

        # 保存ITHscore结果（无CI）
        if os.path.exists(ithscore_no_ci_file):
            existing_df = pd.read_csv(ithscore_no_ci_file)
            combined_df = pd.concat([existing_df, ithscore_no_ci_df_new], ignore_index=True)
            combined_df.to_csv(ithscore_no_ci_file, index=False, na_rep='NaN')
        else:
            ithscore_no_ci_df_new.to_csv(ithscore_no_ci_file, index=False, na_rep='NaN')

        logging.info(f"患者 {patient_id} 的结果已实时保存")

    except Exception as e:
        logging.error(f"保存患者 {patient_id} 结果时出错: {e}")


# --- 修改后的患者处理函数 ---
def process_patient(patient_data, base_output_folder, params, enabled_feature_classes):
    """处理单个患者的核心函数"""
    patient_id, img_file, mask_file = patient_data

    # 定义输出子文件夹路径
    ited_fldr = os.path.join(base_output_folder, "1_iTED_Features")
    ithscore_fldr = os.path.join(base_output_folder, "2_ITHscore_3D")
    summary_fldr = os.path.join(base_output_folder, "3_Summary_Stats")
    subreg_seg_fldr = os.path.join(base_output_folder, "4_Intermediate_Subregion_Segmentations")
    subreg_feat_fldr = os.path.join(base_output_folder, "5_Intermediate_Subregion_Features")
    viz_fldr = os.path.join(base_output_folder, "6_Visualizations")
    qc_fldr = os.path.join(base_output_folder, "8_Quality_Control")

    # 初始化状态和特征字典
    patient_ited_features = {'PatientID': patient_id}
    patient_ithscore_results = {'PatientID': patient_id}

    status_log = {
        'PatientID': patient_id, 'Status': 'Started_Process', 'Error': None,
        'NumTumorVoxels': 0, 'TumorVolume_mm3': 0.0, 'OriginalNumTumorVoxels': 0,
        'SlicTargetSegments': 0, 'InitialSubregions': 0,
        'FilteredSmallSubregions': 0, 'ConstantIntensitySubregions': 0,
        'ValidSubregionsForFeatures': 0, 'FeatureExtractionFailCount': 0,
        'ValidSubregionsForITED': 0, 'Selected_K_ITED': None,
        'NumITEDFeaturesCalculated': 0, 'ProcessingTime_sec': 0.0,
        'ExpansionAttemptsMade': 0,
        '3D_ITHscore': None, 'ITHscore_NClusters': None,
        'ThyroidConstraintUsed': False,
        'ExpansionMethod': None,
        'QC_Passed': None,
        'QC_Score': None,
        'Statistical_Power': None,
        'Power_Recommendation': None,
        'SLIC_Stability_Dice': None,
        'SLIC_Stable': None,
        'SLIC_Subregions_CV': None,
        '3D_ITHscore_CI_lower': None,
        '3D_ITHscore_CI_upper': None,
        '3D_ITHscore_CI_width': None,
    }

    start_time_patient = time.time()

    # 初始化特征提取器
    extractor_settings = {k: params[k] for k in
                          ['binCount', 'interpolator', 'resamplePixelSpacing', 'normalize', 'normalizeScale',
                           'geometryTolerance', 'correctMask', 'verbose'] if k in params}
    if 'binWidth' in PARAMS: extractor_settings['binWidth'] = PARAMS['binWidth']
    if not extractor_settings.get('resamplePixelSpacing'): extractor_settings.pop('resamplePixelSpacing', None)

    try:
        extractor = RadiomicsFeatureExtractor(**extractor_settings)
        extractor.disableAllFeatures()
        for fc in enabled_feature_classes:
            extractor.enableFeatureClassByName(fc)
    except Exception as init_e:
        status_log['Status'] = 'Failed_Extractor_Init'
        status_log['Error'] = str(init_e)
        logging.error(f"错误: 患者 {patient_id} 初始化提取器失败: {init_e}")
        return status_log, patient_ited_features, patient_ithscore_results

    try:
        # --- 步骤 0: 加载与检查 ---
        status_log['Status'] = 'Loading Data'
        image_sitk = sitk.ReadImage(img_file)
        mask_sitk_original = sitk.ReadImage(mask_file)
        image_np = sitk.GetArrayFromImage(image_sitk)
        mask_np_original = sitk.GetArrayFromImage(mask_sitk_original)

        if image_np.shape != mask_np_original.shape:
            raise ValueError("图像和掩膜尺寸不匹配")

        # Initial tumor mask extraction
        tumor_mask_np = (mask_np_original == 11).astype(np.uint8)
        tumor_mask_sitk = sitk.GetImageFromArray(tumor_mask_np)
        tumor_mask_sitk.CopyInformation(mask_sitk_original)

        num_tumor_voxels = int(np.sum(tumor_mask_np))
        original_num_tumor_voxels = num_tumor_voxels
        status_log['NumTumorVoxels'] = num_tumor_voxels
        status_log['OriginalNumTumorVoxels'] = original_num_tumor_voxels

        if num_tumor_voxels == 0:
            raise ValueError("掩膜 (标签11) 不包含肿瘤体素")

        voxel_volume = np.prod(mask_sitk_original.GetSpacing())
        tumor_volume = num_tumor_voxels * voxel_volume
        status_log['TumorVolume_mm3'] = tumor_volume

        # --- 质量控制 ---
        if params.get('ENABLE_QUALITY_CONTROL', True):
            logging.info(f"  患者 {patient_id} 执行质量控制...")
            qc_results, qc_passed = perform_quality_control(image_sitk, tumor_mask_sitk, patient_id)
            status_log['QC_Passed'] = qc_passed
            status_log['QC_Score'] = qc_results.get('Quality_Score', 0)

            # 保存质控结果
            qc_filename = os.path.join(qc_fldr, f'patient_{patient_id}_qc_results.csv')
            pd.DataFrame([qc_results]).to_csv(qc_filename, index=False)

            if not qc_passed and qc_results.get('Quality_Score', 0) < 50:
                logging.warning(f"警告: 患者 {patient_id} 质量控制评分低 ({qc_results.get('Quality_Score', 0)}/100)")
                status_log['QC_Warning'] = qc_results.get('Recommendation', 'Low quality data')

        # --- 加载甲状腺掩膜 ---
        thyroid_mask_sitk = None
        if params.get('USE_THYROID_CONSTRAINT', True):
            thyroid_mask_folder = "/data/mask_thyroid_zlyy"
            thyroid_mask_file = os.path.join(thyroid_mask_folder, os.path.basename(mask_file))

            if os.path.exists(thyroid_mask_file):
                try:
                    thyroid_mask_sitk = sitk.ReadImage(thyroid_mask_file)
                    thyroid_mask_np = sitk.GetArrayFromImage(thyroid_mask_sitk)

                    if not np.any(thyroid_mask_np[tumor_mask_np > 0] > 0):
                        logging.warning(f"警告: 患者 {patient_id} 的肿瘤不在甲状腺掩膜内，将不使用甲状腺约束")
                        thyroid_mask_sitk = None
                    else:
                        status_log['ThyroidConstraintUsed'] = True
                        logging.info(f"患者 {patient_id}: 加载甲状腺掩膜成功，将用于约束扩展")
                except Exception as e:
                    logging.warning(f"警告: 患者 {patient_id} 加载甲状腺掩膜失败: {e}")
                    thyroid_mask_sitk = None
            else:
                logging.info(f"患者 {patient_id}: 未找到甲状腺掩膜文件")

        # 扩展循环代码
        expansion_attempts = 0
        max_attempts = params.get('MAX_EXPANSION_ATTEMPTS', 0)
        expansion_enabled = params.get('ENABLE_MASK_EXPANSION', False)
        expansion_distance_mm = params.get('EXPANSION_DISTANCE_MM', 3.0)

        num_valid_subregions_with_features = 0
        subregion_features_list = []
        subregion_labels_np = None

        while True:
            # 扩展代码 - 使用智能扩展函数
            if expansion_enabled and expansion_attempts > 0:
                logging.warning(
                    f"      患者 {patient_id} 尝试第 {expansion_attempts} 次掩膜扩展 ({expansion_distance_mm}mm)...")
                status_log['Status'] = f'Expanding Mask Attempt {expansion_attempts}'

                try:
                    # 使用智能扩展函数
                    if params.get('ADAPTIVE_EXPANSION', True) and thyroid_mask_sitk is not None:
                        expanded_mask_sitk, expansion_info = smart_mask_expansion(
                            tumor_mask_sitk,
                            thyroid_mask_sitk,
                            expansion_distance_mm * expansion_attempts  # 逐步增加扩展距离
                        )
                        status_log['ExpansionMethod'] = expansion_info.get('method', 'unknown')

                        logging.info(f"      患者 {patient_id} 使用 {expansion_info['method']} 方法扩展, "
                                     f"体积从 {expansion_info.get('original_volume', 0)} 增加到 "
                                     f"{expansion_info.get('expanded_volume', 0)}")
                    else:
                        # 使用原始的各向同性扩展
                        spacing = tumor_mask_sitk.GetSpacing()
                        radius_voxels = [int(math.ceil(expansion_distance_mm * expansion_attempts / s)) for s in
                                         spacing]

                        dilater = sitk.BinaryDilateImageFilter()
                        dilater.SetKernelType(sitk.sitkBall)
                        dilater.SetKernelRadius(radius_voxels)
                        dilater.SetForegroundValue(1)

                        expanded_mask_sitk = dilater.Execute(tumor_mask_sitk)
                        status_log['ExpansionMethod'] = 'isotropic'

                    tumor_mask_sitk = expanded_mask_sitk
                    tumor_mask_np = sitk.GetArrayFromImage(tumor_mask_sitk)
                    num_tumor_voxels = int(np.sum(tumor_mask_np))
                    tumor_volume = num_tumor_voxels * voxel_volume

                    if num_tumor_voxels == 0:
                        logging.warning(f"警告: 患者 {patient_id} 扩展后掩膜变空，停止扩展。")
                        status_log['Status'] = 'Failed_Expansion_EmptyMask'
                        status_log['Error'] = f"Mask became empty after {expansion_attempts} expansion(s)."
                        num_valid_subregions_with_features = 0
                        break

                    status_log['NumTumorVoxels'] = num_tumor_voxels
                    status_log['TumorVolume_mm3'] = tumor_volume
                    logging.info(f"      患者 {patient_id} 扩展后肿瘤体素: {num_tumor_voxels}")

                except Exception as dilate_e:
                    logging.error(
                        f"错误: 患者 {patient_id} 第 {expansion_attempts} 次掩膜扩展失败: {dilate_e}\n{traceback.format_exc()}")
                    status_log['Status'] = 'Failed_MaskExpansion'
                    status_log['Error'] = f"Dilation failed on attempt {expansion_attempts}: {dilate_e}"
                    break

            # SLIC分割
            status_log['Status'] = f'Running SLIC (Attempt {expansion_attempts})'
            estimated_n_segments_vol = max(1, int(num_tumor_voxels / params['TARGET_VOXELS_PER_SUBREGION']))
            n_segments_slic = max(params['MIN_DESIRED_SUBREGIONS'], estimated_n_segments_vol)
            status_log['SlicTargetSegments'] = n_segments_slic

            try:
                subregion_labels_np = slic(image_np, n_segments=n_segments_slic,
                                           compactness=params['slic_compactness'],
                                           sigma=params['slic_sigma'],
                                           start_label=1,
                                           mask=tumor_mask_np,
                                           enforce_connectivity=True,
                                           channel_axis=None)
                subregion_labels_np[tumor_mask_np == 0] = 0
            except Exception as slic_e:
                error_msg = f"SLIC 分割失败 (Attempt {expansion_attempts}): {slic_e}\n{traceback.format_exc()}"
                status_log['Status'] = f'Failed_SLIC_Attempt_{expansion_attempts}'
                status_log['Error'] = error_msg
                logging.error(f"错误: 患者 {patient_id} {error_msg}")
                num_valid_subregions_with_features = 0
                break

            unique_subregion_labels = np.unique(subregion_labels_np)
            valid_subregion_labels = unique_subregion_labels[unique_subregion_labels > 0]

            if expansion_attempts == 0:
                status_log['InitialSubregions'] = len(valid_subregion_labels)

            # SLIC稳定性检查（只在第一次未扩展时）
            if params.get('ENABLE_SLIC_STABILITY_CHECK', True) and expansion_attempts == 0:
                logging.info(f"  患者 {patient_id} 执行SLIC稳定性检查...")
                stability_metrics = assess_slic_stability(
                    image_np,
                    tumor_mask_np,
                    n_segments_slic,
                    params,
                    n_iterations=params.get('SLIC_STABILITY_ITERATIONS', 5)
                )

                status_log['SLIC_Stability_Dice'] = stability_metrics.get('mean_dice', -1)
                status_log['SLIC_Stable'] = stability_metrics.get('is_stable', False)
                status_log['SLIC_Subregions_CV'] = stability_metrics.get('n_subregions_cv', -1)

                if not stability_metrics.get('is_stable', False):
                    logging.warning(
                        f"警告: 患者 {patient_id} SLIC分割不稳定 (Dice={stability_metrics.get('mean_dice', -1):.3f})")

            # 特征提取
            status_log['Status'] = f'Extracting Subregion Features (Attempt {expansion_attempts})'
            current_attempt_features_list = []
            subregion_ids_for_patient = []
            feature_extraction_fail_count = 0
            filtered_out_small_count = 0
            constant_intensity_count = 0
            subregion_sizes = []

            for sub_label in valid_subregion_labels:
                current_subregion_mask_np = (subregion_labels_np == sub_label).astype(np.uint8)
                subregion_voxel_count = int(np.sum(current_subregion_mask_np))
                subregion_sizes.append(subregion_voxel_count)

                if subregion_voxel_count < params['MIN_SUBREGION_VOXELS']:
                    filtered_out_small_count += 1
                    continue

                current_subregion_mask_sitk = sitk.GetImageFromArray(current_subregion_mask_np)
                current_subregion_mask_sitk.CopyInformation(image_sitk)

                try:
                    current_subregion_image_np = image_np[current_subregion_mask_np > 0]
                    intensity_std_dev = np.std(current_subregion_image_np)
                    if np.isclose(intensity_std_dev, 0.0, atol=1e-6):
                        constant_intensity_count += 1
                        feature_extraction_fail_count += 1
                        continue

                    features = extractor.execute(image_sitk, current_subregion_mask_sitk, label=1)

                    filtered_features = {}
                    for k, v in features.items():
                        is_diagnostic = k.startswith('diagnostics_')
                        value_to_check = v
                        if isinstance(v, np.ndarray) and v.size == 1:
                            value_to_check = v.item()
                        is_numeric = isinstance(value_to_check, (int, float, np.number))
                        is_valid_numeric = is_numeric and np.isfinite(float(value_to_check))
                        if is_valid_numeric and not is_diagnostic:
                            filtered_features[k.replace('original_', '')] = float(value_to_check)

                    if not filtered_features and features:
                        feature_extraction_fail_count += 1
                        continue
                    elif not features:
                        feature_extraction_fail_count += 1
                        continue

                    filtered_features['SubregionID'] = sub_label
                    filtered_features['SubregionVoxelCount'] = subregion_voxel_count
                    current_attempt_features_list.append(filtered_features)
                    subregion_ids_for_patient.append(sub_label)

                except Exception as feat_ex:
                    logging.error(
                        f"错误: 患者 {patient_id} 提取亚区 {sub_label} (大小: {subregion_voxel_count}, 尝试: {expansion_attempts}) 特征失败: {feat_ex}\n{traceback.format_exc()}")
                    feature_extraction_fail_count += 1

            num_valid_subregions_with_features = len(current_attempt_features_list)
            status_log['FilteredSmallSubregions'] = filtered_out_small_count
            status_log['ConstantIntensitySubregions'] = constant_intensity_count
            status_log['ValidSubregionsForFeatures'] = num_valid_subregions_with_features
            status_log['FeatureExtractionFailCount'] = feature_extraction_fail_count
            status_log['ExpansionAttemptsMade'] = expansion_attempts

            logging.warning(
                f"      患者 {patient_id} 尝试 {expansion_attempts}: 找到 {num_valid_subregions_with_features} 个有效亚区 (阈值 {params['MIN_VALID_SUBREGIONS_FOR_ITED']})。肿瘤体素: {num_tumor_voxels} (原始: {original_num_tumor_voxels})")

            # 统计功效检查
            if params.get('ENABLE_STATISTICAL_POWER_CHECK', True):
                power, recommendation = check_statistical_power(num_valid_subregions_with_features)
                status_log['Statistical_Power'] = power
                status_log['Power_Recommendation'] = recommendation
                logging.info(f"      患者 {patient_id} 统计功效: {power:.2f} - {recommendation}")

            if num_valid_subregions_with_features >= params['MIN_VALID_SUBREGIONS_FOR_ITED']:
                logging.info(
                    f"      患者 {patient_id} 在尝试 {expansion_attempts} 后达到足够亚区 ({num_valid_subregions_with_features})，停止扩展。")
                subregion_features_list = current_attempt_features_list
                break

            expansion_attempts += 1

            if not expansion_enabled or expansion_attempts > max_attempts:
                logging.warning(
                    f"      患者 {patient_id} 达到最大扩展次数 ({max_attempts}) 或扩展未启用，但仍只有 {num_valid_subregions_with_features} 个亚区。")
                subregion_features_list = current_attempt_features_list
                break

        # --- 计算3D ITHscore（包含置信区间） ---
        if (params.get('ENABLE_3D_ITHSCORE', True) and
                num_valid_subregions_with_features >= params['MIN_VALID_SUBREGIONS_FOR_ITED'] and
                subregion_labels_np is not None):

            logging.info(f"  患者 {patient_id} [3D] 计算3D ITHscore及置信区间...")
            status_log['Status'] = 'Calculating 3D ITHscore'

            subregion_features_df = pd.DataFrame(subregion_features_list)

            # 使用新的带置信区间的函数
            ithscore_results = calculate_3d_ithscore_with_ci(
                subregion_labels_np,
                subregion_features_df,
                tumor_mask_np,
                params
            )

            # 更新结果
            patient_ithscore_results.update(ithscore_results)
            status_log['3D_ITHscore'] = ithscore_results.get('3D_ITHscore', 0.0)
            status_log['ITHscore_NClusters'] = ithscore_results.get('n_clusters', 0)

            # 更新置信区间信息
            status_log['3D_ITHscore_CI_lower'] = ithscore_results.get('3D_ITHscore_CI_lower')
            status_log['3D_ITHscore_CI_upper'] = ithscore_results.get('3D_ITHscore_CI_upper')
            status_log['3D_ITHscore_CI_width'] = ithscore_results.get('3D_ITHscore_CI_width')

            # 保存3D聚类结果
            if params['SAVE_SUBREGION_SEGMENTATION'] and 'cluster_map' in ithscore_results:
                cluster_map_sitk = sitk.GetImageFromArray(
                    ithscore_results['cluster_map'].astype(np.int16)
                )
                cluster_map_sitk.CopyInformation(image_sitk)
                cluster_filename = os.path.join(
                    subreg_seg_fldr,
                    f'patient_{patient_id}_3D_clusters.nii.gz'
                )
                sitk.WriteImage(cluster_map_sitk, cluster_filename)

        # iTED计算 - 包含置信区间
        status_log['ValidSubregionsForITED'] = num_valid_subregions_with_features

        if num_valid_subregions_with_features < params['MIN_VALID_SUBREGIONS_FOR_ITED']:
            reason = "有效亚区数"
            if status_log['ExpansionAttemptsMade'] > 0:
                reason = f"在 {status_log['ExpansionAttemptsMade']} 次扩展后，有效亚区数"

            logging.warning(
                f"警告: 患者 {patient_id} {reason} ({num_valid_subregions_with_features}) 少于iTED阈值 ({params['MIN_VALID_SUBREGIONS_FOR_ITED']})，跳过 iTED。")

            if expansion_enabled and status_log['ExpansionAttemptsMade'] > 0:
                status_log['Status'] = 'Skipped_iTED_InsufficientSubregions_PostExpansion'
            elif status_log['Status'].startswith('Failed_'):
                pass
            else:
                status_log['Status'] = 'Skipped_iTED_InsufficientSubregions'

            patient_ited_features['Status'] = status_log['Status']
            raise StopIteration(f"跳过 iTED 计算 ({status_log['Status']})")

        # --- iTED计算包含置信区间 ---
        status_log['Status'] = 'Calculating iTED Features'
        subregion_features_df = pd.DataFrame(subregion_features_list)
        feature_cols_for_ited = [col for col in subregion_features_df.columns if
                                 col not in ['SubregionID', 'SubregionVoxelCount']]
        ited_calculation_count = 0
        last_best_k = 1

        for feature_name in feature_cols_for_ited:
            feature_values = subregion_features_df[feature_name].dropna().values
            if len(feature_values) < 2:
                patient_ited_features[f'iTED_{feature_name}'] = 0.0
                patient_ited_features[f'iTED_{feature_name}_CI_lower'] = 0.0
                patient_ited_features[f'iTED_{feature_name}_CI_upper'] = 0.0
                continue

            # 使用新的置信区间计算函数
            ited_results = calculate_ited_with_confidence_interval(
                feature_values, feature_name, params
            )

            patient_ited_features[f'iTED_{feature_name}'] = ited_results['iTED']
            patient_ited_features[f'iTED_{feature_name}_CI_lower'] = ited_results['CI_lower']
            patient_ited_features[f'iTED_{feature_name}_CI_upper'] = ited_results['CI_upper']
            patient_ited_features[f'iTED_{feature_name}_CI_width'] = ited_results['CI_width']

            ited_calculation_count += 1
            last_best_k = ited_results['selected_k']

        status_log['Selected_K_ITED'] = last_best_k
        status_log['NumITEDFeaturesCalculated'] = ited_calculation_count
        status_log['Status'] = 'Completed_iTED'

        # --- 修改后的可视化（只生成轴位最大截面，只对特定患者） ---
        patient_id_str = str(patient_id)
        should_visualize = patient_id_str in params.get('VISUALIZATION_PATIENT_LIST', [])

        if params['SAVE_VISUALIZATION'] and subregion_labels_np is not None and should_visualize:
            try:
                final_tumor_mask_for_viz = sitk.GetArrayFromImage(tumor_mask_sitk)

                # 找到肿瘤区域最大的轴位切片
                tumor_area_per_slice = np.sum(final_tumor_mask_for_viz, axis=(1, 2))
                center_slice = np.argmax(tumor_area_per_slice)

                # 创建简化的可视化（只有轴位图）
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                expansion_info = f"(Expanded {status_log['ExpansionAttemptsMade']} times)" if status_log[
                                                                                                  'ExpansionAttemptsMade'] > 0 else "(Original Mask)"
                constraint_info = " [Thyroid Constrained]" if status_log['ThyroidConstraintUsed'] else ""
                plt.suptitle(f"Patient {patient_id} - Analysis Results {expansion_info}{constraint_info}", fontsize=14)

                # 1. 原始图像 + 掩膜轮廓
                axes[0].imshow(image_np[center_slice], cmap='gray')
                axes[0].contour(final_tumor_mask_for_viz[center_slice], levels=[0.5], colors='red', linewidths=2)

                # 如果有甲状腺掩膜，显示其轮廓
                if thyroid_mask_sitk is not None:
                    thyroid_mask_for_viz = sitk.GetArrayFromImage(thyroid_mask_sitk)
                    axes[0].contour(thyroid_mask_for_viz[center_slice], levels=[0.5], colors='green', linewidths=1,
                                    linestyles='dashed')

                axes[0].set_title(f'Original CT + Masks\n(Slice {center_slice}, Tumor: {num_tumor_voxels} voxels)')
                axes[0].axis('off')

                # 2. SLIC亚区分割
                axes[1].imshow(image_np[center_slice], cmap='gray')
                num_subregions_on_slice = len(
                    np.unique(subregion_labels_np[center_slice][subregion_labels_np[center_slice] > 0]))
                cmap_name = 'tab20' if num_subregions_on_slice <= 20 else 'gist_rainbow'
                cmap_subregions = plt.cm.get_cmap(cmap_name, max(20, num_subregions_on_slice + 1))
                masked_subregions = np.ma.masked_where(final_tumor_mask_for_viz[center_slice] == 0,
                                                       subregion_labels_np[center_slice])
                axes[1].imshow(masked_subregions, alpha=0.6, cmap=cmap_subregions, interpolation='none')
                axes[1].set_title(f'SLIC Subregions\n({num_valid_subregions_with_features} valid)')
                axes[1].axis('off')

                # 3. 3D聚类结果或ITH信息
                if 'cluster_map' in patient_ithscore_results:
                    cluster_map = patient_ithscore_results['cluster_map']
                    masked_clusters = np.ma.masked_where(cluster_map[center_slice] < 0, cluster_map[center_slice])
                    axes[2].imshow(image_np[center_slice], cmap='gray')
                    axes[2].imshow(masked_clusters, alpha=0.6, cmap='tab10', interpolation='none')
                    axes[2].set_title(
                        f'3D Clusters (k={patient_ithscore_results.get("n_clusters", "?")})\nITHscore: {patient_ithscore_results.get("3D_ITHscore", 0):.3f}')
                else:
                    axes[2].text(0.5, 0.5, f'3D ITHscore: {patient_ithscore_results.get("3D_ITHscore", 0):.4f}\n'
                                           f'n_clusters: {patient_ithscore_results.get("n_clusters", "N/A")}\n'
                                           f'Statistical Power: {status_log.get("Statistical_Power", 0):.2f}',
                                 transform=axes[2].transAxes, fontsize=12, ha='center', va='center')
                axes[2].axis('off')

                plt.tight_layout()
                viz_filename = os.path.join(viz_fldr, f'patient_{patient_id}_axial_analysis.png')
                plt.savefig(viz_filename, dpi=150)
                plt.close()

                logging.info(f"患者 {patient_id} 的可视化已保存")

            except Exception as viz_e:
                logging.warning(f"警告: 患者 {patient_id} 可视化失败: {viz_e}\n{traceback.format_exc()}")
                if status_log['Status'] == 'Completed_iTED':
                    status_log['Status'] = 'Visualization Failed After iTED'

        # 保存处理结果
        if params['SAVE_SUBREGION_SEGMENTATION'] and subregion_labels_np is not None:
            # 保存SLIC分割结果
            final_subregion_mask_sitk = sitk.GetImageFromArray(subregion_labels_np.astype(np.uint16))
            final_subregion_mask_sitk.CopyInformation(image_sitk)
            subreg_filename = os.path.join(subreg_seg_fldr, f'patient_{patient_id}_subregions_final.nii.gz')
            sitk.WriteImage(final_subregion_mask_sitk, subreg_filename)

            # 如果进行了扩展，保存扩展后的肿瘤掩膜
            if status_log['ExpansionAttemptsMade'] > 0:
                expanded_tumor_mask_sitk = sitk.GetImageFromArray(tumor_mask_np.astype(np.uint8))
                expanded_tumor_mask_sitk.CopyInformation(image_sitk)
                expanded_mask_filename = os.path.join(
                    subreg_seg_fldr,
                    f'patient_{patient_id}_expanded_tumor_mask.nii.gz'
                )
                sitk.WriteImage(expanded_tumor_mask_sitk, expanded_mask_filename)
                logging.info(f"患者 {patient_id}: 保存扩展后的肿瘤掩膜 (扩展 {status_log['ExpansionAttemptsMade']} 次)")

        if params['SAVE_SUBREGION_FEATURES'] and subregion_features_list:
            subregion_features_df_tosave = pd.DataFrame(subregion_features_list)
            subreg_feat_filename = os.path.join(subreg_feat_fldr, f'patient_{patient_id}_subregion_features.csv')
            subregion_features_df_tosave.to_csv(subreg_feat_filename, index=False)

        if status_log['Status'] == 'Completed_iTED' or status_log['Status'] == 'Visualization Failed After iTED':
            status_log['Status'] = 'Success'

        # --- 新增：实时保存结果 ---
        if params.get('SAVE_RESULTS_IMMEDIATELY', True) and status_log['Status'] == 'Success':
            ited_output_file = os.path.join(ited_fldr, 'all_patients_iTED_features.csv')
            ithscore_output_file = os.path.join(ithscore_fldr, 'all_patients_3D_ITHscore.csv')
            ited_no_ci_file = os.path.join(ited_fldr, 'all_patients_iTED_features_no_CI.csv')
            ithscore_no_ci_file = os.path.join(ithscore_fldr, 'all_patients_3D_ITHscore_no_CI.csv')

            save_patient_results_immediately(patient_id, patient_ited_features, patient_ithscore_results,
                                             ited_output_file, ithscore_output_file,
                                             ited_no_ci_file, ithscore_no_ci_file)

    except StopIteration as stop_iter:
        logging.info(f"信息: 患者 {patient_id} 处理中止 (跳过 iTED): {stop_iter}")
        if 'PatientID' not in patient_ited_features:
            patient_ited_features['PatientID'] = patient_id
        if 'Status' not in patient_ited_features or not patient_ited_features['Status']:
            patient_ited_features['Status'] = status_log['Status']

    except (ValueError, RuntimeError) as e:
        logging.info(f"信息: 患者 {patient_id} 处理中遇到错误: {e}")
        if status_log['Status'] == 'Started_Process':
            status_log['Status'] = 'Failed_Load_Or_Mask'
        if not status_log['Status'].startswith('Failed_') and not status_log['Status'].startswith('Skipped_'):
            status_log['Status'] = 'Failed_Processing_Error'
        status_log['Error'] = str(e)
        if 'PatientID' not in patient_ited_features:
            patient_ited_features['PatientID'] = patient_id

    except Exception as e:
        logging.error(f"严重错误: 患者 {patient_id} 处理失败: {e}")
        logging.error(traceback.format_exc())
        status_log['Status'] = 'Failed_Unexpected'
        status_log['Error'] = str(e)
        if 'PatientID' not in patient_ited_features:
            patient_ited_features['PatientID'] = patient_id

    finally:
        end_time_patient = time.time()
        processing_time = end_time_patient - start_time_patient
        status_log['ProcessingTime_sec'] = round(processing_time, 2)
        logging.warning(
            f"--- 患者 {patient_id} 处理完成，耗时: {processing_time:.2f} 秒. 最终状态: {status_log['Status']} (扩展尝试: {status_log['ExpansionAttemptsMade']}) ---")

        if 'PatientID' not in status_log:
            status_log['PatientID'] = patient_id
        if 'PatientID' not in patient_ited_features:
            patient_ited_features['PatientID'] = patient_id
        if 'PatientID' not in patient_ithscore_results:
            patient_ithscore_results['PatientID'] = patient_id

        if 'Status' not in patient_ited_features and 'Status' in status_log:
            patient_ited_features['Status'] = status_log['Status']

        return status_log, patient_ited_features, patient_ithscore_results


# --- 主程序入口 ---
if __name__ == "__main__":
    # 保存参数和代码备份
    params_df = pd.DataFrame.from_dict(PARAMS, orient='index', columns=['Value'])
    params_df.index.name = 'Parameter'
    params_df.to_csv(os.path.join(code_folder, 'script_parameters.csv'))
    logging.warning(f"参数设置已保存到: {os.path.join(code_folder, 'script_parameters.csv')}")

    script_path = os.path.abspath(sys.argv[0])
    script_name = os.path.basename(script_path)
    try:
        shutil.copyfile(script_path, os.path.join(code_folder, script_name))
        logging.warning(f"当前脚本 '{script_name}' 已备份到: {code_folder}")
    except Exception as copy_e:
        logging.warning(f"警告: 无法备份脚本 '{script_name}': {copy_e}")

    # 准备待处理的文件列表
    SKIP_EXISTING = True
    logging.warning(f"\n--- 准备待处理患者列表 (跳过已处理: {SKIP_EXISTING}) ---")

    files_to_process = []
    all_potential_files = []
    processed_count_previous = 0

    ited_output_file = os.path.join(ited_features_folder, 'all_patients_iTED_features.csv')
    ithscore_output_file = os.path.join(ithscore_folder, 'all_patients_3D_ITHscore.csv')
    processed_ids = set()

    if SKIP_EXISTING and os.path.exists(ited_output_file):
        try:
            existing_df = pd.read_csv(ited_output_file)
            if 'PatientID' in existing_df.columns:
                processed_ids = set(existing_df['PatientID'].astype(str))
                processed_count_previous = len(processed_ids)
                logging.warning(f"从 {ited_output_file} 加载了 {processed_count_previous} 个已处理的患者ID。")
        except Exception as read_e:
            logging.warning(f"警告: 无法读取现有的iTED特征文件 {ited_output_file} 来检查跳过: {read_e}")

    # 输入文件夹路径
    image_folder = "/data/data_zlyy"
    mask_folder = "/data/mask_thyroid_cancer_zlyy"

    # --- 新的文件扫描逻辑 ---
    import glob

    # 获取所有图像文件
    image_files = glob.glob(os.path.join(image_folder, "*.nii.gz"))
    logging.warning(f"在图像文件夹中找到 {len(image_files)} 个文件")

    # 获取所有掩膜文件
    mask_files = glob.glob(os.path.join(mask_folder, "*.nii.gz"))
    logging.warning(f"在掩膜文件夹中找到 {len(mask_files)} 个文件")

    # 创建文件名到完整路径的映射
    image_dict = {os.path.basename(f): f for f in image_files}
    mask_dict = {os.path.basename(f): f for f in mask_files}

    # 找出同时存在图像和掩膜的文件
    common_filenames = set(image_dict.keys()) & set(mask_dict.keys())
    logging.warning(f"找到 {len(common_filenames)} 对匹配的图像-掩膜文件")

    # 处理每对匹配的文件
    for filename in sorted(common_filenames):  # 排序以保证处理顺序一致
        # 提取患者ID（去掉.nii.gz后缀）
        patient_id = filename.replace('.nii.gz', '')

        # 如果文件名是纯数字，转换为整数；否则保持字符串
        try:
            patient_id_for_processing = int(patient_id)
        except ValueError:
            patient_id_for_processing = patient_id

        img_file = image_dict[filename]
        mask_file = mask_dict[filename]

        all_potential_files.append((patient_id_for_processing, img_file, mask_file))

        # 检查是否已处理
        if SKIP_EXISTING and str(patient_id) in processed_ids:
            continue
        else:
            files_to_process.append((patient_id_for_processing, img_file, mask_file))

    # 报告缺失的文件
    image_only = set(image_dict.keys()) - set(mask_dict.keys())
    mask_only = set(mask_dict.keys()) - set(image_dict.keys())

    if image_only:
        logging.warning(f"警告: {len(image_only)} 个图像文件没有对应的掩膜: {list(image_only)[:5]}...")
    if mask_only:
        logging.warning(f"警告: {len(mask_only)} 个掩膜文件没有对应的图像: {list(mask_only)[:5]}...")

    logging.warning(f"\n总共找到 {len(all_potential_files)} 个有效患者数据对。")
    if SKIP_EXISTING:
        logging.warning(f"之前已处理 (在iTED输出文件中找到): {processed_count_previous}")
    logging.warning(f"本次需要处理的患者数量: {len(files_to_process)}")

    if not files_to_process:
        logging.warning("没有需要处理的患者，程序退出。")
        sys.exit()

    # 使用 multiprocessing Pool
    num_processes = PARAMS['NUM_PROCESSES']
    logging.warning(f"\n--- 开始并行处理，使用 {num_processes} 个进程 ---")
    start_time_total = time.time()

    process_func = partial(process_patient,
                           base_output_folder=main_output_folder,
                           params=PARAMS,
                           enabled_feature_classes=ENABLED_FEATURE_CLASSES)

    try:
        with Pool(processes=num_processes) as pool:
            async_result = pool.map_async(process_func, files_to_process)
            pool.close()
            pool.join()
            results = async_result.get()
    except Exception as pool_e:
        logging.error(f"严重错误: Multiprocessing Pool 遇到错误: {pool_e}")
        logging.error(traceback.format_exc())
        results = []
        sys.exit("并行处理失败。")

    logging.warning("\n--- 所有进程处理完成 ---")

    # 收集和保存结果
    logging.warning("\n--- 保存最终聚合结果 ---")
    processing_summary_collected = []
    qc_results_collected = []
    valid_results_count = 0

    if results:
        for result in results:
            if result and isinstance(result, tuple) and len(result) == 3:
                status, ited_data, ithscore_data = result

                if status and isinstance(status, dict):
                    processing_summary_collected.append(status)

                    # 收集质控信息
                    if status.get('QC_Passed') is not None:
                        qc_summary = {
                            'PatientID': status.get('PatientID'),
                            'QC_Passed': status.get('QC_Passed'),
                            'QC_Score': status.get('QC_Score', 0),
                            'Statistical_Power': status.get('Statistical_Power', 0),
                            'Power_Recommendation': status.get('Power_Recommendation', ''),
                            'SLIC_Stable': status.get('SLIC_Stable'),
                            'SLIC_Stability_Dice': status.get('SLIC_Stability_Dice', -1)
                        }
                        qc_results_collected.append(qc_summary)

                valid_results_count += 1
            else:
                logging.error(f"错误: 从进程池收到无效的结果格式: {result}")

        logging.info(f"从进程池收集到 {valid_results_count} 个有效结果元组。")
    else:
        logging.warning("警告: 未从进程池收到任何结果。")

    # 保存质控摘要
    if qc_results_collected:
        qc_summary_df = pd.DataFrame(qc_results_collected)
        qc_summary_filename = os.path.join(qc_folder, 'all_patients_qc_summary.csv')
        qc_summary_df.to_csv(qc_summary_filename, index=False, na_rep='NaN')
        print(f"质控摘要已保存到: {qc_summary_filename}")

    # 保存处理摘要
    summary_filename = os.path.join(summary_folder, 'processing_summary_log.csv')
    if processing_summary_collected:
        summary_df = pd.DataFrame(processing_summary_collected)
        expected_cols = ['PatientID', 'Status', 'Error', 'NumTumorVoxels', 'TumorVolume_mm3',
                         'OriginalNumTumorVoxels', 'ExpansionAttemptsMade', 'SlicTargetSegments',
                         'InitialSubregions', 'FilteredSmallSubregions', 'ConstantIntensitySubregions',
                         'ValidSubregionsForFeatures', 'FeatureExtractionFailCount',
                         'ValidSubregionsForITED', 'Selected_K_ITED', 'NumITEDFeaturesCalculated',
                         'ProcessingTime_sec', '3D_ITHscore', 'ITHscore_NClusters',
                         'ThyroidConstraintUsed', 'ExpansionMethod',
                         'QC_Passed', 'QC_Score', 'Statistical_Power', 'Power_Recommendation',
                         'SLIC_Stability_Dice', 'SLIC_Stable', 'SLIC_Subregions_CV',
                         '3D_ITHscore_CI_lower', '3D_ITHscore_CI_upper', '3D_ITHscore_CI_width']

        # 确保所有列都存在
        for col in expected_cols:
            if col not in summary_df.columns:
                summary_df[col] = None

        summary_df = summary_df.reindex(columns=expected_cols)

        if SKIP_EXISTING and os.path.exists(summary_filename):
            try:
                summary_df.to_csv(summary_filename, mode='a', header=False, index=False, na_rep='NaN')
                print(f"新的处理摘要已追加到: {summary_filename}")
            except Exception as append_sum_e:
                print(f"错误: 无法追加到现有摘要文件: {append_sum_e}")
                summary_filename_new = os.path.join(summary_folder,
                                                    f'processing_summary_log_new_{time.strftime("%Y%m%d%H%M%S")}.csv')
                summary_df.to_csv(summary_filename_new, index=False, na_rep='NaN')
                print(f"本轮运行的处理摘要已单独保存到: {summary_filename_new}")
        else:
            summary_df.to_csv(summary_filename, index=False, na_rep='NaN')
            print(f"聚合的处理摘要日志已保存到: {summary_filename}")

    # 结束
    end_time_total = time.time()
    total_minutes = (end_time_total - start_time_total) / 60
    print(f"\n--- 全部处理完成 ---")
    print(f"总处理耗时: {total_minutes:.2f} 分钟")
    print(f"所有输出保存在主文件夹: {main_output_folder}")

    # 最终统计
    if processing_summary_collected:
        total_processed_in_run = len(processing_summary_collected)
        success_ited = sum(1 for s in processing_summary_collected if s.get('Status') == 'Success')
        skipped_ited_no_expansion = sum(
            1 for s in processing_summary_collected if s.get('Status') == 'Skipped_iTED_InsufficientSubregions')
        skipped_ited_post_expansion = sum(1 for s in processing_summary_collected if
                                          s.get('Status') == 'Skipped_iTED_InsufficientSubregions_PostExpansion')
        failed_other = sum(1 for s in processing_summary_collected if s.get('Status', '').startswith('Failed_'))
        successful_3d_ithscore = sum(
            1 for s in processing_summary_collected if s.get('3D_ITHscore') is not None and s.get('3D_ITHscore') > 0)
        thyroid_constrained = sum(1 for s in processing_summary_collected if s.get('ThyroidConstraintUsed', False))
        qc_passed_count = sum(1 for s in processing_summary_collected if s.get('QC_Passed', False))
        high_power_count = sum(1 for s in processing_summary_collected if s.get('Statistical_Power', 0) >= 0.8)
        slic_stable_count = sum(1 for s in processing_summary_collected if s.get('SLIC_Stable', False))

        print(f"\n本次运行处理统计 ({total_processed_in_run} 患者):")
        print(f"  成功完成 (含iTED计算): {success_ited}")
        print(f"  成功计算3D ITHscore: {successful_3d_ithscore}")
        print(f"  使用甲状腺边界约束: {thyroid_constrained}")
        print(f"  通过质量控制: {qc_passed_count}")
        print(f"  统计功效≥0.8: {high_power_count}")
        print(f"  SLIC分割稳定: {slic_stable_count}")
        print(f"  跳过 iTED (初始亚区不足): {skipped_ited_no_expansion}")
        print(f"  跳过 iTED (扩展后仍不足): {skipped_ited_post_expansion}")
        print(f"  处理中失败/错误: {failed_other}")
