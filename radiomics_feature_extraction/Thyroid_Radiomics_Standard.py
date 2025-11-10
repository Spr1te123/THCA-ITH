import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
import time
import logging
import shutil
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')
logging.getLogger('radiomics').setLevel(logging.ERROR)

warnings.filterwarnings('ignore')

# --- 输出文件夹设置 ---
main_output_folder = "/data/Habitat_radiomics/traditional_radiomics_output_zlyybu"
radiomics_features_folder = os.path.join(main_output_folder, "1_Radiomics_Features")
summary_folder = os.path.join(main_output_folder, "2_Summary_Stats")
visualization_folder = os.path.join(main_output_folder, "3_Visualizations")
code_folder = os.path.join(main_output_folder, "4_Code_And_Settings")
qc_folder = os.path.join(main_output_folder, "5_Quality_Control")

for folder in [main_output_folder, radiomics_features_folder, summary_folder,
               visualization_folder, code_folder, qc_folder]:
    os.makedirs(folder, exist_ok=True)

# --- 参数设置 ---
PARAMS = {
    # Pyradiomics settings
    'binCount': 32,
    'interpolator': 'sitkBSpline',
    'resamplePixelSpacing': None,  # 设置为 [1, 1, 1] 可以重采样到1mm各向同性
    'normalize': True,
    'normalizeScale': 100,
    'geometryTolerance': 1e-5,
    'correctMask': True,
    'verbose': False,

    # 可视化设置
    'SAVE_VISUALIZATION': True,
    'VISUALIZATION_PATIENT_LIST': 'all',  # 'all' 或者特定患者ID列表 ['1', '2', '3']

    # 多进程设置
    'NUM_PROCESSES': max(1, cpu_count() - 2),

    # 质量控制参数
    'ENABLE_QUALITY_CONTROL': True,
    'MIN_SNR': 5.0,
    'MIN_CONTRAST': 10.0,
    'MIN_TUMOR_VOLUME_MM3': 50.0,

    # 保存选项
    'SAVE_RESULTS_IMMEDIATELY': False,  # 设置为False，在最后统一保存（避免并发问题）
}

# --- 特征提取器设置 ---
ENABLED_FEATURE_CLASSES = [
    'shape',  # 形状特征
    'firstorder',  # 一阶统计特征
    'glcm',  # 灰度共生矩阵
    'glrlm',  # 灰度游程矩阵
    'glszm',  # 灰度大小区域矩阵
    'gldm',  # 灰度依赖矩阵
    'ngtdm'  # 邻域灰度差矩阵
]

# 如果需要，可以启用特定的图像滤波器
ENABLED_IMAGE_TYPES = {
    'Original': {},  # 原始图像
    # 'LoG': {'sigma': [1.0, 2.0, 3.0]},  # 高斯拉普拉斯滤波
    # 'Wavelet': {},  # 小波变换
}


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
            background_values = []

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

        # 4. 强度分布检查
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
            qc_results['Recommendation'] = "High quality - proceed with analysis"
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


def save_patient_results_immediately(patient_id, patient_features, output_file):
    """分析完一个患者后立即保存结果"""
    import threading

    # 使用线程锁来避免并发写入问题
    if not hasattr(save_patient_results_immediately, 'lock'):
        save_patient_results_immediately.lock = threading.Lock()

    try:
        # 确保PatientID存在且有效
        if 'PatientID' not in patient_features or patient_features['PatientID'] is None:
            logging.error(f"患者 {patient_id} 的特征字典中缺少有效的PatientID")
            return

        # 准备数据
        features_df_new = pd.DataFrame([patient_features])

        # 使用线程锁来避免并发写入问题
        with save_patient_results_immediately.lock:
            # 保存或追加到文件
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # 检查是否已存在该患者（避免重复）
                if str(patient_id) not in existing_df['PatientID'].astype(str).values:
                    combined_df = pd.concat([existing_df, features_df_new], ignore_index=True)
                    combined_df.to_csv(output_file, index=False, na_rep='NaN')
                else:
                    logging.warning(f"患者 {patient_id} 已存在于输出文件中，跳过保存")
            else:
                features_df_new.to_csv(output_file, index=False, na_rep='NaN')

            logging.info(f"患者 {patient_id} 的结果已实时保存")

    except Exception as e:
        logging.error(f"保存患者 {patient_id} 结果时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())


def process_patient(patient_data, base_output_folder, params, enabled_feature_classes, enabled_image_types):
    """处理单个患者的核心函数"""
    patient_id, img_file, mask_file = patient_data

    # 确保patient_id是字符串格式
    patient_id = str(patient_id)

    # 定义输出子文件夹路径
    radiomics_fldr = os.path.join(base_output_folder, "1_Radiomics_Features")
    summary_fldr = os.path.join(base_output_folder, "2_Summary_Stats")
    viz_fldr = os.path.join(base_output_folder, "3_Visualizations")
    qc_fldr = os.path.join(base_output_folder, "5_Quality_Control")

    # 初始化特征字典和状态日志
    patient_features = {'PatientID': patient_id}

    status_log = {
        'PatientID': patient_id,
        'Status': 'Started_Process',
        'Error': None,
        'NumTumorVoxels': 0,
        'TumorVolume_mm3': 0.0,
        'ProcessingTime_sec': 0.0,
        'QC_Passed': None,
        'QC_Score': None,
        'NumFeaturesExtracted': 0
    }

    start_time_patient = time.time()

    # 初始化特征提取器
    extractor_settings = {k: params[k] for k in
                          ['binCount', 'interpolator', 'resamplePixelSpacing', 'normalize', 'normalizeScale',
                           'geometryTolerance', 'correctMask', 'verbose'] if k in params}

    # 如果不需要重采样，移除该参数
    if not extractor_settings.get('resamplePixelSpacing'):
        extractor_settings.pop('resamplePixelSpacing', None)

    try:
        extractor = RadiomicsFeatureExtractor(**extractor_settings)
        extractor.disableAllFeatures()

        # 启用选定的特征类
        for fc in enabled_feature_classes:
            extractor.enableFeatureClassByName(fc)

        # 启用选定的图像类型
        for imageType, settings in enabled_image_types.items():
            if imageType == 'Original':
                continue  # Original is enabled by default
            extractor.enableImageTypeByName(imageType, **settings)

    except Exception as init_e:
        status_log['Status'] = 'Failed_Extractor_Init'
        status_log['Error'] = str(init_e)
        logging.error(f"错误: 患者 {patient_id} 初始化提取器失败: {init_e}")
        return status_log, patient_features

    try:
        # --- 步骤 1: 加载图像和掩膜 ---
        status_log['Status'] = 'Loading Data'
        image_sitk = sitk.ReadImage(img_file)
        mask_sitk_original = sitk.ReadImage(mask_file)

        image_np = sitk.GetArrayFromImage(image_sitk)
        mask_np_original = sitk.GetArrayFromImage(mask_sitk_original)

        if image_np.shape != mask_np_original.shape:
            raise ValueError("图像和掩膜尺寸不匹配")

        # 提取肿瘤掩膜（标签11）
        tumor_mask_np = (mask_np_original == 11).astype(np.uint8)
        tumor_mask_sitk = sitk.GetImageFromArray(tumor_mask_np)
        tumor_mask_sitk.CopyInformation(mask_sitk_original)

        num_tumor_voxels = int(np.sum(tumor_mask_np))
        status_log['NumTumorVoxels'] = num_tumor_voxels

        if num_tumor_voxels == 0:
            raise ValueError("掩膜 (标签11) 不包含肿瘤体素")

        voxel_volume = np.prod(mask_sitk_original.GetSpacing())
        tumor_volume = num_tumor_voxels * voxel_volume
        status_log['TumorVolume_mm3'] = tumor_volume

        # --- 步骤 2: 质量控制 ---
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

        # --- 步骤 3: 提取影像组学特征 ---
        logging.info(f"  患者 {patient_id} 开始提取影像组学特征...")
        status_log['Status'] = 'Extracting Features'

        try:
            # 提取特征
            features = extractor.execute(image_sitk, tumor_mask_sitk, label=1)

            # 过滤和整理特征
            feature_count = 0
            for key, value in features.items():
                # 跳过诊断信息
                if key.startswith('diagnostics_'):
                    continue

                # 检查值的有效性
                value_to_check = value
                if isinstance(value, np.ndarray) and value.size == 1:
                    value_to_check = value.item()

                is_numeric = isinstance(value_to_check, (int, float, np.number))
                is_valid_numeric = is_numeric and np.isfinite(float(value_to_check))

                if is_valid_numeric:
                    # 清理特征名称（移除'original_'前缀）
                    clean_key = key.replace('original_', '')
                    patient_features[clean_key] = float(value_to_check)
                    feature_count += 1

            status_log['NumFeaturesExtracted'] = feature_count
            logging.info(f"  患者 {patient_id} 成功提取 {feature_count} 个特征")

            if feature_count == 0:
                raise ValueError("未能提取任何有效特征")

        except Exception as feat_ex:
            logging.error(f"错误: 患者 {patient_id} 特征提取失败: {feat_ex}\n{traceback.format_exc()}")
            status_log['Status'] = 'Failed_Feature_Extraction'
            status_log['Error'] = str(feat_ex)
            raise

        # --- 步骤 4: 可视化（可选） ---
        patient_id_str = str(patient_id)
        should_visualize = (params['SAVE_VISUALIZATION'] and
                            (params['VISUALIZATION_PATIENT_LIST'] == 'all' or
                             patient_id_str in params.get('VISUALIZATION_PATIENT_LIST', [])))

        if should_visualize:
            try:
                # 找到肿瘤区域最大的轴位切片
                tumor_area_per_slice = np.sum(tumor_mask_np, axis=(1, 2))
                center_slice = np.argmax(tumor_area_per_slice)

                # 创建可视化
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                plt.suptitle(f"Patient {patient_id} - Radiomics Analysis", fontsize=14)

                # 1. 原始图像 + 掩膜轮廓
                axes[0].imshow(image_np[center_slice], cmap='gray')
                axes[0].contour(tumor_mask_np[center_slice], levels=[0.5], colors='red', linewidths=2)
                axes[0].set_title(f'CT Image + Tumor Mask\n(Slice {center_slice}, Volume: {tumor_volume:.1f} mm³)')
                axes[0].axis('off')

                # 2. 特征统计信息
                axes[1].axis('off')

                # 创建特征摘要文本
                feature_text = f"Extracted Features: {feature_count}\n\n"
                feature_text += f"Tumor Volume: {tumor_volume:.1f} mm³\n"
                feature_text += f"Tumor Voxels: {num_tumor_voxels}\n"

                # 添加一些关键特征值（如果存在）
                if 'shape_VoxelVolume' in patient_features:
                    feature_text += f"\nShape Volume: {patient_features['shape_VoxelVolume']:.1f} mm³"
                if 'firstorder_Mean' in patient_features:
                    feature_text += f"\nMean Intensity: {patient_features['firstorder_Mean']:.1f}"
                if 'firstorder_StandardDeviation' in patient_features:
                    feature_text += f"\nStd Deviation: {patient_features['firstorder_StandardDeviation']:.1f}"

                if status_log.get('QC_Score') is not None:
                    feature_text += f"\n\nQC Score: {status_log['QC_Score']}/100"

                axes[1].text(0.1, 0.5, feature_text, transform=axes[1].transAxes,
                             fontsize=12, verticalalignment='center')

                plt.tight_layout()
                viz_filename = os.path.join(viz_fldr, f'patient_{patient_id}_radiomics_summary.png')
                plt.savefig(viz_filename, dpi=150)
                plt.close()

                logging.info(f"  患者 {patient_id} 的可视化已保存")

            except Exception as viz_e:
                logging.warning(f"警告: 患者 {patient_id} 可视化失败: {viz_e}")

        # 标记处理成功
        status_log['Status'] = 'Success'

        # --- 步骤 5: 实时保存结果 ---
        if params.get('SAVE_RESULTS_IMMEDIATELY', True) and status_log['Status'] == 'Success':
            output_file = os.path.join(radiomics_fldr, 'all_patients_radiomics_features.csv')
            save_patient_results_immediately(patient_id, patient_features, output_file)

    except Exception as e:
        logging.error(f"错误: 患者 {patient_id} 处理失败: {e}")
        logging.error(traceback.format_exc())
        if status_log['Status'] == 'Started_Process':
            status_log['Status'] = 'Failed_Processing'
        status_log['Error'] = str(e)

    finally:
        end_time_patient = time.time()
        processing_time = end_time_patient - start_time_patient
        status_log['ProcessingTime_sec'] = round(processing_time, 2)

        # 确保PatientID存在
        if 'PatientID' not in patient_features or patient_features['PatientID'] is None:
            patient_features['PatientID'] = patient_id

        logging.info(f"--- 患者 {patient_id} 处理完成，耗时: {processing_time:.2f} 秒. 状态: {status_log['Status']} ---")

        return status_log, patient_features


# --- 主程序入口 ---
if __name__ == "__main__":
    # 保存参数和代码备份
    params_df = pd.DataFrame.from_dict(PARAMS, orient='index', columns=['Value'])
    params_df.index.name = 'Parameter'
    params_df.to_csv(os.path.join(code_folder, 'script_parameters.csv'))
    logging.info(f"参数设置已保存到: {os.path.join(code_folder, 'script_parameters.csv')}")

    # 保存特征设置
    feature_settings = {
        'Enabled_Feature_Classes': ENABLED_FEATURE_CLASSES,
        'Enabled_Image_Types': list(ENABLED_IMAGE_TYPES.keys())
    }
    with open(os.path.join(code_folder, 'feature_settings.txt'), 'w') as f:
        for key, value in feature_settings.items():
            f.write(f"{key}: {value}\n")

    # 备份当前脚本
    script_path = os.path.abspath(sys.argv[0])
    script_name = os.path.basename(script_path)
    try:
        shutil.copyfile(script_path, os.path.join(code_folder, script_name))
        logging.info(f"当前脚本 '{script_name}' 已备份到: {code_folder}")
    except Exception as copy_e:
        logging.warning(f"警告: 无法备份脚本 '{script_name}': {copy_e}")

    # 准备待处理的文件列表
    SKIP_EXISTING = True
    logging.info(f"\n--- 准备待处理患者列表 (跳过已处理: {SKIP_EXISTING}) ---")

    files_to_process = []
    all_potential_files = []
    processed_count_previous = 0

    radiomics_output_file = os.path.join(radiomics_features_folder, 'all_patients_radiomics_features.csv')
    processed_ids = set()

    if SKIP_EXISTING and os.path.exists(radiomics_output_file):
        try:
            existing_df = pd.read_csv(radiomics_output_file)
            if 'PatientID' in existing_df.columns:
                processed_ids = set(existing_df['PatientID'].astype(str))
                processed_count_previous = len(processed_ids)
                logging.info(f"从 {radiomics_output_file} 加载了 {processed_count_previous} 个已处理的患者ID。")
        except Exception as read_e:
            logging.warning(f"警告: 无法读取现有的特征文件: {read_e}")

    # 输入文件夹路径
    image_folder = "/data/Habitat_radiomics/data_zlyybu"
    mask_folder = "/data/Habitat_radiomics/mask_thyroid_cancer_zlyybu"

    # 文件扫描
    import glob

    # 获取所有图像文件
    image_files = glob.glob(os.path.join(image_folder, "*.nii.gz"))
    logging.info(f"在图像文件夹中找到 {len(image_files)} 个文件")

    # 获取所有掩膜文件
    mask_files = glob.glob(os.path.join(mask_folder, "*.nii.gz"))
    logging.info(f"在掩膜文件夹中找到 {len(mask_files)} 个文件")

    # 创建文件名到完整路径的映射
    image_dict = {os.path.basename(f): f for f in image_files}
    mask_dict = {os.path.basename(f): f for f in mask_files}

    # 找出同时存在图像和掩膜的文件
    common_filenames = set(image_dict.keys()) & set(mask_dict.keys())
    logging.info(f"找到 {len(common_filenames)} 对匹配的图像-掩膜文件")

    # 处理每对匹配的文件
    for filename in sorted(common_filenames):
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
        logging.warning(f"警告: {len(image_only)} 个图像文件没有对应的掩膜")
    if mask_only:
        logging.warning(f"警告: {len(mask_only)} 个掩膜文件没有对应的图像")

    logging.info(f"\n总共找到 {len(all_potential_files)} 个有效患者数据对。")
    if SKIP_EXISTING:
        logging.info(f"之前已处理: {processed_count_previous}")
    logging.info(f"本次需要处理的患者数量: {len(files_to_process)}")

    if not files_to_process:
        logging.info("没有需要处理的患者，程序退出。")
        sys.exit()

    # 使用 multiprocessing Pool
    num_processes = PARAMS['NUM_PROCESSES']
    logging.info(f"\n--- 开始并行处理，使用 {num_processes} 个进程 ---")
    start_time_total = time.time()

    process_func = partial(process_patient,
                           base_output_folder=main_output_folder,
                           params=PARAMS,
                           enabled_feature_classes=ENABLED_FEATURE_CLASSES,
                           enabled_image_types=ENABLED_IMAGE_TYPES)

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

    logging.info("\n--- 所有进程处理完成 ---")

    # 收集和保存结果
    logging.info("\n--- 保存最终聚合结果 ---")
    processing_summary_collected = []
    qc_results_collected = []
    successful_features = []

    if results:
        for result in results:
            if result and isinstance(result, tuple) and len(result) == 2:
                status, features_data = result

                if status and isinstance(status, dict):
                    processing_summary_collected.append(status)

                    # 只收集成功的特征数据
                    if (status.get('Status') == 'Success' and
                            features_data and
                            'PatientID' in features_data and
                            features_data['PatientID'] is not None):
                        successful_features.append(features_data)

                    # 收集质控信息
                    if status.get('QC_Passed') is not None:
                        qc_summary = {
                            'PatientID': status.get('PatientID'),
                            'QC_Passed': status.get('QC_Passed'),
                            'QC_Score': status.get('QC_Score', 0)
                        }
                        qc_results_collected.append(qc_summary)

    # 如果没有使用实时保存，则在这里保存所有成功的结果
    if not PARAMS.get('SAVE_RESULTS_IMMEDIATELY', True) and successful_features:
        all_features_df = pd.DataFrame(successful_features)
        features_output_file = os.path.join(radiomics_features_folder, 'all_patients_radiomics_features.csv')
        all_features_df.to_csv(features_output_file, index=False, na_rep='NaN')
        print(f"所有患者的影像组学特征已保存到: {features_output_file}")

    # 验证和清理最终的特征文件
    radiomics_output_file = os.path.join(radiomics_features_folder, 'all_patients_radiomics_features.csv')
    if os.path.exists(radiomics_output_file):
        try:
            final_df = pd.read_csv(radiomics_output_file)
            # 移除PatientID为NaN的行
            clean_df = final_df[final_df['PatientID'].notna()]
            # 移除重复的患者
            clean_df = clean_df.drop_duplicates(subset=['PatientID'], keep='first')
            # 重新保存清理后的数据
            clean_df.to_csv(radiomics_output_file, index=False, na_rep='NaN')
            print(f"最终特征文件已清理，包含 {len(clean_df)} 个患者的数据")
        except Exception as e:
            logging.error(f"清理最终特征文件时出错: {e}")

    # 保存质控摘要
    if qc_results_collected:
        qc_summary_df = pd.DataFrame(qc_results_collected)
        qc_summary_filename = os.path.join(qc_folder, 'all_patients_qc_summary.csv')
        qc_summary_df.to_csv(qc_summary_filename, index=False, na_rep='NaN')
        print(f"质控摘要已保存到: {qc_summary_filename}")

    # 保存处理摘要
    if processing_summary_collected:
        summary_df = pd.DataFrame(processing_summary_collected)
        summary_filename = os.path.join(summary_folder, 'processing_summary_log.csv')
        summary_df.to_csv(summary_filename, index=False, na_rep='NaN')
        print(f"处理摘要已保存到: {summary_filename}")

    # 结束
    end_time_total = time.time()
    total_minutes = (end_time_total - start_time_total) / 60
    print(f"\n--- 全部处理完成 ---")
    print(f"总处理耗时: {total_minutes:.2f} 分钟")
    print(f"所有输出保存在主文件夹: {main_output_folder}")

    # 最终统计
    if processing_summary_collected:
        total_processed = len(processing_summary_collected)
        success_count = sum(1 for s in processing_summary_collected if s.get('Status') == 'Success')
        failed_count = sum(1 for s in processing_summary_collected if s.get('Status', '').startswith('Failed_'))
        qc_passed_count = sum(1 for s in processing_summary_collected if s.get('QC_Passed', False))

        print(f"\n本次运行处理统计 ({total_processed} 患者):")
        print(f"  成功完成: {success_count}")
        print(f"  处理失败: {failed_count}")
        print(f"  通过质量控制: {qc_passed_count}")

        # 特征数统计
        feature_counts = [s.get('NumFeaturesExtracted', 0) for s in processing_summary_collected if
                          s.get('Status') == 'Success']
        if feature_counts:
            print(f"  平均每个患者提取特征数: {np.mean(feature_counts):.1f}")
