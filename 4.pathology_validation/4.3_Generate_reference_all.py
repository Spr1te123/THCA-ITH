import os
import numpy as np
import cv2
from tiatoolbox.wsicore.wsireader import WSIReader
import glob
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ImprovedReferenceGenerator:
    """改进的参考图像生成器"""

    def __init__(self, tile_size=1024, tiles_per_wsi=3, use_all_wsi=True):
        self.tile_size = tile_size
        self.tiles_per_wsi = tiles_per_wsi
        self.use_all_wsi = use_all_wsi  # 新增参数：是否使用所有WSI

    def extract_quality_tiles(self, wsi_path, n_tiles=3):
        """提取高质量tiles - 更严格的标准"""
        tiles = []

        try:
            wsi = WSIReader.open(wsi_path)
            wsi_dims = wsi.info.slide_dimensions

            logger.info(f"Processing {os.path.basename(wsi_path)}")

            # 获取缩略图
            try:
                thumbnail = wsi.slide_thumbnail()
            except:
                # 备用方法
                scale = max(wsi_dims[0], wsi_dims[1]) // 2000
                thumb_size = (wsi_dims[0] // scale, wsi_dims[1] // scale)
                thumbnail = wsi.read_region((0, 0), resolution=scale, size=thumb_size)

            if hasattr(thumbnail, 'convert'):
                thumbnail = np.array(thumbnail.convert('RGB'))

            # 使用更智能的组织检测
            gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)

            # 多阈值策略检测组织
            # 1. Otsu阈值
            otsu_thresh, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 2. 自适应阈值
            binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 51, 10)

            # 3. 组合两种方法
            tissue_mask = cv2.bitwise_or(binary_otsu, binary_adaptive)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

            # 找到组织区域
            contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 按面积排序，选择最大的几个区域
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            if not contours:
                logger.warning(f"No tissue found in {os.path.basename(wsi_path)}")
                return tiles

            # 计算缩放因子
            scale_x = wsi_dims[0] / thumbnail.shape[1]
            scale_y = wsi_dims[1] / thumbnail.shape[0]

            # 在每个大轮廓中采样
            sampled_count = 0
            max_attempts_per_contour = 20

            for contour in contours:
                if sampled_count >= n_tiles:
                    break

                x, y, w, h = cv2.boundingRect(contour)

                # 只处理足够大的区域
                if w < 50 or h < 50:  # 缩略图坐标
                    continue

                for attempt in range(max_attempts_per_contour):
                    if sampled_count >= n_tiles:
                        break

                    # 在轮廓内随机采样
                    rand_x = np.random.randint(x + 20, max(x + 21, x + w - 20))
                    rand_y = np.random.randint(y + 20, max(y + 21, y + h - 20))

                    # 检查点是否在轮廓内
                    if cv2.pointPolygonTest(contour, (rand_x, rand_y), False) < 0:
                        continue

                    # 转换到全分辨率坐标
                    wsi_x = int(rand_x * scale_x - self.tile_size // 2)
                    wsi_y = int(rand_y * scale_y - self.tile_size // 2)

                    # 确保坐标有效
                    wsi_x = max(0, min(wsi_x, wsi_dims[0] - self.tile_size))
                    wsi_y = max(0, min(wsi_y, wsi_dims[1] - self.tile_size))

                    # 读取tile
                    try:
                        tile = wsi.read_region(
                            location=(wsi_x, wsi_y),
                            level=0,
                            size=(self.tile_size, self.tile_size)
                        )

                        if hasattr(tile, 'convert'):
                            tile = np.array(tile.convert('RGB'))

                        # 严格的质量检查
                        if self.check_tile_quality_strict(tile):
                            tiles.append({
                                'image': tile,
                                'source': os.path.basename(wsi_path),
                                'location': (wsi_x, wsi_y),
                                'quality_score': self.calculate_quality_score(tile)
                            })
                            sampled_count += 1
                            logger.info(f"  Added tile {sampled_count}/{n_tiles}")

                    except Exception as e:
                        logger.debug(f"Failed to read tile: {e}")

            logger.info(f"Extracted {len(tiles)} quality tiles from {os.path.basename(wsi_path)}")

        except Exception as e:
            logger.error(f"Error processing {wsi_path}: {e}")

        return tiles

    def check_tile_quality_strict(self, tile):
        """严格的tile质量检查"""
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

        # 1. 亮度检查
        mean_intensity = np.mean(gray)
        if mean_intensity < 50 or mean_intensity > 220:
            return False

        # 2. 组织含量检查（更严格）
        # 使用多个阈值
        tissue_mask1 = gray < 200  # 宽松阈值
        tissue_mask2 = gray < 180  # 严格阈值

        tissue_ratio1 = np.sum(tissue_mask1) / tissue_mask1.size
        tissue_ratio2 = np.sum(tissue_mask2) / tissue_mask2.size

        # 要求至少60%的区域是组织（宽松阈值）
        # 且至少40%的区域是组织（严格阈值）
        if tissue_ratio1 < 0.6 or tissue_ratio2 < 0.4:
            return False

        # 3. 对比度和纹理检查
        std_intensity = np.std(gray)
        if std_intensity < 20:  # 提高标准差要求
            return False

        # 4. 边缘检测 - 确保有足够的结构
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        if edge_ratio < 0.02:  # 至少2%的边缘
            return False

        # 5. 颜色检查 - 确保是正常的HE染色
        # 检查红色和蓝色通道
        red_channel = tile[:, :, 0]
        blue_channel = tile[:, :, 2]

        # HE染色应该同时有红色和蓝色成分
        red_mean = np.mean(red_channel[tissue_mask1])
        blue_mean = np.mean(blue_channel[tissue_mask1])

        if red_mean < 100 or blue_mean < 100:  # 颜色太暗
            return False

        # 6. 检查是否过度染色
        oversaturated = np.sum(gray > 240) / gray.size
        if oversaturated > 0.3:  # 超过30%是白色
            return False

        return True

    def calculate_quality_score(self, tile):
        """计算tile的质量分数"""
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

        # 多个质量指标
        tissue_mask = gray < 200
        tissue_ratio = np.sum(tissue_mask) / tissue_mask.size

        std_dev = np.std(gray[tissue_mask]) if np.any(tissue_mask) else 0

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 颜色平衡
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1][tissue_mask]) if np.any(tissue_mask) else 0

        # 综合评分
        score = (
                tissue_ratio * 0.4 +
                min(std_dev / 50, 1.0) * 0.2 +
                min(edge_density * 50, 1.0) * 0.2 +
                min(saturation / 100, 1.0) * 0.2
        )

        return score

    def select_best_tiles(self, all_tiles, n_final=16):
        """选择最好的tiles"""
        if len(all_tiles) <= n_final:
            return all_tiles

        # 按质量分数排序
        sorted_tiles = sorted(all_tiles, key=lambda x: x['quality_score'], reverse=True)

        # 选择质量最好的前70%
        high_quality_tiles = sorted_tiles[:int(len(sorted_tiles) * 0.7)]

        # 从高质量tiles中选择多样化的
        return self.select_diverse_from_quality(high_quality_tiles, n_final)

    def select_diverse_from_quality(self, tiles, n_final):
        """从高质量tiles中选择多样化的样本"""
        if len(tiles) <= n_final:
            return tiles

        # 提取特征
        features = []
        for tile_info in tiles:
            tile = tile_info['image']

            # 颜色直方图特征
            hist_r = cv2.calcHist([tile], [0], None, [32], [0, 256]).flatten()
            hist_g = cv2.calcHist([tile], [1], None, [32], [0, 256]).flatten()
            hist_b = cv2.calcHist([tile], [2], None, [32], [0, 256]).flatten()

            # 归一化
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)

            feature = np.concatenate([hist_r, hist_g, hist_b])
            features.append(feature)

        features = np.array(features)

        # K-means聚类
        n_clusters = min(n_final, len(tiles))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # 从每个聚类选择质量最好的
        selected_tiles = []
        for i in range(n_clusters):
            cluster_tiles = [tiles[j] for j in range(len(tiles)) if labels[j] == i]
            if cluster_tiles:
                # 选择该聚类中质量最好的
                best_tile = max(cluster_tiles, key=lambda x: x['quality_score'])
                selected_tiles.append(best_tile)

        return selected_tiles


def generate_improved_reference(wsi_dir, output_dir, use_all_wsi=True, max_wsi=None):
    """生成改进的参考图像"""

    os.makedirs(output_dir, exist_ok=True)

    # 创建生成器
    generator = ImprovedReferenceGenerator(
        tile_size=1024,
        tiles_per_wsi=3,  # 每个WSI提取3个tiles
        use_all_wsi=use_all_wsi
    )

    # 获取WSI文件
    wsi_patterns = ['*.svs', '*.tif', '*.tiff', '*.ndpi']
    wsi_files = []
    for pattern in wsi_patterns:
        wsi_files.extend(glob.glob(os.path.join(wsi_dir, pattern)))

    logger.info(f"Found {len(wsi_files)} WSI files")

    # 决定使用多少个WSI
    if use_all_wsi:
        wsi_to_process = wsi_files
        logger.info(f"Using ALL {len(wsi_to_process)} WSI files")
    else:
        # 如果不使用所有，则使用指定数量
        num_to_use = max_wsi if max_wsi else 20
        wsi_to_process = wsi_files[:num_to_use]
        logger.info(f"Using {len(wsi_to_process)} WSI files (out of {len(wsi_files)})")

    # 从每个WSI提取高质量tiles
    all_tiles = []

    # 使用进度条
    for wsi_path in tqdm(wsi_to_process, desc="Processing WSIs"):
        tiles = generator.extract_quality_tiles(wsi_path, n_tiles=generator.tiles_per_wsi)
        all_tiles.extend(tiles)

    logger.info(f"Total tiles collected: {len(all_tiles)}")

    if len(all_tiles) == 0:
        logger.error("No quality tiles found in any WSI!")
        return None

    # 选择最好的16个tiles
    selected_tiles = generator.select_best_tiles(all_tiles, n_final=16)

    # 创建4x4网格
    reference_image = np.ones((4 * 1024, 4 * 1024, 3), dtype=np.uint8) * 255

    for idx, tile_info in enumerate(selected_tiles[:16]):
        row = idx // 4
        col = idx % 4
        y_start = row * 1024
        x_start = col * 1024
        reference_image[y_start:y_start + 1024, x_start:x_start + 1024] = tile_info['image']

    # 保存参考图像
    reference_path = os.path.join(output_dir, 'reference_all_wsi.png')
    cv2.imwrite(reference_path, cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR))

    # 也保存为不同标准化方法的版本
    cv2.imwrite(os.path.join(output_dir, 'reference_macenko.png'),
                cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'reference_reinhard.png'),
                cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR))

    # 保存详细的质量报告
    with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
        f.write("Reference Image Generation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total WSI files processed: {len(wsi_to_process)}\n")
        f.write(f"Total tiles extracted: {len(all_tiles)}\n")
        f.write(f"Tiles selected for reference: {len(selected_tiles)}\n\n")

        f.write("Selected Tiles Details:\n")
        f.write("-" * 50 + "\n")
        for i, tile in enumerate(selected_tiles[:16]):
            f.write(f"Tile {i + 1:2d}: {tile['source']:<30} Score: {tile['quality_score']:.3f}\n")

        # 统计每个WSI贡献的tiles
        f.write("\nTiles per WSI:\n")
        f.write("-" * 50 + "\n")
        wsi_counts = {}
        for tile in selected_tiles[:16]:
            source = tile['source']
            wsi_counts[source] = wsi_counts.get(source, 0) + 1

        for source, count in sorted(wsi_counts.items()):
            f.write(f"{source:<30} : {count} tiles\n")

    # 创建可视化报告
    create_visual_report(selected_tiles[:16], output_dir)

    logger.info(f"Reference image saved to: {reference_path}")
    logger.info(f"Quality report saved to: {os.path.join(output_dir, 'quality_report.txt')}")

    return reference_path


def create_visual_report(selected_tiles, output_dir):
    """创建可视化报告显示选中的tiles"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for i in range(16):
        if i < len(selected_tiles):
            axes[i].imshow(selected_tiles[i]['image'])
            axes[i].set_title(f"{selected_tiles[i]['source']}\nScore: {selected_tiles[i]['quality_score']:.3f}",
                              fontsize=8)
        axes[i].axis('off')

    plt.suptitle('Selected Reference Tiles', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selected_tiles_visualization.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate reference image from WSI files')
    parser.add_argument('--wsi-dir', default='/data/wsi',
                        help='Directory containing WSI files')
    parser.add_argument('--output-dir', default='/results/reference_images_all',
                        help='Output directory')
    parser.add_argument('--use-all', action='store_true', default=True,
                        help='Use all WSI files (default: True)')
    parser.add_argument('--max-wsi', type=int, default=None,
                        help='Maximum number of WSI to use (only if --use-all is False)')

    args = parser.parse_args()

    # 生成参考图像
    reference_path = generate_improved_reference(
        args.wsi_dir,
        args.output_dir,
        use_all_wsi=args.use_all,
        max_wsi=args.max_wsi
    )

    if reference_path:
        print(f"\n{'=' * 60}")
        print("Reference image generation completed!")
        print(f"Reference image: {reference_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"{'=' * 60}")
    else:
        print("Failed to generate reference image!")
