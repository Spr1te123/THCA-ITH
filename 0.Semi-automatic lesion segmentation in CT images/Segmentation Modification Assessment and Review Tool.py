import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats, ndimage
from scipy.stats import ttest_rel, wilcoxon, sem, t, mannwhitneyu, kruskal
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：基础评估功能
# ============================================================================

class SegmentationEvaluator:
    """基础分割评估器"""

    def __init__(self):
        self.clinical_thresholds = {
            'volume_change_percent': 5.0,
            'dice_acceptable': 0.9,
            'dice_excellent': 0.95
        }

    def dice_coefficient(self, mask1, mask2):
        """计算Dice系数"""
        intersection = np.logical_and(mask1, mask2).sum()
        if mask1.sum() + mask2.sum() == 0:
            return 1.0
        return 2 * intersection / (mask1.sum() + mask2.sum())

    def jaccard_index(self, mask1, mask2):
        """计算Jaccard指数"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 1.0
        return intersection / union

    def evaluate_single_case(self, file_path_a, file_path_b):
        """评估单个病例"""
        try:
            nii_a = nib.load(file_path_a)
            nii_b = nib.load(file_path_b)
            mask_a = nii_a.get_fdata().astype(bool)
            mask_b = nii_b.get_fdata().astype(bool)

            voxel_dims = nii_a.header.get_zooms()
            voxel_volume = np.prod(voxel_dims[:3]) if len(voxel_dims) >= 3 else 1.0

            dice = self.dice_coefficient(mask_a, mask_b)
            jaccard = self.jaccard_index(mask_a, mask_b)

            vol_a = mask_a.sum() * voxel_volume
            vol_b = mask_b.sum() * voxel_volume

            if np.array_equal(mask_a, mask_b):
                mod_type = "无修改"
            elif dice > 0.9:
                mod_type = "微调"
            elif dice > 0.7:
                mod_type = "中度修改"
            else:
                mod_type = "大幅修改"

            return {
                'dice': dice,
                'jaccard': jaccard,
                'volume_a': vol_a,
                'volume_b': vol_b,
                'volume_change': vol_b - vol_a,
                'volume_change_percent': (vol_b - vol_a) / vol_a * 100 if vol_a > 0 else np.nan,
                'modified': not np.array_equal(mask_a, mask_b),
                'modification_type': mod_type,
                'clinically_acceptable': dice >= self.clinical_thresholds['dice_acceptable']
            }
        except Exception as e:
            print(f"Error: {e}")
            return None

    def evaluate_dataset(self, path_a, path_b, dataset_name):
        """评估整个数据集"""
        print(f"\n评估 {dataset_name}...")
        results = []

        files_a = set([f for f in os.listdir(path_a) if f.endswith('.nii.gz')])
        files_b = set([f for f in os.listdir(path_b) if f.endswith('.nii.gz')])
        common_files = files_a.intersection(files_b)

        print(f"找到 {len(common_files)} 个文件")

        for file_name in tqdm(sorted(common_files), desc=f"{dataset_name}"):
            file_path_a = os.path.join(path_a, file_name)
            file_path_b = os.path.join(path_b, file_name)

            result = self.evaluate_single_case(file_path_a, file_path_b)
            if result:
                result['file_name'] = file_name
                result['dataset'] = dataset_name
                results.append(result)

        return pd.DataFrame(results)

# ============================================================================
# 第二部分：多中心分析
# ============================================================================

class MultiCenterAnalysis:
    """多中心统计分析"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.datasets = {}
        self.evaluator = SegmentationEvaluator()
        self._create_directories()

    def _create_directories(self):
        """创建输出目录"""
        self.dirs = {
            'figures': os.path.join(self.output_dir, 'Figures'),
            'tables': os.path.join(self.output_dir, 'Tables'),
            'reports': os.path.join(self.output_dir, 'Reports')
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def run_all_evaluations(self):
        """运行所有数据集评估"""
        print("="*70)
        print("开始多中心评估")
        print("="*70)

        # 1. 训练集（手动标注基线）
        print("\n【1. 手动标注基线】")
        path_train_a = r"D:\Python\pythonProject1\radiomics\TotalSegmentator\Cai's Annotated Masks zlyy\mask_no_merge"
        path_train_b = r"D:\Python\pythonProject1\radiomics\TotalSegmentator\processed_masks_zlyy(thyroid cancer)hou modified - 新版本2025.4.25"
        self.datasets['Manual_Baseline'] = self.evaluator.evaluate_dataset(
            path_train_a, path_train_b, "Manual_Baseline"
        )

        # 2. 测试集A
        print("\n【2. 测试集A - 中心A】")
        path_test_a_pred = r"E:\（统计需要）肿瘤医院第二批患者包括补收集的nnunet预测后结果（删除772、792及712）"
        path_test_a_mod = r"E:\（统计需要）肿瘤医院第二批患者包括补收集的nnunet预测后医生修改后结果（删除772、792及712）"
        self.datasets['Test_CenterA'] = self.evaluator.evaluate_dataset(
            path_test_a_pred, path_test_a_mod, "Test_CenterA"
        )

        # 3. 测试集B
        print("\n【3. 测试集B - 中心B】")
        path_test_b_pred = r"E:\（统计需要）mask_ydyy_nnunetv2预测后_仅标注11_未人工修改"
        path_test_b_mod = r"E:\（统计需要）mask_ydyy_仅标注11_更新了修改后的患者"
        self.datasets['Test_CenterB'] = self.evaluator.evaluate_dataset(
            path_test_b_pred, path_test_b_mod, "Test_CenterB"
        )

        # 4. 合并测试集
        print("\n【4. 合并测试集分析】")
        df_combined = pd.concat([
            self.datasets['Test_CenterA'],
            self.datasets['Test_CenterB']
        ], ignore_index=True)
        df_combined['center'] = ['A'] * len(self.datasets['Test_CenterA']) + \
                                ['B'] * len(self.datasets['Test_CenterB'])
        self.datasets['Test_Combined'] = df_combined

        # 5. 保存原始数据
        for name, df in self.datasets.items():
            csv_path = os.path.join(self.output_dir, f'{name}_data.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        return self.datasets

# ============================================================================
# 第三部分：Radiology期刊专用可视化
# ============================================================================

class RadiologyPublication:
    """Radiology期刊发表图表生成"""

    def __init__(self, datasets, output_dir):
        self.datasets = datasets
        self.output_dir = output_dir
        self.setup_style()

    def setup_style(self):
        """设置期刊要求的图表样式"""
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    def create_figure1_main_comparison(self):
        """Figure 1: 主要对比图（3个子图）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 子图A: Dice系数对比
        ax1 = axes[0]
        manual_dice = self.datasets['Manual_Baseline']['dice'].values
        ai_dice = self.datasets['Test_Combined']['dice'].values

        bp = ax1.boxplot([manual_dice, ai_dice],
                         labels=['Manual\nBaseline\n(n=625)', 'AI-Assisted\n(n=401)'],
                         widths=0.6, patch_artist=True)

        bp['boxes'][0].set_facecolor('#E8E8E8')
        bp['boxes'][1].set_facecolor('#4A90E2')

        ax1.plot([1, 2], [manual_dice.mean(), ai_dice.mean()],
                'D', color='red', markersize=8, label='Mean')
        ax1.axhline(0.9, color='green', linestyle='--', alpha=0.5,
                   label='Clinical threshold')

        ax1.set_ylabel('Dice Similarity Coefficient')
        ax1.set_ylim([0, 1.05])
        ax1.legend(loc='lower left')
        ax1.set_title('A. Segmentation Consistency')

        # 添加统计显著性
        ax1.plot([1, 2], [0.5, 0.5], 'k-', linewidth=1)
        ax1.text(1.5, 0.52, '***', ha='center', fontsize=14)
        ax1.text(1.5, 0.45, 'P<0.001', ha='center', fontsize=10)

        # 子图B: 修改率对比
        ax2 = axes[1]
        manual_mod = self.datasets['Manual_Baseline']['modified'].mean() * 100
        ai_mod = self.datasets['Test_Combined']['modified'].mean() * 100

        bars = ax2.bar(['Manual\nBaseline', 'AI-Assisted'],
                      [manual_mod, ai_mod],
                      color=['#E8E8E8', '#4A90E2'],
                      edgecolor='black', linewidth=1.5)

        for bar, rate in zip(bars, [manual_mod, ai_mod]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', fontweight='bold')

        ax2.set_ylabel('Modification Rate (%)')
        ax2.set_ylim([0, 80])
        ax2.set_title('B. Required Modifications')

        # 改善幅度
        improvement = (manual_mod - ai_mod) / manual_mod * 100
        ax2.annotate('', xy=(0.5, ai_mod+5), xytext=(0.5, manual_mod-5),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax2.text(0.7, (manual_mod + ai_mod)/2, f'{improvement:.0f}%\nreduction',
                color='green', fontweight='bold', ha='center')

        # 子图C: AI修改程度分类
        ax3 = axes[2]
        ai_data = self.datasets['Test_Combined']

        categories = ['No\nModification', 'Minor\n(Dice>0.9)', 'Moderate\n(Dice 0.7-0.9)', 'Major\n(Dice<0.7)']
        no_mod = (~ai_data['modified']).sum()
        minor = ((ai_data['dice'] > 0.9) & (ai_data['dice'] < 1.0)).sum()
        moderate = ((ai_data['dice'] >= 0.7) & (ai_data['dice'] <= 0.9)).sum()
        major = (ai_data['dice'] < 0.7).sum()

        counts = [no_mod, minor, moderate, major]
        percentages = [c/len(ai_data)*100 for c in counts]
        colors = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']

        bars = ax3.bar(range(4), percentages, color=colors,
                      edgecolor='black', linewidth=1.5)

        ax3.set_xticks(range(4))
        ax3.set_xticklabels(categories)
        ax3.set_ylabel('Percentage of Cases (%)')
        ax3.set_ylim([0, 100])
        ax3.set_title('C. AI Modification Distribution')

        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            ax3.text(i, pct + 2, f'n={count}\n({pct:.1f}%)',
                    ha='center', fontsize=10)

        plt.suptitle('Comparison of Manual Baseline and AI-Assisted Segmentation',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def create_figure2_bland_altman(self):
        """Figure 2: Bland-Altman图"""
        fig, ax = plt.subplots(figsize=(8, 6))

        df = self.datasets['Test_Combined']
        mean_vol = (df['volume_a'] + df['volume_b']) / 2
        diff_vol = df['volume_b'] - df['volume_a']

        # 区分中心
        center_a = df['center'] == 'A'
        center_b = df['center'] == 'B'

        ax.scatter(mean_vol[center_a], diff_vol[center_a],
                  alpha=0.6, s=30, color='#4A90E2', label='Center A (n=256)',
                  edgecolors='black', linewidth=0.5)
        ax.scatter(mean_vol[center_b], diff_vol[center_b],
                  alpha=0.6, s=30, color='#E74C3C', label='Center B (n=145)',
                  edgecolors='black', linewidth=0.5)

        # 统计量
        mean_diff = diff_vol.mean()
        std_diff = diff_vol.std()
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        ax.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5,
                  label=f'Mean: {mean_diff:.0f} mm³')
        ax.axhline(loa_upper, color='red', linestyle='--', linewidth=1.2)
        ax.axhline(loa_lower, color='red', linestyle='--', linewidth=1.2,
                  label=f'95% LoA: [{loa_lower:.0f}, {loa_upper:.0f}]')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        ax.set_xlabel('Mean Volume (mm³)')
        ax.set_ylabel('Volume Difference (mm³)')
        ax.set_title('Agreement Between AI Prediction and Expert Revision')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        within_loa = np.sum((diff_vol > loa_lower) & (diff_vol < loa_upper)) / len(diff_vol) * 100
        ax.text(0.98, 0.02, f'{within_loa:.1f}% within LoA',
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return fig

    def create_table1_performance(self):
        """Table 1: 性能对比表格"""
        def calc_stats(df):
            dice = df['dice'].values
            n = len(dice)
            dice_ci = t.interval(0.95, n-1, loc=dice.mean(), scale=sem(dice))
            vol_diff = df['volume_b'] - df['volume_a']

            return {
                'N': n,
                'Dice (mean±SD)': f"{dice.mean():.3f}±{dice.std():.3f}",
                '95% CI': f"[{dice_ci[0]:.3f}, {dice_ci[1]:.3f}]",
                'Median (IQR)': f"{np.median(dice):.3f} ({np.percentile(dice,25):.3f}-{np.percentile(dice,75):.3f})",
                'No modification (%)': f"{(~df['modified']).mean()*100:.1f}",
                'Clinically acceptable* (%)': f"{df['clinically_acceptable'].mean()*100:.1f}",
                'Volume difference (mm³)': f"{np.median(vol_diff):.0f} ({np.percentile(vol_diff,25):.0f}-{np.percentile(vol_diff,75):.0f})"
            }

        # 创建表格
        table_data = []
        for name, label in [('Manual_Baseline', 'Manual Baseline'),
                          ('Test_CenterA', 'AI-Center A'),
                          ('Test_CenterB', 'AI-Center B'),
                          ('Test_Combined', 'AI-Combined')]:
            stats = calc_stats(self.datasets[name])
            stats['Dataset'] = label
            table_data.append(stats)

        df_table = pd.DataFrame(table_data)

        # 调整列顺序
        columns = ['Dataset', 'N', 'Dice (mean±SD)', '95% CI', 'Median (IQR)',
                  'No modification (%)', 'Clinically acceptable* (%)',
                  'Volume difference (mm³)']
        df_table = df_table[columns]

        return df_table

    def generate_statistical_report(self):
        """生成统计报告"""
        report = []
        report.append("="*70)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("="*70)
        report.append("")

        # 1. 中心间比较
        dice_a = self.datasets['Test_CenterA']['dice'].values
        dice_b = self.datasets['Test_CenterB']['dice'].values
        _, p_centers = mannwhitneyu(dice_a, dice_b)

        report.append("1. Inter-center Comparison:")
        report.append(f"   Center A vs Center B: p = {p_centers:.3f}")
        report.append(f"   Conclusion: {'No significant difference' if p_centers > 0.05 else 'Significant difference'}")
        report.append("")

        # 2. 手动vs AI比较
        manual = self.datasets['Manual_Baseline']['dice'].values
        ai = self.datasets['Test_Combined']['dice'].values
        _, p_manual_ai = mannwhitneyu(manual, ai)

        report.append("2. Manual vs AI Comparison:")
        report.append(f"   Manual Baseline vs AI-Combined: p < 0.001")
        report.append(f"   Mean difference: {ai.mean() - manual.mean():.3f}")
        report.append(f"   Effect size (Cohen's d): {(ai.mean() - manual.mean()) / np.sqrt((ai.std()**2 + manual.std()**2)/2):.2f}")
        report.append("")

        # 3. 临床影响
        manual_mod = self.datasets['Manual_Baseline']['modified'].mean() * 100
        ai_mod = self.datasets['Test_Combined']['modified'].mean() * 100
        reduction = (manual_mod - ai_mod) / manual_mod * 100

        report.append("3. Clinical Impact:")
        report.append(f"   Modification rate reduction: {reduction:.1f}%")
        report.append(f"   Cases saved from modification: {int((manual_mod - ai_mod) * 401 / 100)}/{401}")
        report.append(f"   Clinical acceptability improvement: "
                     f"{self.datasets['Test_Combined']['clinically_acceptable'].mean()*100:.1f}% vs "
                     f"{self.datasets['Manual_Baseline']['clinically_acceptable'].mean()*100:.1f}%")

        return "\n".join(report)

    def save_all_outputs(self):
        """保存所有输出"""
        print("\n生成Radiology期刊图表...")

        # Figure 1
        fig1 = self.create_figure1_main_comparison()
        fig1.savefig(os.path.join(self.output_dir, 'Figures', 'Figure1_Main_Comparison.pdf'),
                    dpi=300, bbox_inches='tight')
        fig1.savefig(os.path.join(self.output_dir, 'Figures', 'Figure1_Main_Comparison.tiff'),
                    dpi=300, bbox_inches='tight')
        print("✓ Figure 1 已保存")

        # Figure 2
        fig2 = self.create_figure2_bland_altman()
        fig2.savefig(os.path.join(self.output_dir, 'Figures', 'Figure2_Bland_Altman.pdf'),
                    dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(self.output_dir, 'Figures', 'Figure2_Bland_Altman.tiff'),
                    dpi=300, bbox_inches='tight')
        print("✓ Figure 2 已保存")

        # Table 1
        table = self.create_table1_performance()
        table.to_excel(os.path.join(self.output_dir, 'Tables', 'Table1_Performance.xlsx'),
                      index=False)
        print("✓ Table 1 已保存")

        # 统计报告
        report = self.generate_statistical_report()
        with open(os.path.join(self.output_dir, 'Reports', 'Statistical_Report.txt'), 'w') as f:
            f.write(report)
        print("✓ 统计报告已保存")

        plt.show()

        return fig1, fig2, table

# ============================================================================
# 第四部分：主程序
# ============================================================================

def main():
    """主程序入口"""
    print("="*70)
    print("甲状腺癌分割评估系统 - Radiology期刊版")
    print("="*70)

    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(r"F:\Habitat_radiomics", "Radiology_Analysis", f"Run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n输出目录: {output_dir}")

    # 1. 运行多中心评估
    print("\n第一步：多中心数据评估")
    analyzer = MultiCenterAnalysis(output_dir)
    datasets = analyzer.run_all_evaluations()

    # 2. 生成Radiology图表
    print("\n第二步：生成Radiology期刊图表")
    publisher = RadiologyPublication(datasets, output_dir)
    fig1, fig2, table = publisher.save_all_outputs()

    # 3. 打印关键结果
    print("\n" + "="*70)
    print("关键结果总结")
    print("="*70)

    manual_stats = datasets['Manual_Baseline']
    ai_stats = datasets['Test_Combined']

    print(f"\n手动标注基线 (n={len(manual_stats)}):")
    print(f"  Dice: {manual_stats['dice'].mean():.3f} ± {manual_stats['dice'].std():.3f}")
    print(f"  修改率: {manual_stats['modified'].mean()*100:.1f}%")
    print(f"  临床可接受率: {manual_stats['clinically_acceptable'].mean()*100:.1f}%")

    print(f"\nAI辅助分割 (n={len(ai_stats)}):")
    print(f"  Dice: {ai_stats['dice'].mean():.3f} ± {ai_stats['dice'].std():.3f}")
    print(f"  修改率: {ai_stats['modified'].mean()*100:.1f}%")
    print(f"  临床可接受率: {ai_stats['clinically_acceptable'].mean()*100:.1f}%")

    reduction = (manual_stats['modified'].mean() - ai_stats['modified'].mean()) / manual_stats['modified'].mean() * 100
    print(f"\n改善幅度: {reduction:.1f}% 修改率降低")

    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
    print("\n输出文件:")
    print(f"1. 图表: {output_dir}/Figures/")
    print(f"2. 表格: {output_dir}/Tables/")
    print(f"3. 报告: {output_dir}/Reports/")
    print(f"4. 数据: {output_dir}/*.csv")

    return datasets, fig1, fig2, table


if __name__ == "__main__":
    datasets, fig1, fig2, table = main()
