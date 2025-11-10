The core objective was to develop and rigorously validate a multimodal imaging phenotype that can accurately identify the risk of DM in DTC before surgery, which is known as the clinical-radiomics-intra-tumor ecological diversity (iTED) (CRIT) phenotype. Furthermore, this study investigated the biological mechanisms underlying the proposed CRIT phenotype by integrating digital pathology analysis with the expression profile of DTC DM biomarkers identified through prior proteomics studies, thereby providing an objective biological basis for imaging-based prediction.
多模态数据输入
├── CT影像 (DICOM/NIfTI)
├── 临床数据 (CSV)
├── WSI病理切片 (.svs/.tif)
└── 蛋白表达数据 (CSV/Excel)
    ↓
┌─────────────────────────────────────┐
│ 步骤1: 影像组学特征提取                  │
│ ├── 传统PyRadiomics特征 (~107个)       │
│ ├── iTED异质性特征 (~50个)             │
│ └── 3D空间异质性评分 (1个)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 步骤2: 临床数据处理                     │
│ ├── 数据清洗和预处理                   │
│ ├── 训练集-验证集比较 (Table 1)        │
│ └── 单因素/多因素统计分析               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 步骤3: 机器学习模型训练                 │
│ ├── 特征融合 (影像+临床)               │
│ ├── LightGBM模型训练                  │
│ ├── 内部验证 (5折交叉验证)            │
│ └── 外部验证 (多中心)                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 步骤4: 病理学验证                      │
│ (AI预测的高风险 vs 低风险组)           │
│ ├── WSI细胞核分割 (HoVer-Net)         │
│ ├── 核形态学特征提取 (28特征)          │
│ ├── 肿瘤内异质性(ITH)分析             │
│ └── 病理-影像特征相关性                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 步骤5: 分子验证                        │
│ └── 蛋白表达差异分析                   │
│     (高风险组 vs 低风险组)             │
└─────────────────────────────────────┘
    ↓
最终输出
├── 预测模型 (LightGBM .pkl)
├── 风险分层结果
├── 病理验证报告
├── 分子机制解释
└── 发表级可视化图表
