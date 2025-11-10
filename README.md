The core objective was to develop and rigorously validate a multimodal imaging phenotype that can accurately identify the risk of DM in DTC before surgery, which is known as the clinical-radiomics-intra-tumor ecological diversity (iTED) (CRIT) phenotype. Furthermore, this study investigated the biological mechanisms underlying the proposed CRIT phenotype by integrating digital pathology analysis with the expression profile of DTC DM biomarkers identified through prior proteomics studies, thereby providing an objective biological basis for imaging-based prediction.

thyroid-cancer-metastasis-prediction/
│
├── 1_radiomics_feature_extraction/          # 影像组学特征提取
│   ├── 1.1_traditional_radiomics.py           # PyRadiomics标准特征 (~107个)
│   ├── 1.2_TITAN_iTED_pipeline.py             # iTED + 3D ITHscore (~50个)
│   └── README.md
│
├── 2_clinical_data_processing/              # 临床数据处理
│   ├── 2.1_data_preprocessing.py              # 临床特征预处理
│   ├── 2.2_comparison_table.py                # Table 1生成
│   ├── 2.3_statistical_analysis.py            # 单因素/多因素分析
│   └── README.md
│
├── 3_model_training/                        # 机器学习模型
│   ├── 3.1_lightgbm_training.py               # LightGBM训练与预测
│   └── README.md
│
├── 4_pathology_validation/                  # 病理学验证
│   ├── 4.1_generate_stain_reference.py        # 染色归一化参考
│   ├── 4.2_nucleus_segmentation.py            # HoVer-Net细胞核分割
│   ├── 4.3_batch_segmentation.sh              # 批处理脚本
│   ├── 4.4_nuclear_feature_extraction.py      # 每个核28个特征
│   ├── 4.5_pathology_analysis.py              # 病理-影像关联分析
│   └── README.md
│
├── 5_molecular_validation/                  # 分子验证
│   ├── 5.1_protein_expression_analysis.py     # 蛋白表达分析
│   └── README.md
│
├── README.md                                 # 英文文档
├── README_CN.md                              # 本文档
├── requirements.txt                          # Python依赖
├── environment.yml                           # Conda环境
├── LICENSE                                   # MIT许可证
└── .gitignore
