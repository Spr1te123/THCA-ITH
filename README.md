# Development and validation of a multimodal model for predicting the risk of distant metastasis of differentiated thyroid cancer by quantifying intratumoral heterogeneity in CT images of primary lesions
The core objective was to develop and rigorously validate a multimodal imaging phenotype that can accurately identify the risk of DM in DTC, which is known as the clinical-radiomics-intra-tumor ecological diversity (iTED) (CRIT) phenotype. Furthermore, this study investigated the biological mechanisms underlying the proposed CRIT phenotype by integrating digital pathology analysis with the expression profile of DTC DM biomarkers identified through prior proteomics studies, thereby providing an objective biological basis for imaging-based prediction.

### Repository Structure  
THCA-ITH/  
│  
├── 1_radiomics_feature_extraction/  
│   ├── 1.1_Thyroid_Radiomics_Standard.py           # PyRadiomics features (~107)  
│   ├── 1.2_TITAN_pipeline.py             # iTED + 3D ITHscore (~50)  
│   └── README.md  
│  
├── 2_clinical_data_processing/  
│   ├── 2.1_clinical_data_preprocessing.py              # Clinical feature preprocessing  
│   ├── 2.2_Comparison of clinical baseline data.py                # Table 1 generation  
│   ├── 2.3_Univariate and multivariate analysis.py            # Univariate/multivariate analysis  
│   └── README.md  
│  
├── 3_model_training/  
│   ├── 3.1_lightgbm_training.py               # LightGBM model with SMOTE  
│   └── README.md  
│  
├── 4_pathology_validation/  
│   ├── 4.1_wsi_processing_multi_tissue.py        # HoVer-Net segmentation       
│   ├── 4.2_batch_process_multi_tissue.sh            # Batch process nucleus segmentation  
│   ├── 4.3_generate_reference_all.py              # Stain normalization  
│   ├── 4.4_nuclear_feature_extraction.py      # 28 features per nucleus  
│   ├── 4.5_thyroid_cancer_pathology_analysis.py              # Pathology-radiomics correlation  
│   └── README.md  
│  
├── 5_protein expression validation/  
│   ├── 5.1_protein_expression_analysis.py     # Protein profiling  
│   └── README.md  
│  
├── README.md                                   # This file  
├── README_CN.md                                # Chinese version  
├── requirements.txt                            # Python dependencies  
├── environment.yml                             # Conda environment  
├── LICENSE                                     # MIT License  
└── .gitignore  
