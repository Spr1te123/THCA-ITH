# A biologically validated CT quantifying intratumoral heterogeneity signature predicts risk of distant metastasis in differentiated thyroid cancer:A multicenter study
This multicenter study aims to solve the clinical challenge of predicting the risk of distant metastasis in differentiated thyroid cancer (DTC). We successfully constructed and rigorously validated a multimodal imaging prediction model called "CRIT phenotype" by integrating clinicopathological features, traditional radiomics, and our newly developed iTED feature set that quantifies the heterogeneity of the "microenvironment" within the tumor. The model demonstrated excellent predictive performance and generalization ability in data containing more than 1,000 patients, with a sensitivity of 93.3% in identifying high-risk patients and a negative predictive value of 98.9% in independent external validation. More importantly, we provide an in-depth biological explanation for this imaging "black box" for the first time: through digital pathology analysis, we found that the high-risk CRIT phenotype was associated with lower tumor cell heterogeneity, providing clinical imaging evidence for the "dominant clonal amplification" hypothesis of aggressive tumors; At the same time, proteomic validation further revealed that this high-risk phenotype is closely related to the downregulation of the expression of key proteins of mitochondrial metabolism (SUCLG1 and DLAT), pointing to the underlying mechanism of mitochondrial dysfunction. Therefore, this study It not only provides a powerful personalized risk stratification tool, but also innovatively combines the predictive potential of preoperative imaging with the diagnostic certainty of postoperative pathology to build a more robust risk assessment system, and established a comprehensive research paradigm integrating "macroscopic imaging-micropathology-protein expression", which opened up a new way for the development of imaging biomarkers with a solid biological foundation.

### Repository Structure  
THCA-ITH/  
│  
├── 0_Semi-automatic_lesion_segmentation_in_CT_images/  
│   ├── 0.1_CBAM.py           # The Convolutional Block Attention Module  
│   ├── 0.2_Segmentation_Modification_Assessment_and_Review_Tool.py             # Segmentation Performance Evaluation  
│   └── README.md  
│  
├── 1_radiomics_feature_extraction/  
│   ├── 1.1_Thyroid_Radiomics_Standard.py           # PyRadiomics features (~107)  
│   ├── 1.2_TITAN_pipeline.py             # iTED + 3D ITHscore (~50)  
│   └── README.md  
│  
├── 2_clinical_data_processing/  
│   ├── 2.1_Clinical_data_preprocessing.py              # Clinical feature preprocessing  
│   ├── 2.2_Comparison_of_clinical_baseline_data.py                # Table 1 generation  
│   ├── 2.3_Univariate_and_multivariate_analysis.py            # Univariate/multivariate analysis  
│   └── README.md  
│  
├── 3_model_training/  
│   ├── 3.1_Lightgbm_training.py               # LightGBM model with SMOTE  
│   └── README.md  
│  
├── 4_pathology_validation/  
│   ├── 4.1_wsi_processing_multi_tissue.py        # HoVer-Net segmentation       
│   ├── 4.2_Batch_process_multi_tissue.sh            # Batch process nucleus segmentation  
│   ├── 4.3_Generate_reference_all.py              # Stain normalization  
│   ├── 4.4_Nuclear_feature_extraction.py      # 28 features per nucleus  
│   ├── 4.5_Thyroid_cancer_pathology_analysis.py              # Pathology-radiomics correlation  
│   └── README.md  
│  
├── 5_protein expression validation/  
│   ├── 5.1_Protein_expression_analysis.py     # Protein profiling  
│   └── README.md  
│  
├── README.md                                   # This file  

