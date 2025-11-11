1.Automated Anatomical Context Segmentation: 
  The TotalSegmentator tool¹(https://github.com/wasserth/TotalSegmentator) was employed for the automated multi-organ segmentation of sequential CT images to provide anatomical context. Although this model is capable of segmenting over 100 anatomical structures, we retained only those relevant for the localization of thyroid cancer: the thyroid gland (label 1), trachea (label 2), left/right common carotid arteries (labels 3, 4), cervical vertebrae C5–C7 (labels 5–7), first thoracic vertebra T1 (label 8), and left/right clavicles (labels 9, 10).
2.Ground-Truth Lesion Delineation: 
  For the training set, thyroid cancer lesion masks were manually delineated by expert radiologists to establish the ground truth.
3.Multi-label Mask Generation: 
  For each case, the 11 anatomical structure annotations and the lesion delineation were merged into a single multi-label mask file, which served as the input for training the CBAM-nnUNetv2 model.
4.Integration of CBAM into the nnUNetv2 Framework: 
  The network architecture was based on nnUNetv2²(https://github.com/MIC-DKFZ/nnUNet), with the core implementation imported from the dynamic-network-architectures library. To integrate the Convolutional Block Attention Module (CBAM)³, the library was first installed from its source. A custom Python module was created at ./dynamic-network-architectures/dynamic_network_architectures/architectures/cbamunet.py to house the CBAM implementation (by pasting the content of the CBAM.py script). Subsequently, the model configuration was updated by modifying the network_class_name parameter within the preprocessed dataset's nnUNetPlans.json file to dynamic_network_architectures.architectures.cbamunet.CBAMPlainConvUNet. Following these configuration changes, the CBAM-nnUNetv2 model was trained using the standard nnUNetv2 command pipeline².
5.Segmentation Performance Evaluation: 
  The performance of the resulting segmentation model was rigorously evaluated against the ground-truth annotations.

References:
1.Wasserthal, J., Breit, H.-C., Meyer, M.T., et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence (2023). https://doi.org/10.1148/ryai.230024
2.Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211 (2021).
3.Woo, S., Park, J., Lee, J.Y. & Kweon, I.S. Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19) (2018).
