###1.Automated Segmentation of Anatomical Structures:
The TotalSegmentator tool¹ was employed for the automated multi-organ segmentation of sequential CT images to provide anatomical context. Although the model is capable of segmenting over 100 structures, we retained only those relevant for the localization of thyroid cancer: the thyroid gland (label 1), trachea (label 2), left/right common carotid arteries (labels 3, 4), cervical vertebrae C5–C7 (labels 5–7), first thoracic vertebra T1 (label 8), and left/right clavicles (labels 9, 10).
(Note: The reference ¹ should point to the GitHub link: https://github.com/wasserth/TotalSegmentator)
###2.Manual Delineation of Lesion Masks for Training:
For the training samples, thyroid cancer lesion masks were manually delineated by expert radiologists to serve as the ground truth.
###3.Creation of a Multi-label Training Mask:
For each case, the 11 anatomical structure annotations and the lesion delineation were merged into a single multi-label mask file, which served as the input for training the CBAM-nnUNetv2 model.
###4.Integration of CBAM into the nnUNetv2 Framework:
The integration of the Convolutional Block Attention Module (CBAM) was achieved by customizing the nnUNetv2 framework. The base network architecture was directly imported from the library. A custom module, , was created within this library to implement the CBAM architecture. Subsequently, the model configuration was updated by modifying the parameter in the preprocessed dataset's file to point to our custom class. With these modifications, the model was trained using the standard nnUNetv2 command pipeline, with further details available at the official repository ().dynamic-network-architecturescbamunet.pynetwork_class_namennUNetPlans.jsonCBAMPlainConvUNethttps://github.com/MIC-DKFZ/nnUNet
###5.Segmentation Performance Evaluation:
The performance of the segmentation model was rigorously evaluated by assessing its agreement with the ground-truth annotations using a comprehensive set of metrics.
