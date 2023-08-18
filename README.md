# Group 9 Final Project
**Pre-trained deep learning image segmentation: Are simple classical methods still relevant for clinical MRI?**

*Contributers: Jonathan Williams, Eddy Rosales, and Jesse Maki*
<br>
<br>

### ***Introduction***

  Magnetic resonance imaging (MRI) is a commonly used non-invasive medical imaging modality, particularly in neuroimaging. Brain structure segmentations are used in medical research for volumetric analysis of the brain, enabling researchers to quantify size and structural changes of brain anatomy. While manual labeling is seen as the gold standard, automated methods are favorable for studies with large cohorts. FreeSurfer, released in 1999, is the most used software suite for analyzing neuroimaging data, among its features is automatic segmentation [4, 5]. However, these classical segmentation models, like Freesurfer’s recon-all pipeline, although standard, are computationally demanding to run. Processing a single subject can take anywhere from 4 to 12 hours depending on available resources. These computational demand poses challenges, especially in the context of large cohort studies with thousands of subjects. The FMRIB Software Library’s (FSL) [6], another open-source tool for neuroimaging, can be used for more basic/simple segmentations. FSL’s FAST pipeline is a segmentation tool based on k-means clustering, which can be used for segmenting salient structures like grey/white matter, and cerebrospinal fluid. These large structures can still be used to study brain atrophy.
  
  Deep learning methods have been widely used for different image classification tasks. Yet, models can be difficult to train, and can have limited generalizability for different segmentation tasks. However, in the limited domain of clinical MRI images, pre trained segmentation models can be run much faster than classical methods without sacrificing performance. Synthseg [7] is an example of a deep learning segmentation tool that does not require re-training, it uses generative models to create synthetic data at training time to improve generalizability between different MRI sequences.

### ***Problem Statement***
1. The benefits of using simple segmentation methods for the segmentation of clinical brain MRI is unclear, as pre-trained deep learning have the potential of better performance with inexpensive computational requirements.
2. To use pre-trained deep learning tools effectively it is critical to understand the limitation in performance per anatomical label. This understanding will lead to informed decisions of when to take advantage of the benefits of these pre-trained models.

### ***Datasets Used***
* Demographic rich MRI image dataset
  * IXI Dataset T1 Weighted Images: [Online] Available: brain-development.org/ixi-dataset/
  * This dataset contains 600 unique individuals of ages ranging from 18-89, specially to create our
age-volume prediction models.
* Information rich MRI image dataset:
  * Human Brain Phantom MRI Dataset. [Online] Available:
kaggle.com/datasets/ukeppendorf/frequently-traveling-human-phantom-fthp-dataset
  * This dataset includes 500 scans on the same subject in different conditions like slice acquisition
and scanner model. We used this to check the robustness of both models in different scanning conditions.
* Manual label dataset:
  * Internet Brain Segmentation Repository [Online] Available: nitrc.org/frs/?group_id=48
  * This dataset contains manual labels created by clinicians and will be used as a ground truth to
validate accuracy in each segmentation model tested.

### ***Data Benchmarks*** 

As individuals age, there is a well-documented trend of overall decreased brain volume [9] and thinning
of grey matter [10]. Our initial problem statement centers around leveraging this widely observed phenomenon to establish a benchmark for evaluating lower-precision techniques that offer quicker computational processing. The age-prediction problem takes advantage of the simple classical method’s limitation of segmenting only large features. Using 600 T1-weighted MRI from the IXI dataset [1] volumes were processed using FSL’s FAST and Synthseg. Using five different models - linear regression, light gradient boosted, extreme gradient boosting, random forest, and a 3D convolutional neural network, we generated predictions using the label outputs from each of the models. We benchmarked performance by measuring the R-square and mean absolute error (MAE) and comparing between the two segmentation models.

For our second problem statement we will compare the difference between manual segmentations and automatically generated labels with Freesurfer (state-of-the art classical approach) and Synthseg. Using eighteen manually labeled MRI volumes from Internet Brain Segmentation Repository [3]. We calculated the dice scores from each of the automatic segmentation models to the manual labels. Dice is a commonly used similarity metric in segmentation problems which ranges from 0 to 100, where 0 is no overlap and 100 is complete overlap. Higher similarity scores compared to the ground truth labels imply a more accurate model.

### ***Approaches Algorithms and Tools***   
  *Preprocessing and segmentations*  
Scripts were used to automate commands for Freesurfer, FSL, and Synthseg. Before all segmentations could be done, volumes needed to be “skull stripped” before using the FSL FAST, pipeline. Additional commands were generated to do skull stripping using FSL BET. In the following figure we show the difference between the original and skull stripped MRI volume.

<img width="869" alt="Screenshot 2023-08-18 at 6 25 32 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/96d5b96c-18b8-4aba-b07f-9e574bcfc749">

Each segmentation pipeline had options to export final segmentation into .csv files with the total voxel volume of each of the labels. Information in the csv files was parsed with python and used to create all regressions. The actual segmentation volumes were used to train the 3D convolutional neural network, and calculate the dice coefficients between different methods.

<img width="875" alt="Screenshot 2023-08-18 at 6 26 37 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/89c284e5-dfe9-4358-add0-720b16c64e4d">

Next, volumes were cropped using free surfers mri_convert [11], and dice was calculated using mri_compute_overlap [8] both part of the FreeSurfer software suite. Commands also had to be created using python to automate the process. Subsequently, outputs from these commands were parse and plotted using matplotlib.

*Regression Model*  
This research implements a diverse set of machine learning models and algorithms to analyze datasets processed through FSL and Synthseg. Python libraries used include pandas, matplotlib, and scikit-learn are utilized. Data preprocessing involved parsing information from the FSL/Synthseg outputs and partitioning data into mutually exclusive training and testing subsets. Notably, feature scaling proceeds via the `StandardScaler` from scikit-learn to preclude model bias toward variables with greater magnitudes.

In the modeling phase, four distinct regression models are deployed on each dataset: `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, and `LinearRegression`. Hyperparameter tuning of each model occurs to optimize results, including setting `n_estimators`, `max_depth`, and `learning_rate` in a model-specific manner. The objective remains predicting age based on features such as brain volumes. Model was evaluated via the R-squared and Mean Absolute Error (MAE).

*3D-CNN Model*    
We used a 3D convolutional neural networks (CNNs) to analyze three types of segmented brain volumes (white matter and gray matter segmented volumes (FSL), as well as SynthSeg’s fully segmented volumes). The volumes were preprocessed and split into training, validation, and test sets for efficient storage in as a numpy array (.npz). The architecture consists of several convolutional blocks, each containing stacked 3D convolutional layers followed by batch normalization, ReLU activation, max-pooling, and dropout for regularization. This is followed by a dense layer with linear activation for the regression task.

Data generators are used for memory-efficient training of the large dataset. Techniques like ReduceLROnPlateau and EarlyStopping are integrated to enhance the learning process. The model is trained to minimize mean squared error loss using the Adam optimizer. Validation strategies, including segregating the data and leveraging callbacks on the validation set, ensure the model generalizes well. We utilized a training split of 64% training, 16% validation, and 20% testing.

### ***Metrics and Evaluations***

In image analysis, a commonly used technique for evaluating the similarity of segmented regions is the Sørensen-Dice coefficient. This method is crucial when comparing manual segmentations to those generated by classical techniques (such as Freesurfer, FSL k-means) or machine learning models (such as Synthseg).

The Dice coefficient measures the spatial overlap between two sets with a value of 100 indicating a perfect match. Lower values suggest inaccuracies in the segmentation algorithm. A smaller distance signifies a closer match to the ground truth, while a larger distance indicates greater deviation between the regions. By using these metrics, we can evaluate and compare the performance of different segmentation techniques and machine learning models, promoting the development of more precise and reliable approaches to image analysis in the realm of MRI scans
Trend-lines were created for each of the prediction methods, comparing the predicted versus actual values. R- square and mean absolute error was calculated on these trends to evaluate performance of each of the models to create segmentations for an age prediction model.

### ***Results***

*Inter-Scanner Variability*
<img width="836" alt="Screenshot 2023-08-18 at 6 29 56 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/22a564c7-6d3c-4ea7-a0c0-ddf6964b7e29">


Inter-scanner variability demonstrated using boxplots with volume calculated over the 116 distinct models in the Human Brain Phantom MRI dataset. Volume calculated on the left using SynthSeg while FSL was used on the right boxplot. Strong inter-model variability demonstrated via FSL, while SynthSeg had more consistent readings across different models.

*Regression Models Graphs*
<img width="603" alt="Screenshot 2023-08-18 at 6 30 18 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/3a2ffd83-29a7-4541-af49-21107174427f">

The XGBRegressor model trained on the FSL dataset performed the best with an R2 score of 0.568 and MAE of 8.412. Interestingly, the second-best was the linear regression model, but with the SEG dataset, achieving an R2 of 0.566 and MAE of 8.476. The remaining models had lower accuracy in age prediction with R2 scores ranging from 0.425 to 0.548. This highlights the impact of segmentation methods and model types on performance.

*3D-CNN Regression Plots* 

<img width="725" alt="Screenshot 2023-08-18 at 6 33 46 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/60b86282-4204-455b-b8bb-5e6b4fd172f8">

<img width="750" alt="Screenshot 2023-08-18 at 6 34 08 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/4eebfc84-8063-4543-b762-2adcc4798205">

<img width="696" alt="Screenshot 2023-08-18 at 6 35 31 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/1a7aaec6-ff3d-4ece-9141-106b084fa15e">

The results obtained from the 3D CNN models, although more complex than previous models, showed little improvement in MAE from the predictions calculated in this notebook. There are several factors that may have affected the performance of the model. Firstly, the model architecture only allows for one set of segmented volume images to be uploaded for training and testing, unlike the other models that include white matter, gray matter, and cerebrospinal fluid volumes. Secondly, the sample size of the dataset is relatively small compared to that of other professional machine learning researchers for age-volume predictions. Lastly, the age distribution in the dataset is more prevalent in the 25-30 and 55-70 age brackets, resulting in higher accuracy among the higher distributions.

*Dice Score Comparisons with Manual Segmentations*
<img width="843" alt="Screenshot 2023-08-18 at 6 36 35 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/d1727d6d-ad99-4b86-839f-bd50df774de2">

Visualized is the resulting dice scores (0-100) comparing Freesurfer’s recon-all and Synthseg. Previously we also ran recon-all without segmentation refinement (-autorecon1 [5]) and noted no difference in dice score between FreeSurfer with and without segmentation refinement. The lowest performing label for Synthseg was the left vessel, a very small structure just a couple of voxels large, notably recon-all was able to have greater accuracy with this small structure. Besides the left vessel, Synthseg had greater consistency overall, with with only two outliers between all labels, while freesurfer had about 30 outliers in total. Yet the median performace between both models is comparable, always landing within error to each other. Note some labels had to be omitted because they were not labeled in the manual labels such as the cerebrospinal fluid. The following figure shows the differences in the Synthseg and manual label, note the CSF on the Synthseg label labeled in grey.

<img width="866" alt="Screenshot 2023-08-18 at 6 37 36 PM" src="https://github.com/jessexmaki/CAP4770-Final-Project/assets/87655161/d218bec2-3ccf-462b-bb99-0c4512459302">

### ***Conclusions and Challenges***
*Data Challenges*

  One of the most significant challenges we encountered at the beginning of the project was the limited scope of the initial dataset [2]. It consisted of information about only one patient tracked over two years. The dataset did provide meaningful insight into variability between MRI scanner models. However, this limited dataset posed a problem in generalizing the findings and models to a larger scope in terms of age. It constrained our ability to create robust and representative analytical models. To overcome this, we turned to the IXI dataset [1], which provided a more extensive collection of over 600 MRI samples across a diverse population. This enhanced our analysis's representativeness and enabled us to derive more meaningful and generalizable insights.
  
Initially we decided on a central server to handle the preprocessing and processing of the datasets. Data utilization became a factor and the server in Amazon Web Services had to be expanded to accommodate the large .NII files which are the MRI images. To mitigate the storage challenges, we resorted to using local machines for computations. This shift allowed for parallel processing on several local machines, and we leveraged AWS (Amazon Web Services) S3 storage services to save and share processed and preprocessed data.

In retrospect, the lessons learned from these challenges provide valuable insights into scalability and data management. The decision between centralized and decentralized computers, selecting appropriate datasets, and efficiently handling large-scale MRI data are all factors that will influence project planning. The experience gained through overcoming these challenges has paved the way for a more informed approach to handling similar large-scale analytical projects in the future.

*Accuracy and Scalability* 

Although there is high variability in brain volume when correlated with age, even when normalizing measurements. Pathology, genetics, and many other factors which cannot all be accounted for can affect the relationship between brain volume and age. Thus, any comparison between these types of correlations benefits from having as large sample size as possible. Accuracy of these comparisons will have a direct link with the number of subjects observed. We believe that the comparison between models is still valid since both models are working off the same data.

Other limits to accuracy to consider, are for the second problem statement. Dice does not directly measure accuracy but similarity to the ground truth labels. Although manual labels are the closest to a ground truth we can use in this study, we must acknowledge inter- and itra- labeler variability. Different experts will not have a perfect overlap on their labels, and the same labeler will differ if labeling the same volume at different points in time. To improve accuracy, we would need multiple labelers on the same volume, and compute a majority vote, or probabilistic ground truth. Yet, this is the best dataset we could find with full manual labels on all subcortical regions of the brain.

## References
1. “IXI Dataset” [Online] Available: brain-development.org/ixi-dataset/. [Accessed: August 8, 2023]
2. “Human Brain Phantom MRI Dataset” [Online] Available: kaggle.com/datasets/ukeppendorf/frequently- traveling-human-phantom-fthp-dataset [Accessed: July 3 2023]
3. “Internet Brain Segmentation Repository” [Online] Available: nitrc.org/frs/?group_id=48 [Accessed: July 3 2023]
4. B. Fischl. "FreeSurfer." In Neuroimage (2012).
5. “FreeSurfer Analysis Pipeline Overview”, FreeSurferWiki, 2017. [Online]. Available:
surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferAnalysisPipelineOverview/. [Accessed: July 4, 2023]
6. S.M Smith et al., “Advances in functional and structural MR image analysis and implementation as FSL,” in NeuroImage, (2004)
7. B. Billot, D. Greve, O. Puonti, et. al, “SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining,” in Medical Image Analysis, 2023.
8. L. Zollei, “mri_compute_overlap” [Online] Available: https://surfer.nmr.mgh.harvard.edu/fswiki/mri_compute_overlap
9. R. Peters. "Ageing and the brain: This article is part of a series on ageing edited by Professor Chris Bulpitt." Postgraduate medical journal (2006)
10. S. Fujita, et al. "Characterization of Brain Volume Changes in Aging Individuals With Normal Cognition Using Serial Magnetic Resonance Imaging." JAMA Network Open (2023)
11. D. Greve, and B. Fischl. “mri_convert” [Online] Available: surfer.nmr.mgh.harvard.edu/fswiki/mri_convert

