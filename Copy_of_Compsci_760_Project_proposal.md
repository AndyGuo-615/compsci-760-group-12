# **Project proposal**

**Does the Fingerprint-to-Blood-Group Result Survive a Leakage-Controlled Evaluation?** 

					  
Group number: 12  
Team member: Andy Guo, Xia Yang, Xiting Li, Litong Chen, Zoe Wang  
Date: 

# **1\. Project Summary (Peter)**

Recent machine learning studies have reported high accuracy when predicting blood groups from fingerprint images. Tripathi et al. (2026) reported a validation accuracy of 85.5%, while Phadke et al. (2025) reported an overall accuracy of 88% for fingerprint based blood group classification.

However, medical and forensic studies provide much weaker evidence for this relationship. Bhan et al. studied 1,040 participants and found a statistical association between fingerprint patterns and blood groups. However, the authors stated that the result was not strong enough to predict an individual’s blood group. Another study reported a Cramér’s V value of 0.10, which suggests a very weak association. (Siddiqui et al., 2025\) 

This difference raises an important question. The high machine learning results may reflect a real signal in fingerprint images. However, they may also be affected by repeated images, very similar images, fingerprints from the same person, or other features of the dataset.

This project will compare two evaluation methods. The first will use a normal random image split. The second will use a group split that keeps related images together.

The main goal is to measure how much model performance remains after known forms of data leakage have been controlled. The project does not aim only to obtain the highest accuracy. It aims to provide a fair and reliable evaluation.

# **2\. Background and Motivation (Peter)**

Blood group testing normally requires a blood sample, testing materials, and trained staff. A reliable method that does not require blood collection could be useful in emergency care and areas with limited medical resources.

Because of this possible use, several researchers have applied convolutional neural networks (CNNs) to fingerprint images. These models usually classify fingerprints into eight blood group categories (Phadke et al., 2025; Tripathi et al., 2026):

1. A positive  
2. A negative  
3. B positive  
4. B negative  
5. AB positive  
6. AB negative  
7. O positive  
8. O negative

Phadke et al. (2025) used a CNN for this eight-class classification problem and reported an overall accuracy of 88%. Tripathi et al. (2026) later reported a training accuracy of 87.5% and a validation accuracy of 85.5% using a deep learning model.

However, research based on traditional fingerprint patterns does not show the same level of predictive power. Bhan et al. (2026) examined fingerprint patterns and blood groups from 1,040 participants. Their results suggested a statistical association, but the researchers stated that the findings were not strong enough to predict an individual’s blood group.

Similarly, Siddiqui et al. (2025) studied 130 participants and found no statistically significant relationship between dominant fingerprint patterns and ABO or Rh blood groups. The study reported a Cramér’s V value of 0.10, suggesting a very weak association between the variables (Siddiqui et al., 2025).

One possible reason for the difference between the machine learning results and the medical evidence is data leakage.

Data leakage happens when information from the test data becomes available during model development. Kapoor and Narayanan (2023) explained that leakage has affected machine-learning research across many scientific fields and can produce results that are much more positive than a model’s true performance.

Fingerprint datasets may be especially sensitive to this problem. One person can provide several fingerprint images. A dataset may also contain copies, edited versions, or highly similar images from the same finger.

If these related images appear in both the training and test sets, the model may recognise a particular image or person instead of learning a general relationship between fingerprint features and blood groups. This could lead to an overly optimistic estimate of performance (Kapoor & Narayanan, 2023).

# **3\. Problem Statement**

Many image classification projects divide data by randomly assigning individual images to training, validation and test sets.

However, fingerprint images may not be independent. A fingerprint dataset may contain:

* Exact copies of the same image  
* Resized or rotated copies  
* Cropped images  
* Images with different brightness levels  
* Several images from the same finger  
* Different fingerprints from the same person  
* Images collected during the same session  
* Images processed using the same method  
* Class specific preprocessing, compression or background patterns 

If related images appear in different parts of the dataset, the reported test performance might be too high.

A second problem is batch or acquisition confounding. Even if fingerprints from the same subject are keep in only one data split, different blood group classes may still contain systematic differences caused by scanners, collection sessions, preprocessing, image resolution, compression or background. A CNN model could learn these class-related artefacts instead of learning a biological relationship between fingerprint ridges and blood groups. Therefore, subject level separation alone cannot guarantee that all dataset confounding has been removed. 

The main dataset may also lack clear subject identity information. This makes it difficult to perform a confirmed subject based split.

The central problem of this project is therefore:   
**Does the high fingerprint blood group classification result remain when detectable relationships between images are controlled?**

# **4\. Project Aim**

The main aim of this project is to determine whether fingerprint blood group classification performance remains strong after known and detectable forms of data leakage are controlled.

The project also aims to:

1. Check the dataset for exact and near duplicate images.  
2. Identify fingerprints that may come from the same source, finger or subject.  
3. Compare a random image split with a group based split.  
4. test whether simple image features can predict the dataset labels.  
5. Examine whether batch or acquisition-related artefacts may be associated with blood group classes.   
6. Produce a clear and repeatable evaluation process.

# **5\. Research Questions**

**Research Question 1**  
***How much fingerprint blood group classification performance remains after exact duplicates, near duplicates and possible source or subject groups are separated?***

This question will be answered by comparing the same model under two data split methods.

**Research Question 2**  
***Is the remaining performance based on fingerprint ridge information, or can the labels be predicted from metadata, background features or possible batch/acquisition artefacts such as brightness, contrast, file size, image dimensions and image quality?***   
This question will be examined using metadata models, dataset audits, image masking and model interpretation. These experiments will also test whether class-specific acquisition or processing artefacts may explain part of the model performance. 

**Research Question 3 (Optional)**  
***Can the Rh factor be predicted more reliably than the complete eight blood group classes?***

This experiment will only be completed if enough time remains after the main experiments.

# **6\. Dataset (Zoe)**

## **6.1 Fingerprint Blood Group Classification Dataset**

## **The main dataset will be the Fingerprint Blood Group Classification Dataset uploaded to Kaggle by Nanubala Sravani (2026). The dataset provides fingerprint images with blood group class labels and is currently available through Kaggle.**

The eight target classes are:

1. A positive  
2. A negative  
3. B positive  
4. B negative  
5. AB positive  
6. AB negative  
7. O positive  
8. O negative

The dataset will be used for:

* Training the blood group model  
* Model validation  
* Model testing  
* Random image splitting  
* Leakage controlled splitting  
* Metadata experiments  
* Image masking experiments

The exact image count, image dimensions, file format and class size will be confirmed after the dataset has been downloaded.

For this course project, the labels provided by the dataset will be treated as the target blood group classes. The project will focus on the quality of the machine learning evaluation.

## **6.2 SOCOFing Dataset** 

The Sokoto Coventry Fingerprint Dataset, or SOCOFing, will be used as a control dataset.

SOCOFing contains 6,000 fingerprint images from 600 subjects. It includes labels for subject identity, gender, hand and finger name. It also contains altered versions of some fingerprint images. It does not provide blood group labels.

SOCOFing will therefore not be used to measure blood group prediction accuracy.

It will be used to:

* Test fingerprint matching  
* Test subject grouping  
* Compare predicted groups with real subject labels  
* Compare random image splitting with subject based splitting  
* Test whether the leakage detection method works correctly

## **6.3 Krishna GitHub Dataset (optional)**

A second public fingerprint blood group dataset is available from the Krishna111809 GitHub repository. The repository contains fingerprint images arranged into eight blood group folders and includes code for models such as ResNet, VGG16, AlexNet, and LeNet.

This dataset will not be used in the main experiments at the current stage. It will be kept as a backup data source and will only be considered if additional data or an external comparison is needed later in the project.

Before any use, the GitHub dataset will be compared with the main Sravani dataset using:

* Total image count  
* Number of images in each class  
* File names  
* SHA-256 hashes  
* Perceptual hashes  
* Image-similarity measures

This comparison is necessary to determine whether the two sources contain the same or closely related images. If substantial overlap is found, the GitHub dataset will be treated as another version of the main dataset rather than as an independent source.

Only if the GitHub dataset is later required and confirmed to be sufficiently independent will it be considered as a possible external test dataset. It will not be combined directly with the Nanubala Sravani (2026) dataset.

# **7\. Proposed Methodology (Andy)**

## **7.1 Create a Data Manifest**

Before model training, the team will create a data manifest.  
The manifest will contain one row for each image.

Possible fields include: **image\_path, blood\_group, width, height, file\_format, file\_size, sha256, brightness, contrast, image\_quality, group\_id**

The group\_id will be added after duplicate and similarity analysis.  
The manifest will allow the team to record where every image is placed and why.

## **7.2 Basic Dataset Audit**

The first analysis will examine the structure of the main dataset.  
Our team will check:

* Total number of images  
* Number of images in each class  
* Image dimensions  
* File formats  
* File sizes  
* Missing files  
* Damaged files  
* Brightness  
* Contrast  
* Background features  
* Filename patterns  
* Image quality

This stage will show whether the classes are balanced and whether a simple image features differences between blood group folders.

## **7.3 Early Low-Cost Control Experiments** 

Before attempting subject reconstruction with fingerprint matching tools, the project will first run several low-cost control experiments. These experiments may provide early evidence of dataset artefacts without requiring complex fingerprint matching.

The first experiment will be a metadata only baseline using features such as image width, height, file size, brightness, contrast, background intensity and image quality. If these simple features can predict blood group labels substantially above baseline, this would suggest that class-related information exists outside the fingerprint ridge patterns.

A label permutation test will also be performed early in the project. Blood group labels will be randomly shuffled and the model will be trained again. Performance should fall close to chance or the majority-class baseline. Unexpectedly high performance would suggest leakage, memorisation or an error in the evaluation pipeline.

These experiments will be completed before the more complex NBIS-based fingerprint matching stage.

## 

## **7.4 Exact Duplicate Detection**

A SHA 256 hash will be calculated for every image.  
If two images have the same SHA 256 value, they will be treated as exact copies.  
All exact copies will receive the same group\_id.  
Exact copies will not be allowed to appear in different training, validation or test sets.

## **7.5 Near Duplicate Detection**

Some related images may not have the same SHA 256 value because they have been: **Resized, Rotated, Cropped, Compressed, Brightened, Darkened, Saved in another format**

Possible methods for detecting these images include:

* Perceptual hashing  
* Structural Similarity Index  
* Image embedding similarity  
* Fingerprint feature matching

Images judged to be closely related will be placed in the same group.

## **7.6 Fingerprint Matching and Grouping**

The project will attempt to identify images that may come from the same finger or subject.

Possible methods include:

* NBIS MINDTCT  
* NBIS BOZORTH3  
* NFIQ quality scores  
* Fingerprint embeddings  
* Filename analysis  
* Image similarity

MINDTCT may be used to detect fingerprint minutiae, while BOZORTH3 may be used to compare minutiae between fingerprint images.

The results will help create a group\_id for related images.  
The project will not claim that all true subjects have been perfectly identified. The groups will represent the relationships that can be detected using the available methods.

## **7.7 Validate the Grouping Method Using SOCOFing**

The grouping method will first be tested on SOCOFing.

The procedure will be:

1. Hide the real subject labels.  
2. Apply the matching or grouping method.  
3. Create predicted subject groups.  
4. Compare the predicted groups with the real subject labels.  
5. Measure false matches and missed matches.

Possible measures include:

* Pairwise precision  
* Pairwise recall  
* Pairwise F1  
* Adjusted Rand Index

The performance of the grouping method on SOCOFing will determine how Protocol B is interpreted. If the method achieves reliable grouping performance, the predicted groups will be used to reduce possible subject-level leakage in the main dataset.

If the method performs poorly on SOCOFing, the project will not claim that subject level leakage has been successfully controlled. In this case, Protocol B will use only relationships that can be identified with greater confidence, such as exact duplicates, near duplicates and highly similar images.

The project would then report that subject-level leakage could not be reliably evaluated because the main dataset does not provide confirmed subject IDs. The remaining experiments, including metadata analysis, duplicate detection, label permutation and background tests, would still be used to evaluate other possible sources of dataset leakage and confounding.

# **8\. Evaluation Protocols**

## **8.1 Protocol A: Random Image Split**

The first protocol will use a normal random image split.

A possible division is:

* 70% training  
* 15% validation  
* 15% testing

The class proportions will remain approximately equal across the three sets.  
This protocol will represent the common image level evaluation method.  
Under this method, related images may appear in more than one dataset split.

## **8.2 Protocol B: Leakage Controlled Group Split**

The second protocol will divide the data using the most reliable group\_id available from the leakage detection process.  
All images in one group will appear in only one part of the dataset.

For example:

* Group 001 appears only in training.  
* Group 002 appears only in validation.  
* Group 003 appears only in testing.

The strength of the grouping evidence will be reported. If subject reconstruction is not sufficiently reliable on SOCOFing, Protocol B will not be described as a confirmed subject wise split. Instead, it will be interpreted as a leakage-controlled split based on reliably detected image relationships. 

In this project, leakage-control means that known and detectable sources of leakage are controlled. It does not mean that every unknown relationship or source of confounding in the dataset has been completely removed. 

It does not mean that every unknown relationship in the dataset has been completely removed.

**9\. Model Selection and Training**

## **9.1 Baseline Model**

A simple CNN may be used as the first model.  
This model will provide a basic result before transfer learning is applied.

## **9.2 Main Model**

The main model will be ResNet 18\.  
ResNet 18 is suitable because it:

* Has a manageable size  
* Can use transfer learning  
* Can be trained using limited computing resources  
* Is widely used for image classification  
* Is complex enough to learn fingerprint features

## **9.3 Fair Comparison**

Protocol A and Protocol B will use the same:

* Model architecture  
* Image size  
* Optimiser  
* Learning rate  
* Batch size  
* Number of epochs  
* Early stopping rule  
* Loss function  
* Class weights  
* Data augmentation

The main difference will be the data split. Data augmentation will only be applied after the split has been created. It will only be applied to the training set.  
This prevents an original image from appearing in one set while an augmented copy appears in another.

The main experiments will be repeated using multiple random seeds rather than relying on a single train-validation-test split. The initial plan is to use at least five seeds where computational resources allow.

For each protocol, the project will report the mean performance across runs together with variation and 95% confidence intervals. This will reduce the risk that the difference between Protocol A and Protocol B is caused mainly by an unusually easy or difficult data split.

# **10\. Control Experiments**

Some control experiments, especially the metadata-only baseline and label permutation test, will be run early in the project because they are relatively inexpensive and may quickly reveal problems in the dataset or evaluation pipeline. Other control experiments will be completed after the main model pipeline has been established.

## **10.1 Majority Class Baseline**

A simple baseline will always predict the largest blood group class.  
The CNN results will be compared with this baseline.

## **10.2 Metadata Only Baseline**

A simple classifier will be trained using image and file information without using the complete fingerprint image.

Possible features include: **Width, Height, File size, Brightness, Contrast, Background intensity, Image quality.** If this model performs well, the dataset may contain class related information outside the fingerprint ridges. 

If this model performs substantially above baseline, this would suggest that class-related information exists outside the fingerprint ridge patterns. Such information may reflect dataset, processing or acquisition artefacts. However, because the original acquisition metadata are unavailable, the project will not attempt to determine the exact source of these differences. This possibility will still be investigated even if subject level grouping is successful.

## **10.3 Label Permutation Test**

The blood group labels will be randomly shuffled.  
The model will then be trained again.  
Performance should fall close to the chance or majority class baseline.  
If performance remains high, this may suggest leakage or memorisation.

## **10.4 Ridge and Background Tests**

The project may create different versions of the images.  
One version will keep the main fingerprint ridge area and the other version will hide the centre of the fingerprint and keep only the background and borders.

The results will help show whether the model depends on fingerprint ridges or other image features.

## **10.5 Grad CAM**

Grad CAM may be used to show which image areas influence the model.

It may show whether the model focuses on:

* Fingerprint ridges  
* Fingerprint centres  
* Borders  
* Background areas  
* Noise  
* Sensor marks

The Grad CAM will only be treated as supporting evidence. It will not be used as the main proof that the model has learned a biological relationship.

# **11\. Evaluation Measures**

The project will not use accuracy as the only evaluation measure.  
The report will compare: 

| Experiment | Accuracy | Balanced Accuracy | Macro F1 |
| :---- | :---- | :---- | :---- |
| Majority class baseline  |  |  |  |
| Metadata only baseline  |  |  |  |
| Random image split  |  |  |  |
| Leakage controlled split  |  |  |  |
| Shuffled label test  |  |  |  |
| Ridge only test  |  |  |  |
| Background only test  |  |  |  |

The main result will be the difference between the random image split and the leakage controlled group split. For experiments repeated across multiple seeds, results will be reported using the mean, standard deviation or variation across runs, and 95% confidence intervals where appropriate.

Therefore, the comparison between Protocol A and Protocol B will focus on both the average performance difference and the uncertainty around that difference, rather than comparing two single accuracy values.

# **12\. Expected Results**

## **Outcome 1: The Performance Does Not Survive**

The model may achieve high performance under a random image split, but its performance may drop significantly after duplicate images, related fingerprints and other possible sources of leakage are controlled.

It is also possible that the project will not reproduce the high accuracy reported in earlier studies under either evaluation protocol.

These results would suggest that previous performance may have been affected by duplicated images, subject information, batch/acquisition confounding, or other dataset artefacts. In this case, the project would show that fingerprint blood group classification results require stricter evaluation before they can be trusted.

## **Outcome 2: The Performance Survives**

The model may continue to perform clearly above the majority class, metadata and shuffled label baselines after leakage controls are applied.

This would suggest that fingerprint images may contain a stronger blood group signal than current medical evidence indicates, although unknown acquisition or dataset confounds could still remain. However, the result would still need to be tested using an independent dataset before broader biological or medical conclusions could be made.

There are two main possible outcomes for this project, and both would be meaningful. In both cases, the project would produce a useful result. The goal is not to achieve the highest possible accuracy. The goal is to provide a reliable and fair evaluation of fingerprint blood group classification.

# **13\. Project Significance**

The project will provide a more careful evaluation of fingerprint blood group classification.

Its main contribution will be the comparison between:

* A random image split  
* A leakage controlled group split

The project may also create a reusable process for:

* Detecting duplicate biometric images  
* Finding near duplicate images  
* Grouping related fingerprints  
* Checking metadata signals  
* Testing machine learning results for leakage

These methods may also be useful for other biometric and medical image projects.

# **14\. Technique Requirements**

The project is expected to use:

* Python  
* PyTorch  
* Sklearn   
* OpenCV  
* ImageHash  
* NBIS tools

# **15\. Risks and Responses (Litong)**

**Risk 1: Missing Subject Labels**  
The main dataset may not contain subject IDs.

Response: The project will use duplicate detection, image similarity and fingerprint matching to create estimated groups.

**Risk 2: Poor Fingerprint Matching**  
Some images may have low quality and may be difficult to match.

Response: The matching method will first be tested on SOCOFing using known subject identities as ground truth. If the method performs reliably, predicted groups may be used to reduce subject level leakage in the main dataset. If it performs poorly, the project will not claim successful subject reconstruction. Protocol B will then rely on more reliable relationships such as exact duplicates, near duplicates and highly similar images, while subject level leakage will be reported as an unresolved limitation.

**Risk 3: Unknown Acquisition or Dataset Confounding**  
The main dataset does not provide detailed information about the original scanners, collection sessions or preprocessing procedures. Therefore, systematic acquisition differences between blood group classes cannot be directly identified or ruled out.

Response: The project will examine observable image properties such as image dimensions, file size, brightness, contrast, background and image quality. Metadata-only and ridge/background experiments will test whether non-ridge information is associated with the class labels. However, the exact source of any detected artefacts will not be claimed unless supported by the available data.  
**Risk 4: Uneven Class Sizes**  
Some blood groups may contain more images than others.

Response: Balanced Accuracy, Macro F1 and recall for each class will be reported. Class weights may also be used.

**Risk 5: Low Model Performance**  
The model may fail to reproduce earlier results.

Response: It's still a valid outcome because the project focus on reliable evaluation rather than maximum accuracy.

# **16\. Ethical Considerations**

The project will use public datasets and will not collect new fingerprints or blood samples.

Fingerprints are biometric data. Therefore, the project will:

* Follow dataset usage conditions  
* Store the data securely  
* Avoid identifying real individuals  
* Use the data only for academic research  
* Report limitations clearly  
* Avoid making clinical recommendations

The model will be treated as an experimental classifier. It will not be presented as a replacement for medical blood testing.

# **17\. Outputs**

The project plans to produce:

1. A complete data manifest.  
2. A report on class balance and image properties.  
3. A duplicate and near duplicate analysis.  
4. A fingerprint grouping method tested on SOCOFing, together with an assessment of whether subject reconstruction is reliable for Protocol B.  
5. A model using a random image split.  
6. A model using a leakage controlled split.  
7. Majority class, metadata and shuffled label baselines.  
8. Ridge and background experiments.  
9. Accuracy, Balanced Accuracy, Macro F1, confusion matrices, multi-seed averages and confidence intervals.  
10. A comparison between the two main protocols.  
11. Reproducible project code.  
12. A final report and presentation.

# **References**

Bhan, S., Singh, T. S., Sandhu, S., Rahman, S. S., and Sarma, M. P. (2026). *Forensic examination to determine the correlation between fingerprint patterns and blood groups in the population of Assam*. Scientific Reports, 16, 10845\.  
[https://www.researchgate.net/publication/403221586\_Forensic\_examination\_to\_determine\_the\_correlation\_between\_fingerprint\_patterns\_and\_blood\_groups\_in\_the\_population\_of\_Assam](https://www.researchgate.net/publication/403221586_Forensic_examination_to_determine_the_correlation_between_fingerprint_patterns_and_blood_groups_in_the_population_of_Assam)

Kapoor, S., and Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9).  
[https://www.sciencedirect.com/science/article/pii/S2666389923001599](https://www.sciencedirect.com/science/article/pii/S2666389923001599)

Paudel et al. (2025). *A dermatoglyphic study of primary fingerprints pattern in relation to gender and blood group among residents of Kathmandu Valley, Nepal*. IET Biometrics.  
[https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/bme2/9993120](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/bme2/9993120)

Phadke, M., Raut, A., Sawant, C., Ramekar, G., and Badhan, S. (2025). Fingerprint based blood group detection using CNN. In *Proceedings of the International Conference on Wireless Communication*. Springer.  
[https://doi.org/10.1007/978-981-95-3616-0\_21](https://doi.org/10.1007/978-981-95-3616-0_21)

Shehu, Y. I., Ruiz-Garcia, A., Palade, V., and James, A. (2018). Sokoto Coventry Fingerprint Dataset. arXiv:1807.10609. [https://www.kaggle.com/datasets/ruizgara/socofing](https://www.kaggle.com/datasets/ruizgara/socofing)

Nanubala Sravani. (2026, July 6). *Fingerprint Blood Group Classification Dataset*. Kaggledatasets. [https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset](https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset)

Tripathi, A., Balpande, S., Chowdari, P.T., Basha, P.M., Sai, B.D., Chethan, P. (2026). Deep Learning-Based Blood Group Prediction Using Fingerprint Biometrics. In: Pundir, A.K.S., Yadav, A., Domínguez, R.M., Das, S. (eds) Recent Trends in Communication and Intelligent Systems. ICRTCIS 2025\. Algorithms for Intelligent Systems. Springer, Cham. [https://doi.org/10.1007/978-3-032-09832-0\_3](https://doi.org/10.1007/978-3-032-09832-0_3)

Siddiqui, A. R. O., Kumar, D. S., and Jyothi, Ch. (2025). *Association Between Fingerprint Patterns and ABO (±Rh) Blood Groups.* [https://impactfactor.org/PDF/IJCPR/17/IJCPR%2CVol17%2CIssue9%2CArticle253.pdf](https://impactfactor.org/PDF/IJCPR/17/IJCPR%2CVol17%2CIssue9%2CArticle253.pdf)

krishna111809. (2026). *fingerprint-based-blood-group-detection/dataset/dataset\_blood\_group at main · krishna111809/fingerprint-based-blood-group-detection*. GitHub. [https://github.com/krishna111809/fingerprint-based-blood-group-detection/tree/main/dataset/dataset\_blood\_group](https://github.com/krishna111809/fingerprint-based-blood-group-detection/tree/main/dataset/dataset_blood_group)

