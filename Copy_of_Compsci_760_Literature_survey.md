# **Does the Fingerprint-to-Blood-Group Result Survive a Leakage-Controlled Evaluation?** 

\\author{  
\\IEEEauthorblockN{Andy Guo, Litong Chen, Zoe Wang, Xia Yang, and Xiting Li}  
\\IEEEauthorblockA{  
Department of Computer Science\\\\  
The University of Auckland\\\\  
Auckland, New Zealand\\\\  
\\{jguo404, lche940, xwan414, xyia728, qli237\\}@aucklanduni.ac.nz  
}  
}

# **Abstract**

Fingerprint based blood group classification has received attention as a possible noninvasive method for predicting ABO and Rh blood groups. Recent machine learning research has reported high classification accuracy, while biological and forensic studies have generally found weak or inconsistent relationships between fingerprint patterns and blood groups. This survey examines this conflict by reviewing research on biological associations, fingerprint recognition, machine learning classification, data leakage, and model interpretation. The methods and evaluation procedures used in the selected studies are critically compared, with particular attention given to image relationships, subject information, data splitting, and dataset artefacts. The review finds that existing machine learning results do not clearly establish whether classification performance comes from genuine fingerprint characteristics. Duplicate images, related fingerprints, subject characteristics, and image backgrounds may provide alternative explanations. A reliable evaluation should therefore compare a random image split with a group based split that keeps detected related images together. Duplicate detection, fingerprint matching, control experiments, multiple evaluation metrics, and Grad CAM can provide further evidence about the information used by the model. The main research gap is whether the reported performance remains after detectable image relationships and dataset artefacts are controlled.

# **Keywords**

# **I. Introduction**

Blood group identification is required in blood transfusion, organ transplantation, and some emergency medical procedures. Conventional blood typing is performed by collecting and testing a blood sample. Although this method is reliable, suitable equipment, testing materials, and trained staff are required. Researchers have therefore considered whether fingerprints could provide information about a person’s blood group \[15\]. Fingerprints are suitable for biometric analysis because their ridge patterns are formed before birth and remain relatively stable throughout life. General fingerprint patterns are commonly divided into loops, whorls, and arches. More detailed features, such as ridge endings and bifurcations, are known as minutiae and have been widely used for personal identification \[1\], \[2\]. Recent advances in deep learning have allowed detailed visual information to be learned directly from fingerprint images. This has created interest in using fingerprints for tasks beyond identity recognition, including the prediction of ABO and Rh blood groups. If reliable prediction were possible, it might support rapid or noninvasive screening. However, such a system could not replace clinical blood testing without strong evidence from independent and carefully controlled studies.

The existing evidence presents an important conflict. Biological and forensic studies have compared the frequencies of loops, whorls, and arches across different blood groups. Some statistically significant relationships have been reported \[7\], \[10\], \[11\]. However, the findings have not been consistent across populations or blood group categories. Their predictive strength has also been limited. These studies suggest that fingerprint patterns and blood groups may be associated at the population level, but they do not support accurate prediction for an individual. Recent machine learning research presents a different picture. Phadke et al. \[15\] trained a convolutional neural network using more than 6,000 fingerprint images from eight blood group classes and reported a test accuracy of 88 percent. This result is difficult to explain using the weak relationships found in traditional fingerprint pattern studies. A convolutional neural network may detect detailed ridge or texture information that is not represented by broad pattern categories. However, high performance may also be produced by weaknesses in the dataset or evaluation procedure. Duplicate images, related fingerprints, or images from the same subject may appear in both the training and test sets. The model may also use background, resolution, compression, or image quality as shortcuts \[18\]. In these cases, the reported test accuracy may not represent performance on fingerprints from new subjects.  
Data leakage is therefore a central concern in this task. Leakage occurs when information that should be unavailable during training influences model development or evaluation \[12\]. Research in medical image classification has shown that improper separation of related images can substantially increase reported test performance \[17\]. This risk is especially relevant to fingerprint datasets because one person may provide several fingers, and the same finger may be captured or processed more than once. This survey examines whether the reported fingerprint based blood group classification performance can be considered reliable under these conditions. First, biological evidence is compared with recent machine learning results to distinguish population level association from individual prediction. Second, possible sources of leakage and dataset confounding are examined. Third, methods for detecting related fingerprint images and constructing a more reliable evaluation are reviewed. The central research question is: **How much of the fingerprint based blood group classification performance observed under a random image split remains under a leakage controlled group split?** The proposed study does not assume that leakage is present or that fingerprints cannot contain blood group information. Instead, both explanations are considered. It is expected that performance may decrease after detected image relationships are controlled. If strong performance remains and exceeds suitable baselines, the evidence for useful information within the fingerprint images would become stronger.

This survey reviews biological evidence, fingerprint recognition methods, machine learning evaluation, data leakage, and model interpretation. Studies were selected for their direct relevance to fingerprint based blood group prediction or reliable evaluation. Sections II to IV introduce the background and compare biological evidence with machine learning research. Sections V to VIII examine leakage risks, related image detection, evaluation principles, and model interpretation. Section IX identifies the main research gap, and Section X concludes the survey and presents the direction of the proposed study.

# **II. Background**

## **A. Blood Groups and Fingerprint Characteristics**

Human blood is divided into four main groups under the ABO system. These groups are A, B, AB, and O. Each group can also be Rh positive or Rh negative. This produces eight possible blood group categories.

Blood groups are normally identified through serological testing. A blood sample must be collected and examined using suitable testing materials. This process may be difficult in some emergencies or areas with limited medical resources \[15\]. For this reason, researchers have considered whether blood groups can be estimated through noninvasive methods.

Fingerprints are formed by ridges on the surface of the fingers. Three general patterns are commonly identified. These are loops, whorls, and arches \[7\], \[10\], \[11\]. The patterns are formed before birth and usually remain stable during a person’s life. However, the general pattern alone is not unique. Many people may have the same type of pattern.

More detailed information can be found in fingerprint ridges. Small local features are known as minutiae. Ridge endings and ridge bifurcations are common examples. In traditional fingerprint recognition, images are enhanced before minutiae are extracted and compared \[1\]. Their positions and directions can be used to decide whether two images may come from the same finger.

Deep learning has also been applied to this process. Darlow and Rosman \[2\] developed a neural network for minutiae extraction. Blais et al. \[3\] used deep learning to restore missing parts of fingerprint images. The restored images produced more matched minutiae points. Deshpande et al. \[4\] combined image enhancement, minutiae extraction, and matching in an automated system. These studies show that detailed features can be learned from fingerprint images. However, their purpose was personal identification rather than blood group prediction.

## **B. Fingerprints and Blood Groups**

Early studies of fingerprints and blood groups were based on dermatoglyphic patterns. The proportions of loops, whorls, and arches were compared across different blood groups. Bharadwaja et al. \[10\] reported that loops were the most common pattern in their sample. Whorls were less common, while arches had the lowest frequency.

Fayrouz et al. \[11\] studied 305 Libyan medical students. Ten fingerprints were collected from each participant. Loops represented 50.5 percent of the recorded patterns. Whorls represented 35.1 percent, while arches represented 14.4 percent. Differences were found in some blood groups and finger positions. However, significant results were not found in every comparison.

Bhan et al. \[7\] later studied 1,040 participants from Assam. A statistically significant association between fingerprint patterns and blood groups was reported. However, the authors stated that the result should not be used to predict an individual’s blood group. They also identified a limitation in their analysis. Several fingerprints from the same participant were treated as separate observations. These observations may not be fully independent.

These studies suggest that an association may exist at the population level. However, a significant association does not always provide useful predictive power. Statistical significance only suggests that the variables may not be fully independent in the sample. It does not show that a blood group can be accurately predicted for a new individual.

Machine learning studies use a different approach. An image is provided to a model, and one of the eight blood group categories is produced as the output. Phadke et al. \[15\] trained a convolutional neural network with more than 6,000 fingerprint images. An overall accuracy of 88 percent was reported. The dataset was divided into training, validation, and test sets.

The result reported by Phadke et al. \[15\] is much higher than the relationships found in traditional studies. However, accuracy alone cannot show what information was learned by the model. Fine ridge features may have been used. The model may also have learned repeated image content or other differences within the dataset. The data evaluation method must therefore be examined before the result can be considered reliable.

## **C. Evaluation Concepts**

A machine learning dataset is commonly separated into training, validation, and test sets. The training set is used to learn the model parameters. The validation set is used to guide decisions during model development. The test set is used to estimate performance on unseen data.

A random image split assigns each image to one of these sets. This method assumes that the images are independent. However, several fingerprint images may come from the same person or finger. A dataset may also contain copied, resized, rotated, or enhanced versions of the same image.

Data leakage occurs when information that should be unavailable during training influences the model or its evaluation \[12\]. If related fingerprint images appear in both the training and test sets, the model may recognise their shared content. In this situation, the test data are not fully independent.

Tampu et al. \[17\] demonstrated this problem in medical image classification. Improper separation of related images increased reported accuracy by between 5 and 30 percentage points. Chaibub Neto et al. \[16\] also showed that models may learn characteristics connected with individual subjects instead of the intended prediction target.

Keeping all images from one participant in the same data set can reduce this risk. However, other problems may remain. Different classes may contain differences in brightness, resolution, background, compression, or image quality. A model may use these simple features as shortcuts \[18\].

In this survey, a leakage controlled evaluation means that detectable relationships between images are identified and controlled. These may include exact copies, similar images, and fingerprints that may come from the same finger or participant. This term does not mean that every unknown source of leakage has been removed.

# **III. Biological and Forensic Evidence**

## **A. Evidence Reported Across Different Populations**

The biological evidence in this survey was selected from studies that directly examined the relationship between fingerprint patterns and ABO or Rh blood groups. The selected studies were published in different periods and involved populations from India and Libya. This allows their findings and research designs to be compared. These studies did not use complete fingerprint images as inputs to a prediction model. Instead, loops, whorls, and arches were counted and compared across blood groups.

Bharadwaja et al. \[10\] found that loops were the most common pattern in their sample. Whorls were the second most common, while arches were the least common. This order was generally observed across ABO and Rh groups. Some differences were reported between individual blood groups. For example, a higher proportion of loops was found in blood group A, while a higher proportion of whorls was found in blood group AB. However, the general distributions remained similar across the groups.

A similar pattern was reported by Fayrouz et al. \[11\]. Their study included 305 Libyan medical students and 3,050 fingerprint observations. Loops were again the most frequent pattern, followed by whorls and arches. Significant differences were found for blood group A when Rh status was considered. In contrast, the results for blood groups B, AB, and O were not statistically significant in the same comparison. This suggests that the relationship was not consistent across all blood groups.

Bhan et al. \[7\] provided more recent evidence from 1,040 participants in Assam. Loops were the most common pattern in this study, followed by whorls, arches, and accidental patterns. A significant overall association was reported between fingerprint patterns and blood groups. The Pearson chi square test produced a value of 34.761 and a p value of 0.030. The authors therefore concluded that the two variables were not fully independent. However, they also stated that the relationship was not strong enough for individual prediction.

Some common findings can be identified across these studies. Loops were consistently the most frequent pattern, while arches were much less common. Differences between blood groups were also observed in each study. However, the specific differences were not consistent. A pattern connected with one blood group in one population was not always found in another population. This may have been affected by population differences, blood group distributions, sample sizes, and methods of analysis.

## **B. Methodological Limitations**

The selected studies provide useful evidence at the population level. However, several limitations reduce their value for individual blood group prediction. First, the samples were collected from specific populations. Fayrouz et al. \[11\] used medical students from one Libyan university. Bhan et al. \[7\] studied young adults from Assam. The observed pattern distributions may therefore reflect local demographic or genetic characteristics. The findings cannot be assumed to apply to other populations without further validation.

Second, several fingerprints were collected from each participant. Fayrouz et al. \[11\] analysed 3,050 fingerprints from only 305 people. These fingerprint observations were treated as separate entries in several statistical comparisons. Bhan et al. \[7\] also recognised that treating fingerprints from the same person as independent observations could produce pseudoreplication. This may make the effective sample size appear larger than the number of independent participants.

Third, statistical significance should be separated from predictive performance. A small difference can become statistically significant when many observations are included. This does not mean that the difference is large enough to classify an individual. None of the three studies tested whether an unseen participant could be assigned to the correct one of eight blood group classes.

The studies also used broad fingerprint categories. Loops, whorls, and arches contain much less information than complete fingerprint images. A neural network may examine finer ridge and texture features. However, it may also learn image quality, background, or collection differences. The biological studies cannot determine which type of information is used by an image classification model.

Overall, the biological evidence suggests that fingerprint patterns and blood groups may have a weak or population dependent association. It does not provide direct support for accurate blood group prediction at the individual level. This creates a clear difference between traditional forensic evidence and the high accuracy reported by recent machine learning studies. The methods used in those machine learning studies must therefore be examined before their results can be compared with the biological evidence.

# **IV. Machine Learning Methods and Reported Performance**

## **A. Fingerprint Feature Extraction and Recognition**

Machine learning has been widely used for fingerprint recognition. However, most of the selected studies do not predict blood groups. They focus on image enhancement, feature extraction, or identity matching. These studies were included because their methods can help to detect related fingerprint images. They also show which fingerprint features may be learned by a model.  
Ali et al. \[1\] used a traditional processing pipeline. Fingerprint images were enhanced, converted into binary form, and reduced to thin ridge structures. Ridge endings and bifurcations were then extracted. Several machine learning methods were compared for fingerprint recognition. This approach depends on features that are designed before classification.  
Later studies used deep learning to reduce the need for manual feature design. Darlow and Rosman \[2\] treated minutiae extraction as a learning problem. Their MENet model learned to locate minutiae directly from fingerprint images. Blais et al. \[3\] used convolutional networks to restore missing parts of partial fingerprints. Their best method increased the number of matched minutiae points by 36.94 percent. However, only 35 test images were selected for the detailed matching evaluation. The result is useful, but the small evaluation set limits its strength.  
Image quality has also been addressed through different enhancement methods. Deshpande et al. \[4\] combined deep convolutional enhancement with frequency based filtering. Strong identification results were reported on FVC and NIST fingerprint datasets. Muhammed and Pais \[5\] used super resolution to improve low resolution fingerprint images. Singh et al. \[6\] combined minutiae matching with a fuzzy artificial neural network. Husain et al. \[8\] combined a fully connected convolutional network with improved SIFT features. These methods differ in design, but they share the same aim. A clearer and more stable fingerprint representation is produced before matching.  
The main advantage of minutiae methods is that the matching process is connected with recognised fingerprint structures. However, their performance can be affected by poor image quality, missing ridges, and incorrect minutiae. Deep learning methods can learn more complex representations. They may perform better when images are incomplete or noisy. Their learned features are also more difficult to interpret. In the present project, these recognition methods are relevant to image grouping. They may help identify images that come from the same finger or source. They do not provide evidence that blood groups can be predicted from fingerprints.

## **B. Fingerprint Based Blood Group Classification**

Phadke et al. \[15\] directly examined blood group classification from fingerprint images. Their public dataset contained more than 6,000 images across eight blood group classes. The original class sizes were unequal. For example, the A negative class contained 1,009 images, while the A positive class contained 565 images.  
A convolutional neural network with convolution, pooling, and dropout layers was used. The images were converted to grayscale and processed through noise removal, edge detection, and segmentation. The data were divided into 70 percent for training, 15 percent for validation, and 15 percent for testing. Training was conducted for up to 50 epochs. Early stopping and a reduced learning rate were used to limit overfitting.  
An overall test accuracy of 88 percent was reported \[15\]. The reported macro precision and macro recall were both 89 percent, while the macro F1 score was 88 percent. The confusion matrix showed strong results for all eight classes. These results suggest that the model found information that was associated with the dataset labels.  
Several strengths can be identified. Separate training, validation, and test sets were used. Class imbalance was recognised, and performance was not reported through accuracy alone. Precision, recall, F1 score, and a confusion matrix were also provided. These choices give a clearer view of class performance.  
However, important details were not reported. Random oversampling was used to balance the classes by duplicating images. The paper did not clearly state whether duplication was performed before or after the data split. If duplicated images were placed in different sets, direct leakage may have occurred. The paper also did not report subject identities or explain whether several images came from the same person. It is therefore unclear whether the test set contained fingerprints related to the training data.  
The study used one public dataset and did not include an external test set. Scanner information, collection sessions, and image processing history were not described. Repeated experiments with different random seeds were also not reported. As a result, the variation caused by the random split cannot be assessed.  
The reported accuracy is much higher than the predictive evidence found in traditional dermatoglyphic studies. This difference may reflect useful information in fine fingerprint ridges. It may also reflect repeated images, subject information, or other dataset features. The existing result cannot distinguish between these explanations. A comparison between a random image split and a split that keeps related images together is therefore needed.

# **V. Data Leakage and Alternative Explanations**

## **A. Leakage From Related Images**

Phadke et al. \[15\] showed that a convolutional neural network could predict the labels in their dataset with high accuracy. However, the result does not show which information was used. One possible explanation is data leakage. Kapoor and Narayanan \[12\] define leakage as a situation in which information that should be unavailable is used during model development or evaluation. Their survey found that this problem had produced overly positive conclusions in many scientific fields.  
Fingerprint datasets may contain several forms of related data. An exact copy may appear under a different file name. An image may also be resized, rotated, cropped, compressed, or adjusted in brightness. These files are not exact copies, but they may contain almost the same fingerprint information. If related images are placed in different data sets, the model is tested on content that it has already seen.  
Data augmentation can create a similar problem when it is performed before splitting. An original image may enter the training set, while an augmented version enters the test set. For a valid evaluation, the original data should be divided first. Augmentation should then be applied only to the training set.  
Random oversampling also requires careful handling. Phadke et al. \[15\] balanced the eight classes by duplicating images. However, the order of oversampling and data splitting was not clearly reported. If duplication was completed first, identical files could have entered both training and testing data. This does not prove that leakage occurred, but the risk cannot be excluded.

## **B. Subject and Finger Information**

Even when exact copies are removed, fingerprints from the same person may remain related. A person can provide ten different fingerprints. The images are different, but they may share subject characteristics, collection conditions, and processing history. The same finger may also be captured several times.  
Chaibub Neto et al. \[16\] examined the influence of subject characteristics in machine learning based diagnostic systems. Their results showed that a model may learn information connected with individual subjects. This can improve test performance when records from the same people appear in both training and testing data. However, the result may not generalise to new subjects.  
This issue is important for blood group prediction. All fingerprints from one person have the same blood group label. If several fingers from the same person are divided across the data sets, the model may identify the person rather than a general blood group pattern. A high score could then be obtained without learning a relationship that applies to unseen people.  
A split based on subject identity is normally preferred when several observations are available from each person. All images from one participant should be kept together. However, the public blood group dataset used by Phadke et al. \[15\] does not provide clear subject identifiers. Image similarity and fingerprint matching may be used to estimate groups, but these estimated groups may contain errors.

## **C. Dataset Confounding and Shortcut Learning**

Leakage is not the only possible explanation. A model can also learn differences created during data collection. Blood group folders may contain images produced with different scanners, settings, or processing methods. Differences may also be present in image size, contrast, compression, borders, and backgrounds. If these features are associated with particular labels, they can support classification.  
Geirhos et al. \[18\] describe this behaviour as shortcut learning. A shortcut is a simple decision rule that performs well within one dataset but fails when the testing conditions change. The feature may be predictive in the available data even though it is unrelated to the intended biological task.  
Tampu et al. \[17\] provide evidence of this risk in medical image classification. Their study used optical coherence tomography images. Related images from the same medical volume were sometimes divided across training and testing sets. Improper splitting increased accuracy by between 5 and 30 percentage points.  
The same increase cannot be assumed for fingerprint blood group classification because the tasks are different. However, the study shows why random splitting may be unreliable when images are related. It also shows that the problem can remain hidden when only final accuracy is reported.  
These risks provide alternative explanations for the difference between biological studies and machine learning results. The model may have learned a true signal from fingerprint ridges. It may also have learned repeated images, subject characteristics, or dataset artefacts. These explanations cannot be separated through one random split. Additional checks and a controlled evaluation are therefore required.

# **VI. Detecting Related Fingerprint Images**

To control the leakage risks discussed in Section V, relationships between fingerprint images must first be identified. Exact and near-duplicate relationships are considered first, followed by fingerprint matching to identify likely same-finger relationships and construct groups for leakage-controlled splitting. Finally, SOCOFing is used as a labelled reference dataset to evaluate the reliability of the matching and grouping procedure.

## **A. Exact and Near-Duplicate Detection**

Exact duplicate files can be identified using cryptographic hash functions such as SHA-256 \[20\]. Because the resulting digest is determined by the file contents, renaming a file does not change its hash, so the file can still be identified as an exact duplicate. The limitation is that even small modifications to an image, such as resizing, rotation, cropping, or compression, change its binary contents and are therefore highly likely to produce a different digest. As specified by NIST, any change to the input message will, with very high probability, result in a different message digest \[20\].

File hashing alone is therefore insufficient for detecting near-duplicate images that remain visually similar after modification. Perceptual hashing addresses this limitation by generating representations based on visual content, allowing images that remain similar after certain transformations to produce similar hash values and be compared using a distance or similarity measure \[21\]. However, its robustness is not consistent across all transformations. Zauner evaluated perceptual hash functions under transformations including resizing, JPEG compression, and rotation, and showed that performance varies with both the type and extent of the transformation. Perceptual hashing also requires an application-specific matching threshold, creating a trade-off between false matches and missed matches \[21\].

SSIM provides a complementary measure of image similarity. Originally developed for full-reference image quality assessment, it compares structural information between a reference image and a distorted image \[22\]. In this context, SSIM can be used to examine whether candidate near-duplicate images still preserve similar structural information. However, it assumes that the images are sufficiently comparable and aligned, which makes it more suitable as a supporting similarity measure than as a standalone method for detecting all forms of near duplication.

Overall, cryptographic hashing is well suited to detecting exact duplicate files, while perceptual hashing and SSIM can identify some modified images that remain visually similar. Since these methods respond differently to image transformations and have different limitations, using them together can provide broader coverage of possible exact and near-duplicate relationships in the dataset.

## **B. Fingerprint Matching and Group Construction**

The image-similarity methods described above can identify exact or visually similar images, but visual similarity alone cannot determine whether two fingerprint images come from the same finger. As discussed in Section V, when several fingerprint images come from the same person, the ideal approach is to keep all of that person’s images within the same data split to avoid subject-level leakage. However, when reliable subject identifiers are unavailable, complete subject-level grouping cannot be confirmed. Fingerprint matching can therefore be used to identify a narrower but still important relationship: whether different images are likely to originate from the same finger.

Traditional fingerprint matching is commonly based on minutiae, with ridge endings and ridge bifurcations being the most common local features \[1\], \[23\]. Ali et al. describe a conventional fingerprint recognition process in which images are first preprocessed and minutiae features are then extracted for recognition \[1\]. In NIST’s NBIS software, MINDTCT detects fingerprint minutiae, while BOZORTH3 uses these minutiae to determine whether two fingerprints are likely to come from the same finger \[23\]. The resulting pairwise matches can then be used to construct groups of images that are likely to share the same finger identity.

During leakage-controlled evaluation, all images within the same group can be kept in the same data split, reducing the risk that same-finger images appear in both the training and test sets. However, this does not provide complete subject-level grouping, because different fingers from the same person may still be assigned to different groups. Fingerprint matching is also imperfect. A false match may place images from different fingers in the same group, while a missed match may leave true same-finger images in different groups and allow leakage to remain. The reliability of the grouping procedure should therefore be evaluated before the groups are used for leakage-controlled splitting.

## **C. Validation With SOCOFing**

SOCOFing can be used as a labelled reference dataset to evaluate the matching and grouping procedure described above. It contains 6,000 original fingerprint images from 600 subjects, with subject, hand, and finger information included in the file names \[19\]. These labels can be combined to construct ground-truth finger identities. SOCOFing also provides altered versions of the original fingerprints, allowing the grouping method to be tested on known same-finger relationships under different image modifications \[19\]. Unlike the target blood-group dataset, SOCOFing therefore provides ground-truth information that can be used to directly assess whether the grouping procedure is working as intended. 

For validation, the grouping method can be applied without using the ground-truth labels, and the predicted groups can then be compared with the true finger identities. Pairwise precision and recall measure how accurately same-finger image pairs are recovered, while F1 summarises the balance between the two. The Adjusted Rand Index (ARI) provides a broader measure of agreement between the predicted grouping and the ground-truth grouping while correcting for agreement expected by chance.

However, SOCOFing does not contain blood-group labels, so it can only be used to evaluate the fingerprint matching and grouping procedure rather than blood-group classification itself \[19\]. Good same-finger grouping performance on SOCOFing also does not demonstrate complete subject-level separation in the target dataset, because different fingers from the same person may still remain unlinked by fingerprint matching.

# **VII. Principles for a Reliable Evaluation**

A reliable evaluation depends on more than a single accuracy value. It also requires an appropriate data-splitting strategy, a fair comparison between evaluation protocols, and checks for possible leakage or dataset artefacts. This section therefore outlines the main principles used to compare the random image split with the group-based split, together with the baselines, control experiments, and metrics used to assess model performance and stability.

## **A. Random Image Split and Group-Based Split** 

The evaluation compares two ways of dividing the dataset. The first is a random image split, in which each image is assigned independently to the training, validation, or test set without considering possible relationships between images. This approach effectively treats the images as independent samples. If the dataset contains duplicates, near-duplicates, or other related fingerprint images, random splitting may place them in different sets. This can expose the model to test images that are highly related to its training data and lead to overly optimistic performance estimates \[12\], \[17\]. 

The second approach is a group-based split. Relationships detected in Section VI are used to construct groups, and all images within the same group are kept in the same data split. This prevents detected duplicate, near-duplicate, and same-finger relationships from crossing between the training and test sets, thereby reducing leakage from these known relationships.

However, a group-based split can only control relationships that have been detected. It does not guarantee complete subject-level separation or eliminate every possible source of leakage. It should therefore be viewed as a stricter leakage-control strategy rather than a complete solution to the leakage problem.

## **B. Fair Comparison Between Evaluation Protocols**

The purpose of comparing the random image split with the group-based split is to isolate the effect of the splitting strategy on model performance. The two protocols should therefore differ only in how the data are divided, while preprocessing, model architecture, training procedure, and evaluation settings remain the same. If data augmentation is used, it should be applied only to the training set after the data have been split. Otherwise, augmented versions of the same original image may appear in different splits and introduce leakage \[17\].

Under these controlled conditions, a substantial drop in performance under the group-based split would suggest that related images contributed to the higher performance observed under the random split. If performance remains similar, this would suggest that the detected relationships are unlikely to be the main reason for the high performance. However, this alone would not show that the model is learning genuine fingerprint information, so additional control experiments are still needed to examine other possible explanations.

## **C. Baselines and Control Experiments**

Comparing the random image split with the group-based split is still not enough to explain why a model achieves high performance, so additional baselines and control experiments are needed. First, a majority-class baseline provides a basic reference for determining whether the model clearly outperforms a trivial prediction rule.

Once the model performs above this basic baseline, the next question is whether that performance mainly comes from fingerprint ridge information or from other dataset cues. A metadata-only baseline uses simple information such as image size, file size, brightness, or contrast rather than the full fingerprint image. If this baseline also performs well, the dataset may contain non-ridge information related to the class labels that provides unintended shortcuts for the model \[18\]. Ridge-only and background-only experiments can further examine where the predictive information is concentrated. Strong background-only performance would suggest that non-ridge information contributes substantially to the classification. In contrast, if the ridge-only model retains much of the full-image performance, the useful information is more likely to be concentrated in the fingerprint region. However, this would still not establish a biological relationship with blood group.

A label permutation test can provide an additional sanity check. After the class labels are randomly shuffled, the intended relationship between the images and blood-group labels is removed, so model performance should fall toward chance level. If performance remains unexpectedly high, leakage or another evaluation problem may still be present \[17\]. Overall, these controls can help rule out some alternative explanations, but they cannot by themselves prove a true biological relationship between fingerprints and blood groups.

## **D. Evaluation Metrics and Reliability**

In addition to examining the possible sources of model performance, the results under both splitting protocols should be measured consistently and reliably. The eight blood-group classes in the dataset used by Phadke et al. are not evenly represented \[15\], so overall accuracy alone may not fully reflect model performance. In addition to accuracy, balanced accuracy and macro F1 should be reported to provide class-balanced summaries, while per-class recall and the confusion matrix can show whether errors are concentrated in particular blood-group classes.

Result stability is equally important. A single experimental run may be affected by the particular data split, so both the random image split and the group-based split should be repeated using several random seeds, with results reported as the mean and standard deviation. This helps show whether the performance difference between the two splitting strategies is consistent rather than caused by one particularly favourable split. Tampu et al. similarly used repeated cross-validation and reported mean ± standard deviation when comparing different splitting strategies \[17\].

# **VIII. Explainability and Model Interpretation**

High classification performance alone does not show which image features a model relies on. This section considers how Grad-CAM can be used to examine whether predictions are associated mainly with fingerprint ridge regions or with possible non-ridge cues, while recognising that visual explanations provide supporting rather than conclusive evidence.

## **A. Grad-CAM for Fingerprint Classification**

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a class-specific heatmap that highlights image regions associated with a model’s prediction \[13\]. It uses gradients from a target class to weight convolutional feature maps and produces a coarse localisation map showing which regions are more strongly associated with the class score. This allows predictions to be examined visually without changing or retraining the underlying model \[13\].

In fingerprint classification, Grad-CAM can help examine whether a prediction is mainly associated with the fingerprint ridge region or instead with the background, image borders, or other non-ridge regions. If the highlighted regions are concentrated mainly on such unintended areas, this would be consistent with the model relying on shortcut cues rather than only on fingerprint structure \[18\].

Grad-CAM can also be applied to models trained under the random image split and the group-based split. Comparing the heatmaps from the two protocols can help show whether the image regions associated with the predictions change after leakage control. For example, differences in the extent to which predictions are associated with ridge regions or non-ridge regions would provide supporting evidence that the two models may be relying on different types of information.

## **B. Limits of Visual Explanations**

Grad-CAM has important limitations when used to interpret fingerprint classification. First, it provides only coarse localisation. A heatmap can indicate whether a prediction is mainly associated with the fingerprint region, the background, or other broad parts of the image, but it cannot reliably identify individual ridges or minutiae \[13\]. It should therefore not be interpreted as a precise explanation of the fingerprint features used by the model.

More importantly, localisation does not establish a biological explanation. Even if a heatmap is concentrated on the fingerprint ridge region, this does not prove that ridge features are biologically related to blood group. The model may still be responding to image quality, contrast, processing differences, or other dataset-specific factors that occur within the fingerprint region itself.

Grad-CAM should therefore be treated as supporting evidence rather than direct proof of what the model has learned. It is most useful when interpreted together with the ridge-only and background-only control experiments described in Section VII, which provide a more direct test of whether predictive performance depends mainly on ridge information or on non-ridge information. Evidence from the visual explanations and the control experiments should therefore be considered together before drawing conclusions about the source of the model’s performance.

# **IX. Critical Synthesis and Research Gap**

Taken together, the literature reveals a clear gap between the high performance reported by recent machine-learning studies and the evidence needed to determine whether that performance is reliable.

## **A. Conflict Across the Literature**

Traditional dermatoglyphic studies and recent machine-learning studies provide noticeably different evidence about the relationship between fingerprints and blood groups. Traditional studies have reported some population-level associations between fingerprint patterns and blood groups, but the specific pattern differences are not consistent across studies or populations, and they do not establish reliable individual-level blood-group prediction \[7\], \[10\], \[11\]. In contrast, Phadke et al. \[15\] classified blood groups from full fingerprint images and reported an overall accuracy of 88%, highlighting a clear difference between the two bodies of evidence.

However, these two lines of research do not use exactly the same information. Traditional studies mainly compare broad fingerprint patterns such as loops, whorls, and arches, whereas CNNs can use finer ridge, texture, and other visual features from the full image. The higher machine-learning performance may therefore reflect information that is not captured by traditional pattern analysis, but it may also result from dataset artefacts or other unintended shortcuts \[18\]. The difference between the two literatures does not necessarily represent a contradiction in the biological evidence itself. Instead, it shows that a stricter evaluation is still needed to determine what is driving the high classification performance.

## **B. Limits of Current Evidence**

The current machine-learning evidence still has several important limitations. Phadke et al. \[15\] used a public fingerprint dataset, but the paper does not report clear subject identifiers, image acquisition conditions, scanner information, or how the blood-group labels were verified. The study also used random oversampling to balance the classes, but the order of oversampling and data splitting was not clearly described, and no duplicate audit was reported. As a result, the possible effects of duplicate images, related images, or subject overlap cannot be ruled out.

These issues directly affect how the reported performance should be interpreted. Previous studies have shown that when related samples or records from the same subject appear in both the training and test sets, model performance can be overestimated \[12\], \[16\], \[17\]. Phadke et al. \[15\] also did not use an independently collected external test set or report experiments across multiple random seeds, and no metadata or background controls were included. The current evidence therefore cannot determine whether the high classification performance mainly comes from predictive information in the fingerprint images or from leakage and dataset artefacts.

## **C. Main Research Gap**

The limitations discussed above point to a clear research gap: the studies reviewed here have not established whether high fingerprint-based blood group classification performance remains when detectable image relationships and dataset artefacts are controlled. Two explanations for the reported results therefore remain possible. One is that fingerprint images still contain predictive information related to blood-group classification even under stricter leakage control. The other is that much of the reported performance depends on related images, subject-related information, or dataset artefacts and would decrease once these factors are controlled. Because the studies reviewed here have not applied the necessary leakage controls and artefact checks, the current evidence cannot distinguish between these two explanations.

## **D. Direction of the Proposed Study**

The proposed study directly compares a random image split with a group-based split constructed from the image relationships detected in Section VI. The same preprocessing, model architecture, training procedure, and evaluation settings are used under both protocols so that the effect of the splitting strategy can be examined more clearly. The main question is how much of the high performance observed under a random split remains under the stricter leakage-controlled split. A substantial drop would suggest that detected relationships between images contributed to the higher performance observed under the random split. If performance remains similar, these detected relationships are less likely to be the main explanation. However, similar performance would not by itself demonstrate subject-level generalisation or a biological relationship between fingerprints and blood groups. The control experiments in Section VII and the explainability analysis in Section VIII are therefore needed to examine other possible shortcuts and sources of predictive performance.

The study does not claim to remove every possible source of leakage. The grouping methods can only control relationships that can be detected, so unidentified subject-level relationships or other forms of leakage may still remain. SOCOFing is also not used as a blood-group classification test set because it does not contain blood-group labels \[19\]. Instead, it is used only to validate the fingerprint matching and grouping procedure described in Section VI. Within these limits, the study aims to test directly whether high fingerprint-based blood-group classification performance remains under a stricter leakage-controlled evaluation.

# **X. Conclusion**

This survey examined whether the high performance reported for fingerprint-based blood group classification is supported by sufficiently reliable evidence. Traditional dermatoglyphic studies have reported some population-level associations between fingerprint patterns and blood groups, but the specific findings vary across studies and populations and do not establish reliable individual-level blood-group prediction. In contrast, recent machine-learning work has reported much higher classification performance using full fingerprint images. This difference may reflect information in finer ridge, texture, or other image features that is not captured by broad fingerprint-pattern analysis. However, the literature on data leakage and shortcut learning shows that high performance can also arise when related samples cross data splits or when models exploit unintended dataset cues.

The current evidence therefore leaves an important uncertainty. Existing fingerprint-based blood-group classification results do not clearly establish whether the reported performance mainly comes from predictive information in the fingerprint images or is influenced by duplicate and related images, subject-related information, or dataset artefacts. The main research gap is whether high classification performance remains when detectable image relationships and potential non-ridge shortcuts are more carefully controlled.

The proposed study addresses this gap by comparing a random image split with a leakage-controlled group-based split constructed from detected duplicate, near-duplicate, and same-finger relationships. The two protocols will use the same preprocessing, model architecture, training procedure, and evaluation settings so that the effect of the splitting strategy can be examined more clearly. Metadata-only, ridge-only, and background-only controls, together with Grad-CAM analysis, will then be used to investigate possible sources of any remaining predictive performance. Because the grouping procedure can only control relationships that can be detected, the evaluation cannot guarantee complete subject-level separation or removal of every possible source of leakage.

The central research question is therefore: **how much of the classification performance observed under a random image split remains under a leakage-controlled group split?** A substantial decrease would suggest that detected image relationships contributed to the higher performance under the random split. If strong performance remains, this would provide stronger evidence that predictive information is present in the fingerprint images, although it would still not demonstrate a biological relationship between fingerprints and blood groups. Any broader biological or practical claim would require further validation using independently collected data.

# **Reference:**

\[1\] A. Ali, R. Khan, I. Ullah, A. D. Khan, and A. Munir, “Minutiae based automatic fingerprint recognition: Machine learning approaches,” *2015 IEEE International Conference on Computer and Information Technology; Ubiquitous Computing and Communications; Dependable, Autonomic and Secure Computing; Pervasive Intelligence and Computing*, Liverpool, U.K., 2015, pp. 1148–1153, doi: 10.1109/CIT/IUCC/DASC/PICOM.2015.171.

\[2\] L. N. Darlow and B. Rosman, “Fingerprint minutiae extraction using deep learning,” *2017 IEEE International Joint Conference on Biometrics (IJCB)*, Denver, CO, USA, 2017, pp. 22–30, doi: 10.1109/BTAS.2017.8272678.

\[3\] M.-A. Blais, A. Couturier, and M. A. Akhloufi, “Deep learning for partial fingerprint inpainting and recognition,” in *Image Analysis and Recognition*, Lecture Notes in Computer Science, vol. 12131\. Cham, Switzerland: Springer, 2020, pp. 223–232, doi: 10.1007/978-3-030-50347-5\_20.

\[4\] U. U. Deshpande, V. S. Malemath, S. M. Patil, and S. V. Chaugule, “End-to-end automated latent fingerprint identification with improved DCNN-FFT enhancement,” *Frontiers in Robotics and AI*, vol. 7, Art. no. 594412, Nov. 2020, doi: 10.3389/frobt.2020.594412.

\[5\] A. Muhammed and A. R. Pais, “A novel fingerprint image enhancement based on super resolution,” *2020 6th International Conference on Advanced Computing and Communication Systems (ICACCS)*, Coimbatore, India, 2020, pp. 165–170, doi: 10.1109/ICACCS48705.2020.9074196.

\[6\] S. P. Singh, D. K. Nishad, and S. Khalid, “Enhancing fingerprint identification using Fuzzy-ANN minutiae matching,” *Measurement: Sensors*, vol. 37, Art. no. 101809, 2025, doi: 10.1016/j.measen.2025.101809. 

\[7\] S. Bhan, T. S. Singh, S. Sandhu, S. S. Rahman, and M. P. Sarma, “Forensic examination to determine the correlation between fingerprint patterns and blood groups in the population of Assam,” *Scientific Reports*, vol. 16, Art. no. 10845, 2026, doi: 10.1038/s41598-026-42044-7. 

\[8\] S. O. Husain, R. A. Reddy, P. K. Pareek, S. Jagannathan, and M. Jyothi, “Robust fingerprint minutiae extraction and matching using fully connected deep convolutional neural network and improved SIFT,” *2024 International Conference on Data Science and Network Security (ICDSNS)*, Tiptur, India, 2024, doi: 10.1109/ICDSNS62112.2024.10691043.

\[9\] U. Nagamani, S. M. Tabassum, B. Jayasree, V. Prashanth, S. Srinath, and S. Srikanth, “Fingerprint recognition system based on artificial neural networks integrated with machine learning,” *ITM Web of Conferences*, vol. 79, Art. no. 01018, 2025, doi: 10.1051/itmconf/20257901018.

\[10\]Bharadwaja, A., Saraswat, P. K., Aggarwal, S. K., Banerji, P., & Bharadwaja, S. (2004). Pattern of finger-prints in different ABO blood groups. Journal of Indian Academy of Forensic Medicine, 26(1), 6-9.

\[11\]Fayrouz, I. N. E., Farida, N., & Irshad, A. H. (2012). Relation between fingerprints and different blood groups. Journal of Forensic and Legal Medicine, 19(1), 18–21. [https://doi.org/10.1016/j.jflm.2011.09.004](https://doi.org/10.1016/j.jflm.2011.09.004)

\[12\]Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804\. [https://doi.org/10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)

\[13\]Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV) (pp. 618–626). [https://doi.org/10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)

\[14\]​​Tucci, C., Della Greca, A., Tortora, G. *et al.* Explainable biometrics: a systematic literature review. *J Ambient Intell Human Comput* (2024). [https://doi.org/10.1007/s12652-024-04856-1](https://doi.org/10.1007/s12652-024-04856-1) 

\[15\] M. Phadke, A. Raut, C. Sawant, G. Ramekar, and S. Badhan, “Fingerprint based blood group detection using CNN,” in *Proceedings of International Conference on Wireless Communication (ICWiCom 2025\)*, Lecture Notes in Electrical Engineering, vol. 1499\. Singapore: Springer, 2025, pp. 241–250, doi: 10.1007/978-981-95-3616-0\_21.

\[16\] E. Chaibub Neto *et al*., “Detecting the impact of subject characteristics on machine learning-based diagnostic applications,” *npj Digital Medicine*, vol. 2, Art. no. 99, 2019, doi: 10.1038/s41746-019-0178-x.

\[17\] I. E. Tampu, A. Eklund, and N. Haj-Hosseini, “Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images,” *Scientific Data*, vol. 9, Art. no. 580, 2022, doi: 10.1038/s41597-022-01618-6.

\[18\] R. Geirhos *et al*., “Shortcut learning in deep neural networks,” *Nature Machine Intelligence*, vol. 2, pp. 665–673, 2020, doi: 10.1038/s42256-020-00257-z. 

\[19\] @misc{shehu2018socofing,  
  author    \= {Shehu, Yusuf I. and Ruiz-Garcia, Ariel and Palade, Vasile and James, Anne},  
  title     \= {Sokoto Coventry Fingerprint Dataset},  
  year      \= {2018},  
  eprint    \= {1807.10609},  
  archivePrefix \= {arXiv},  
  primaryClass  \= {cs.CV},  
  howpublished  \= {\\url{https://kaggle.com}}  
}

\[20\] NIST, "Secure Hash Standard (SHS)," FIPS PUB 180-4, National Institute of Standards and Technology, Gaithersburg, MD, USA, 2015, doi: 10.6028/NIST.FIPS.180-4.  
\[21\] C. Zauner, "Implementation and benchmarking of perceptual image hash functions," M.S. thesis, Dept. Embedded Syst. Eng., Univ. Appl. Sci. Hagenberg, Hagenberg, Austria, 2010\.  
\[22\] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, Apr. 2004, doi: 10.1109/TIP.2003.819861.

\[23\] C. I. Watson, M. D. Garris, E. Tabassi, C. L. Wilson, R. M. McCabe, S. Janet, and K. Ko, "User's guide to NIST biometric image software (NBIS)," NIST Interagency Report 7392, National Institute of Standards and Technology, Gaithersburg, MD, USA, 2007\.

# **Appendix (Optional)**

