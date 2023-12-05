# facial-emotion-recognition

Data
The Facial Emotion Recognition data has 12271 training and 
3068 testing images and relevant labels which indicates the 
facial emotion of the image. Image sizes are same, and it is 
100x100 pixels. There are 7 numeric labels between 1-7 equals 
that emotion labels which are ‘Surprise’, ‘Fear’, ‘Disgust’, 
‘Happiness’, ‘Sadness’, ‘Anger’, ‘Neutral’. Percentage 
distribution for the data labels is 38.9 for class 4(Happiness), 
20.6 for class 7(Neutral), 16.2 for class 5(Sadness), 10.5 for class 
1(Surprise), 5.8 for class 3{Disgust}, 5.7 for class 6(Anger) and 2.3 
for class 2(Fear). It is visualised below in figure 1. There is a slight Fig. 1. The label distribution of classes
imbalance on the training data, however oversampling techniques like SMOTE are not considered as there 
may not be significant improvement on the result metrics since labels have not important distribution 
differences.
The video data is an AVI type video which has 854x480 pixels and 4020 frames. Its length is 2 minutes and 
14 seconds. It is a movie trailer video which contains many people with different facial emotions. 
Implemented methods
3 different models are implemented for Facial Emotion Recognition problem, two of them are Support 
Vector Classifiers, one of them consists of SIFT (Scale Invariant Feature Transform) and the other one has 
HOG (Histogram of Gradients) feature description extractors to transform image data into proper data form 
which is suitable for data modelling. Support Vector Machine is significantly strong classification technique 
which uses kernel tricks to perform better predictions which leads to robust performance for image 
classification.[1] SIFT and HOG transformations are selected to prepare image data to modelling because 
they are among the most common image transformation techniques. HOG is considered as better extractor 
than SIFT, especially for SVM classifiers, according to relevant research.[2] The final model is Convolutional 
Neural Network (CNN) model which perform both image data transformation and modelling. 
To perform SIFT transformation, SIFT instance was created through Open CV library in Python. Then image 
data has been transformed into one channel data from RGB data, and then unsigned byte format. Finally,
‘detectandcompute’ function was used to collect image descriptors. After it, number of centroids were 
calculated, for this label number was multiplied with 10 as indicated in lab solutions. Mini batches are used 
to make faster the k-means performing. After k-means prediction on descriptors, the data was ready for 
modelling.
For HOG transformations, ‘skimage’ library was used. Image data were transformed into unsigned byte 
format just like SIFT. Unlike SIFT, HOG functions are more ready to use and makes more transformations 
inside automatically. After hog function extraction, the description data were collected into an array. It was 
ready for modelling.
After respective SIFT and HOG transformations for models, Support Vector Classifiers (SVC) were fitted. At 
that point, since the problem is multiclass classification problem contains multiple labels, 
‘OneVsRestClassifier’ was used to fit the training data. SVC classifiers are designed to model binary label 
datasets originally, therefore this approach has been used since it transformed the labels into binary 
manner to each other. For kernel selections of SVC, ‘rbf’ kernel was used for both models as it usually gives 
more successful results than other kernel types. [1] For evaluation purposes on the training side, cross 
validation technique was used. It separated the data 3 folds and evaluate it on the different selected fold 
each time. Accuracy metric was used to analyse the results. These settings were used for both SVM models.
After normal implementation, grid search was used for hyperparameter tuning. GridsearchCV function was 
used 3 stratified cross validation folds with shuffling. Hyperparameter options were 0.08, 0.01 and 0.8 for C 
value, and ‘poly’ and ‘rbf’ for kernel types. These settings were used for both SVM models.
For Convolutional Neural Network (CNN) model, the structure has 2 2D-convolutional layers with 7 kernel 
sizes and 5 paddings was used as the image dimensions are moderately big (100x100 pixels). Kernel sizes 
and padding were increased to that level as smaller values gave poor accuracy results. To avoid RAM 
crashing, batch sizes are set to 4. Batch normalization and maximum pooling with 5 kernel sizes and 2 
strides were selected to improve the performance of the model. Adam optimiser and Cross Entropy Loss 
functions were used as optimising and loss function calculation purposes. Adam optimiser was researched 
as reliable optimiser for similar research [2]. The model was trained with 200 epochs. 
For ‘EmotionRecognitionVideo’ function, the video was imported via OpenCV’s ‘VideoCapture’ function. 
Random 4 frame images were created with random number generator to perform emotion prediction. For 
prediction, the SVM model with HOG extractor which was specifically created for the ‘EmotionRecognition’ 
function was used with Viola Jones face detection function(haarcascade) after the frame data were 
transformed into one channel data from RGB data, and then unsigned byte format like SIFT. For HOG 
transformation, image pixel sizes were changed into 100x100 for a suitable and fast implementation.
Results
Training validation results for SVM model with SIFT extractor has mean accuracy score of 0.41 and mean 
fitting time of 33 seconds. Gridsearch hyperparameter search with stratified cross validation gave 
maximum of 0.408 accuracy score and 33.7 seconds of mean fitting time with the best combination.
Training validation results for SVM classifier with HOG descriptor has mean accuracy score of 0.623 and 
mean fitting time of 35.5 seconds. Gridsearch hyperparameter search with stratified cross validation gave 
maximum of 0.62 accuracy metric with the best combination. It also gave mean fitting time of 55.5
seconds.
Training validation results for CNN model has mean accuracy 
metric score of 0.66 and fitting time of 15 minutes. There is an 
emerging gap between training loss curve and validation loss 
curve after roughly 100th epochs as seen in figure 2.
On the test side, prediction performance results for SVM 
model with SIFT descriptor shows that the accuracy metric is 
0.35. Confusion matrix and classification report can be seen 
below in figure 3 and 4.
Fig. 2. Training and validation curves for CNN model
Fig. 3. Confusion matrix for SIFT+SVM model Fig. 4. Classification report for SIFT+SVM model
Prediction results for SVM model with HOG descriptor demonstrate that it has roughly 0.64 accuracy level 
on the unseen test dataset. Confusion matrix and classification report can be seen below in figure 5 and 6.
 Fig. 5. Confusion matrix for HOG+SVM model Fig. 6. Classification report for HOG+SVM model
Prediction results for CNN model indicates that it has 0.58 accuracy level on the unseen test dataset. 
Confusion matrix and classification report can be seen below in figure 7 and 8.
 Fig. 7. Confusion matrix for CNN model Fig. 8. Classification report for CNN model
The EmotionRecognitionVideo function face and emotion detection results can be seen with 4 random 
frames such as below in the figure 9. The prediction time was roughly 15 seconds.
Fig. 9. Random 4 images from the selected video data and different emotion labels with face detections
Discussion
The SIFT+SVM model prediction performance was as expected from the training validation side. The 
accuracy result indicates that approximately two of three images were mislabelled. That performance can 
lead inefficient recognition. To improve it, fixed grid feature extraction can be used in the future works. For 
hyperparameter tuning, different value and hyperparameter combinations can perform better metrics for 
validation and test side prediction. 
The HOG+SVM model prediction performance was also as expected with the 0.64 accuracy level. Even it 
can be said that it is slightly better than validation results. Different applications of HOG transformation can 
be researched and considered for future works. Also, like SIFT, different hyperparameter options may be 
considered based on relevant research. 
The CNN model prediction performance was expected to produce performance; however it gave 0.8 lower 
accuracy score than validation accuracy score. The training & validation loss graphic indicates an emerging 
gap between two lines, it is considered as an overfitting problem for model. Therefore, test score was not 
good as validation score.
Comparing three Emotion Recognition models, the SVM model with HOG transformations has the best 
accuracy level to predict emotions on test dataset.
The image data was not oversampled with SMOTE or other oversampling and undersampling techniques,
but it can be analysed for the future works in order relevant improvements.
For ‘in the wild’ video recognition, different neural networks algorithms can be used to have better 
emotion detection. About face detection, Viola Jones algorithm was not efficient to detect faces with 
different angle positions as can be seen in figure 8. Newer and more complex structures can be researched. 
References
1. Patle and D. S. Chouhan, "SVM kernel functions for classification," 2013 International Conference 
on Advances in Technology and Engineering (ICATE), 2013, pp. 1-9, doi: 
10.1109/ICAdTE.2013.6524743.
[Suggested length: ¼ page]
2. H. A. Qazi, U. Jahangir, B. M. Yousuf and A. Noor, "Human action recognition using SIFT and HOG 
method," 2017 International Conference on Information and Communication Technologies (ICICT), 
2017, pp. 6-10, doi: 10.1109/ICICT.2017.8320156
