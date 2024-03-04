### Preprocessing For Baseline Models

stroke_cleaned_v1.csv is cleaned, by converting the age to its nearest integer, and removed Gender "other" which was one row of data. We kept maximum data and have not dropped any column in this dataset. We tested Five models of supervised machine learning by perserving maximum data.
### Splitting the Data 
Data set is split into Target Variable y that has the stroke column. 
X contains the features independent variables, (after deleting the stroke column) that will be used to predict the target variable.

![Alt text](image.png)

Splitting the data into Training and Testing Sets

![Alt text](image-1.png)

Stardarised the data by using the StandardScaler from sKlearn

![Alt text](image-2.png)

Handling Imbalanced Data by using Oversampling with SMOTE And RandomOverSampler from imblearn.over_sampling

### Supervised Machine Learning Models
We used 5 Machine Learning models 

Logistic Regression Model
 Model_LR = LogisticRegression(solver='lbfgs’, max_iter=200,random_state=78)

K_nearest neighbors 
             Model_knn = KNeighborsClassifier(n_neighbors=5)

Descision Tree
	model_DT = tree.DecisionTreeClassifier()

Random Forest
	rf_model = RandomForestClassifier(n_estimators=500, random_state=78)

Support Vector Machine (SVM)
	model_svm = SVC(kernel='linear')

    All above Models has been run on Original data , Oversampled Data with RandomoverSampler and Smote

After Closely observing the Confusion Matrix and classification reports of all the 5 models, We further optimise the Random Forest Model, Support vector MAchine and descion Tree .
The SVM with Oversampled Data offered the best Recall and accuracy for further optimisation
![Alt text](image-3.png)



Confusion Matrix of SVM with oversampled data is showing relatively a low number of False Negatives, though the False positives with the number of 330 is not good so we are further optimising this model to see if its accuracy that is currently 73%, can be improved.

![Alt text](image-4.png)


