# Credit Card Default Prediction

## 1. Problem Statement

Credit card default prediction is a binary classification problem in which the objective is to predict whether a customer is likely to default on their credit card payment in the following month.

In this project, multiple machine learning classification algorithms are implemented and compared using the same credit card default dataset. The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application has also been developed to allow a user to upload test data, select a trained machine learning model, generate predictions, and view the corresponding evaluation results.

---

## 2. Dataset Description

The dataset used in this project is the **Credit Card Default Dataset**, obtained from a public dataset repository.

The dataset contains **30,000 customer records** and **23 predictor features** after removing the original index column. The target variable is:

`default_payment_next_month`

where:

* `0` represents no default in the following month.
* `1` represents default in the following month.

### Feature Groups

The dataset contains information related to:

* Credit limit
* Gender
* Education
* Marital status
* Age
* Previous payment status for six months
* Bill statement amounts for six months
* Previous payment amounts for six months

The dataset satisfies the assignment requirements of having more than 12 features and more than 500 instances.

The data was divided into:

* **Training set:** 24,000 records
* **Test set:** 6,000 records

The test dataset used by the Streamlit application is provided as:

`test_data.csv`

---

## 3. GitHub Repository Link

**GitHub Repository:**
https://github.com/2025ac05728-afk/credit-card-default

The repository contains the complete source code, trained model files, requirements file, test dataset, and this README file.

---

## 4. Models Used and Evaluation

The following five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Evaluation Metrics

The models were evaluated using the following metrics:

* **Accuracy:** Measures the proportion of correctly classified observations.
* **AUC:** Measures the ability of the classifier to distinguish between the two classes.
* **Precision:** Measures the proportion of predicted defaults that were actually defaults.
* **Recall:** Measures the proportion of actual defaults correctly identified by the model.
* **F1 Score:** Harmonic mean of precision and recall.
* **MCC:** A balanced classification metric that considers all four values in the confusion matrix.

### Model Comparison

| ML Model                 | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression      |   0.8170 | 0.7478 |    0.6676 | 0.3436 | 0.4537 | 0.3856 |
| Decision Tree            |   0.7213 | 0.6120 |    0.3803 | 0.4130 | 0.3960 | 0.2156 |
| KNN                      |   0.7935 | 0.7040 |    0.5485 | 0.3753 | 0.4456 | 0.3330 |
| Naive Bayes              |   0.7408 | 0.7314 |    0.4353 | 0.5780 | 0.4966 | 0.3327 |
| Random Forest (Ensemble) |   0.8142 | 0.7574 |    0.6380 | 0.3693 | 0.4678 | 0.3848 |

---

## 5. Observations on Model Performance

| ML Model                     | Observation about Model Performance                                                                                                                                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Logistic Regression achieved the highest accuracy (0.8170), highest precision (0.6676), and highest MCC (0.3856). It provides a strong overall baseline and performs well despite being a relatively simple linear classification model.                 |
| **Decision Tree**            | Decision Tree achieved the lowest overall performance among the five models, with an accuracy of 0.7213 and MCC of 0.2156. Its relatively lower AUC indicates weaker class discrimination on the test data.                                              |
| **KNN**                      | KNN achieved an accuracy of 0.7935 and showed moderate performance across the evaluation metrics. Its performance was better than the Decision Tree but lower than Logistic Regression and Random Forest on most metrics.                                |
| **Naive Bayes**              | Naive Bayes achieved the highest recall (0.5780) and the highest F1 score (0.4966). This indicates that it identified a larger proportion of actual default cases and provided the best balance between precision and recall among the evaluated models. |
| **Random Forest (Ensemble)** | Random Forest achieved the highest AUC (0.7574), indicating the strongest overall ability to distinguish between default and non-default customers. It also achieved high accuracy (0.8142) and a competitive MCC (0.3848).                              |

### Overall Winner for the Dataset

**Overall Winner: Logistic Regression**

Logistic Regression achieved the highest accuracy, precision, and MCC among the evaluated models. It also performed competitively in terms of AUC and F1 score. Therefore, considering the overall balance of the evaluation metrics, Logistic Regression is selected as the overall winner for this dataset.

However, the choice of model can depend on the application objective. If identifying as many actual default cases as possible is the primary objective, **Naive Bayes** is preferable because it achieved the highest recall and F1 score. If ranking and distinguishing between default and non-default customers is the main objective, **Random Forest** is attractive because it achieved the highest AUC.

---

## 6. Streamlit Application
**Live Streamlit App:**
 https://credit-card-default-43rklgguvxc94kkadd2cek.streamlit.app/
 
The project includes an interactive Streamlit web application.

The application provides the following features:

### Dataset Upload

The user can upload a CSV test dataset through the Streamlit interface.

Only test data is uploaded to the application to keep the application lightweight and suitable for Streamlit Community Cloud.

### Model Selection

A model-selection dropdown allows the user to select one of the five trained classification models:

* Logistic Regression
* Decision Tree
* KNN
* Naive Bayes
* Random Forest

### Evaluation Metrics

After uploading the test dataset and selecting a model, the application displays:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* MCC

### Confusion Matrix

The application displays a confusion matrix showing the actual and predicted classes.

### Classification Report

A classification report is also displayed for the selected model.

### Prediction Summary

The application provides a summary of the predicted No Default and Default classes.

---

## 7. Project Structure

```text
credit-card-default/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── credit_card_default_classification.ipynb
│
├── model/
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
│
├── .gitignore
└── .gitattributes
```

The trained model files are stored using Git Large File Storage (Git LFS).

---

## 8. Requirements

The main Python libraries used in the project are:

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn

The exact package requirements are provided in `requirements.txt`.

---

## 9. Live Streamlit Application

**Live Streamlit App:**
[ADD YOUR STREAMLIT COMMUNITY CLOUD LINK HERE]

The deployed application provides an interactive frontend where the user can upload the test CSV file, select a classification model, and view the model's evaluation metrics and classification results.

---

## 10. Conclusion

Five classification models were implemented and evaluated on the Credit Card Default dataset.

Logistic Regression provided the strongest overall performance based on accuracy, precision, and MCC, while Random Forest achieved the highest AUC. Naive Bayes achieved the highest recall and F1 score, making it useful when identifying actual default cases is particularly important.

The trained models were integrated into an interactive Streamlit application, allowing the models to be evaluated through a user-friendly web interface.
