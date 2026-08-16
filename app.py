import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("💳 Credit Card Default Prediction")

st.write(
    """
    This application compares five machine learning models
    for predicting credit card default.
    """
)


# ============================================================
# Model Information
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Logistic Regression":
        BASE_DIR / "model" / "logistic_regression.pkl",

    "Decision Tree":
        BASE_DIR / "model" / "decision_tree.pkl",

    "KNN":
        BASE_DIR / "model" / "knn.pkl",

    "Naive Bayes":
        BASE_DIR / "model" / "naive_bayes.pkl",

    "Random Forest":
        BASE_DIR / "model" / "random_forest.pkl"
}


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_PATHS.keys())
)


# ============================================================
# File Upload
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV test dataset",
    type=["csv"]
)


# ============================================================
# Process Uploaded Dataset
# ============================================================

if uploaded_file is not None:

    # Read CSV
    test_data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully.")

    st.subheader("Dataset Preview")

    st.dataframe(
        test_data.head()
    )

    st.write(
        f"Dataset shape: {test_data.shape}"
    )


    # ========================================================
    # Check Target Column
    # ========================================================

    target_column = "default_payment_next_month"

    if target_column not in test_data.columns:

        st.error(
            f"The uploaded dataset must contain "
            f"'{target_column}' column."
        )

        st.stop()


    # ========================================================
    # Separate Features and Target
    # ========================================================

    X_test_app = test_data.drop(
        columns=[target_column]
    )

    y_test_app = test_data[target_column]


    # ========================================================
    # Load Selected Model
    # ========================================================

    model_path = MODEL_PATHS[selected_model]

    try:

        model = joblib.load(model_path)

    except Exception as e:

        st.error(
            f"Unable to load model: {e}"
        )

        st.stop()


    # ========================================================
    # Prediction
    # ========================================================

    try:

        y_pred = model.predict(
            X_test_app
        )

        y_prob = model.predict_proba(
            X_test_app
        )[:, 1]

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.stop()


    # ========================================================
    # Calculate Metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test_app,
        y_pred
    )

    auc = roc_auc_score(
        y_test_app,
        y_prob
    )

    precision = precision_score(
        y_test_app,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test_app,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test_app,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test_app,
        y_pred
    )


    # ========================================================
    # Display Model
    # ========================================================

    st.header("2. Model Evaluation")

    st.subheader(
        f"Selected Model: {selected_model}"
    )


    # ========================================================
    # Metrics
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

    with col2:
        st.metric(
            "AUC",
            f"{auc:.4f}"
        )

    with col3:
        st.metric(
            "Precision",
            f"{precision:.4f}"
        )


    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with col5:
        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

    with col6:
        st.metric(
            "MCC",
            f"{mcc:.4f}"
        )


    # ========================================================
    # Confusion Matrix
    # ========================================================

    st.header("3. Confusion Matrix")

    cm = confusion_matrix(
        y_test_app,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "No Default",
            "Default"
        ],
        yticklabels=[
            "No Default",
            "Default"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        f"{selected_model} - Confusion Matrix"
    )

    st.pyplot(fig)


    # ========================================================
    # Classification Report
    # ========================================================

    st.header("4. Classification Report")

    report = classification_report(
        y_test_app,
        y_pred,
        target_names=[
            "No Default",
            "Default"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4)
    )


    # ========================================================
    # Prediction Distribution
    # ========================================================

    st.header("5. Prediction Summary")

    prediction_counts = pd.Series(
        y_pred
    ).value_counts()

    prediction_counts.index = [
        "No Default" if x == 0 else "Default"
        for x in prediction_counts.index
    ]

    st.bar_chart(
        prediction_counts
    )


else:

    st.info(
        "Please upload a CSV test dataset to begin."
    )