import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report # Added for the report matrix
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the 5 specific classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# =====================================================================
# ⚙️ MACHINE LEARNING PIPELINE ENGINE 
# =====================================================================

def calculate_all_metrics(y_true, y_pred, y_prob, target_names=None):
    """Computes all evaluation metrics and generates the text classification report."""
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_pred)
    }
    
    # Generate the text report matrix safely
    if target_names is None:
        target_names = ["Class 0", "Class 1"]
        
    scores["class_report_text"] = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names, 
        zero_division=0
    )
    return scores

def generate_default_dataset():
    """Generates synthetic classification data if no user file is uploaded."""
    X, y = make_classification(
        n_samples=600,       
        n_features=12,      
        n_informative=9,    
        n_redundant=3,      
        random_state=42
    )
    feature_names = [f"Feature_{i+1}" for i in range(12)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target_Holder'] = y
    return df, 'Target_Holder'

def run_model_pipeline(model_choice, df=None, target_column=None):
    """Processes custom or default data, trains the model, and returns scores."""
    if df is None or target_column is None:
        df, target_column = generate_default_dataset()
        
    try:
        # 1. Clean data
        df_clean = df.dropna().copy()
        
        if len(df_clean) < 10:
            return {"error": "Dataset has too few samples after dropping missing values (minimum 10 required)."}

        # 2. Separate Features (X) and Target (y)
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # 3. Text Conversion
        object_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(object_cols) > 0:
            X = pd.get_dummies(X, columns=object_cols, drop_first=True)
        
        # Extract unique classes before encoding for descriptive names if they are text strings
        original_classes = sorted(list(y.dropna().unique()))
        
        # Convert target labels to numbers
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_labels = [str(cls) for cls in le.classes_]
        else:
            unique_classes = np.unique(y)
            target_labels = [f"Class {int(cls)}" for cls in unique_classes]
            
        if len(np.unique(y)) != 2:
            return {"error": f"This dashboard is designed for binary classification (2 classes). Your target has {len(np.unique(y))} classes."}

        # 4. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Initialize the selected model
        if model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_choice == "Decision Tree Classifier":
            model = DecisionTreeClassifier(random_state=42)
        elif model_choice == "K-Nearest Neighbor Classifier":
            model = KNeighborsClassifier()
        elif model_choice == "Naive Bayes Classifier":
            model = GaussianNB()
        elif model_choice == "Ensemble Model - Random Forest":
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Invalid model selected: {model_choice}")

        # Train the model
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred 
        
        # Compute metrics and structure report text matrix
        computed_scores = calculate_all_metrics(y_test, y_pred, y_prob, target_names=target_labels)
        
        # Attach structural metadata back to UI
        computed_scores["total_samples"] = len(df_clean)
        computed_scores["total_features"] = X.shape[1]
        
        return computed_scores

    except Exception as e:
        return {"error": f"An error occurred during training processing: {str(e)}"}


# =====================================================================
# 🖥️ STREAMLIT INTERFACE LAYOUT 
# =====================================================================

st.set_page_config(page_title="Custom Dataset ML Dashboard", layout="wide")
st.title("📊 Custom Dataset Machine Learning Dashboard")
st.markdown("Upload a `.csv` or `.data` file, choose your target column, and evaluate 5 different ML models.")
st.divider()

# Sidebar Setup
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file (.csv or .data):", 
    type=["csv", "data"]
)

df = None
target_column = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep=r'\s+|,', engine='python', header=None)
            df.columns = [f"Col_{i+1}" for i in range(len(df.columns))]
            
            if len(df.columns) == 32 and "Col_1" in df.columns:
                df = df.drop(columns=["Col_1"])
                st.sidebar.warning("Note: Patient ID ('Col_1') dropped automatically.")
            
        st.sidebar.success("File uploaded successfully!")
        
        st.sidebar.header("2. Target Settings")
        available_cols = list(df.columns)
        default_index = available_cols.index("Col_2") if "Col_2" in available_cols else 0
        
        target_column = st.sidebar.selectbox(
            "Select the Target (Label) column:",
            options=available_cols,
            index=default_index
        )
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

st.sidebar.header("3. Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model:",
    [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier",
        "Ensemble Model - Random Forest"
    ]
)

st.sidebar.divider()
run_pipeline = st.sidebar.button("🚀 Train & Evaluate Model", use_container_width=True)

if run_pipeline:
    with st.spinner(f"Computing predictions using {model_choice}..."):
        results = run_model_pipeline(model_choice, df, target_column)

    if "error" in results:
        st.error(results["error"])
    else:
        st.success(f"Model Pipeline complete using {model_choice}!")
        st.subheader(f"Evaluation Metrics Summary: {model_choice}")
        
        if uploaded_file is not None:
            st.info(f"📊 Running on custom dataset: **{uploaded_file.name}** ({results['total_samples']} samples, {results['total_features']} features)")
        else:
            st.warning("⚠️ No file uploaded. Running on default dataset.")

        # Metric cards block
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric(label="1. Accuracy", value=f"{results['Accuracy']:.4f}")
        col2.metric(label="2. AUC Score", value=f"{results['AUC Score']:.4f}")
        col3.metric(label="3. Precision", value=f"{results['Precision']:.4f}")

        col4.metric(label="4. Recall", value=f"{results['Recall']:.4f}")
        col5.metric(label="5. F1 Score", value=f"{results['F1 Score']:.4f}")
        col6.metric(label="6. MCC Score", value=f"{results['MCC Score']:.4f}")

        st.divider()

        # =====================================================================
        # 📋 NEW FIX: CLASSIFICATION REPORT OVER EXPANDER
        # =====================================================================
        st.subheader("📋 Comprehensive Classification Report Matrix")
        st.markdown("Detailed breakdown of precision, recall, and support counts for separate classes:")
        
        # Display the formatted monospace report matrix inside a text code component
        if "class_report_text" in results:
            st.code(results["class_report_text"], language="text")
            
else:
    st.info("👈 Set your parameters or upload a dataset in the sidebar, then click **'Train & Evaluate Model'** to view the metrics dashboard.")
