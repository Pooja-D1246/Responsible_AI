import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lime import lime_tabular
import base64

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Responsible AI Dashboard", layout="wide")

# Custom CSS for Beige, Blue, and White Aesthetic
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    .stSidebar { background-color: #F5F5DC; } /* Beige Sidebar */
    h1, h2, h3 { color: #1E3A8A; } /* Deep Blue */
    .stButton>button { 
        background-color: #1E3A8A; 
        color: white; 
        border-radius: 8px;
        border: none;
    }
    /* This changes the general body text and paragraph color */
    p, label { 
        color: #4087c5 !important; /* Muted Light Blue/Slate */
    }

    /* This specifically targets the metric labels and values */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #4087c5 !important;
    }
    .report-box { 
        background-color: #FAF9F6; 
        color: #1E3A8A !important;
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🌐 Responsible AI in Global Data-Driven Decision Making")
# st.markdown("---")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload your Dataset (.csv)", type="csv")
    
    domain = st.selectbox("Select Global Domain", 
                          ["Healthcare", "Finance", "Criminal Justice", "HR & Recruitment", "General"])
    
    model_type = st.selectbox("Select Training Model", 
                              ["Logistic Regression", "Decision Tree", "Random Forest"])

# --- MAIN LOGIC ---
if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna()
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Domain Context", domain)
    
    st.dataframe(df.head(5), use_container_width=True)

    # User Selection for Columns
    st.sidebar.markdown("---")
    all_columns = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Column (Label)", all_columns)
    sensitive_col = st.sidebar.selectbox("Select Sensitive Column (e.g. Gender, Race)", all_columns)
    
    features = [c for c in all_columns if c != target_col]
    
    if st.sidebar.button("🚀 Run Full Analysis"):
        # 1. DATA PREP
        # Simple encoding for demo purposes
        X = pd.get_dummies(df[features], drop_first=True)
        y = pd.get_dummies(df[target_col], drop_first=True).iloc[:, 0]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. MODEL TRAINING
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5)
        else:
            model = RandomForestClassifier(n_estimators=100)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 3. PERFORMANCE METRICS
        st.header("📈 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        m2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
        m4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

        # 4. BIAS DETECTION (Disparate Impact Ratio)
        st.header("⚖️ Bias & Fairness Audit")
        # Calculate ratio of positive outcomes for sensitive groups
        privileged_group = df[sensitive_col].unique()[0]
        unprivileged_group = df[sensitive_col].unique()[1] if len(df[sensitive_col].unique()) > 1 else privileged_group
        
        # Simple logic: Ratio of target mean between groups
        rate_unprivileged = df[df[sensitive_col] == unprivileged_group][target_col].apply(lambda x: 1 if x == df[target_col].unique()[0] else 0).mean()
        rate_privileged = df[df[sensitive_col] == privileged_group][target_col].apply(lambda x: 1 if x == df[target_col].unique()[0] else 0).mean()
        
        di_ratio = rate_unprivileged / (rate_privileged + 1e-9)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(f"**Sensitive Attribute:** {sensitive_col}")
            st.write(f"**Disparate Impact Ratio:** {di_ratio:.2f}")
            if 0.8 <= di_ratio <= 1.25:
                st.success("Fairness check passed (80% Rule)")
            else:
                st.error("Bias detected: Fair impact thresholds breached.")

        with c2:
            fig, ax = plt.subplots(figsize=(6, 3))
            pd.Series({unprivileged_group: rate_unprivileged, privileged_group: rate_privileged}).plot(kind='barh', color=['#1E3A8A', '#D1D5DB'], ax=ax)
            ax.set_title("Positive Outcome Rate by Group")
            st.pyplot(fig)

        # 5. FEATURE IMPORTANCE
        st.header("🔍 Global Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
            
        feat_importances = pd.Series(importances, index=X.columns).nlargest(10)
        fig_feat, ax_feat = plt.subplots()
        feat_importances.plot(kind='barh', color='#1E3A8A', ax=ax_feat)
        st.pyplot(fig_feat)

        # 6. LIME EXPLAINABILITY (First 3 Rows)
        st.header("🧪 Local Explanations (LIME)")
        st.info("Explaining predictions for the top 3 records in the dataset...")
        
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X.columns.tolist(),
            class_names=['Negative', 'Positive'],
            mode='classification'
        )

        for i in range(3):
            exp = explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=5)
            st.write(f"**Record {i+1} Prediction:** {'Positive' if y_pred[i] == 1 else 'Negative'}")
            st.pyplot(exp.as_pyplot_figure())

        # 7. TRANSPARENCY REPORT GENERATION
        st.header("📄 Final Transparency Report")
        report_text = f"""
        RESPONSIBLE AI TRANSPARENCY REPORT
        ---------------------------------
        Domain: {domain}
        Model Used: {model_type}
        Dataset Size: {df.shape[0]} records
        
        FAIRNESS SUMMARY:
        - Sensitive Column: {sensitive_col}
        - Disparate Impact Ratio: {di_ratio:.2f}
        
        PERFORMANCE SUMMARY:
        - Accuracy: {accuracy_score(y_test, y_pred):.2f}
        - F1-Score: {f1_score(y_test, y_pred):.2f}
        
        GLOBAL COMPLIANCE:
        This model was evaluated under {domain} standards. 
        Local explainability provided via LIME to ensure accountability.
        """
        # Create the HTML-friendly version of the text first
        report_html = report_text.replace("\n", "<br>")
        # Then pass that variable into the f-string
        st.markdown(f'<div class="report-box">{report_html}</div>', unsafe_allow_html=True)
        
        # Download Button
        st.download_button(
            label="📥 Download Transparency Report",
            data=report_text,
            file_name="Responsible_AI_Report.txt",
            mime="text/plain"
        )
else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to begin the analysis.")