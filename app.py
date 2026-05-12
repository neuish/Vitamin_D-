import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import warnings

# Machine Learning Imports
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# --- 1. Data Loading & Preprocessing ---
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.dropna(inplace=True)
    
    # Rename columns for consistency
    df.columns = [
        'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
        'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
        'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'vitamin_d_ng_ml', 'deficient'
    ]
    
    # Feature Engineering for Visuals
    df['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, np.inf], 
                             labels=['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'], right=False)
    df['VitaminD_Supplement_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[0, 400, 800, 1000, 2000, np.inf], 
                                             labels=['0', '400', '800', '1000', '2000+'], right=False)
    df['sun_exposure_group'] = pd.cut(df['sun_hours_per_day'], bins=[0, 2, 4, 6, 8], 
                                      labels=['Low\n(0-2h)', 'Moderate\n(2-4h)', 'High\n(4-6h)', 'Very High\n(6-8h)'])
    df['supplement_tier'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 2001], 
                                   labels=['None (0 IU)', 'Low (1-400 IU)', 'Medium (401-800 IU)', 'High (801-1500 IU)', 'Very High (>1500 IU)'], right=False)
    return df

df = load_and_clean_data()

# --- 2. Encoding and ML Setup ---
categorical_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                    'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                    'alcohol_use', 'urban_rural']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_age_vitd_encoded = pd.get_dummies(df[['Age_Group', 'VitaminD_Supplement_Group']], drop_first=True)
df_encoded = pd.concat([df_encoded, df_age_vitd_encoded], axis=1)

# Define X and y
X_full = df_encoded.drop(columns=['deficient', 'vitamin_d_ng_ml', 'sun_exposure_group', 'supplement_tier', 'Age_Group', 'VitaminD_Supplement_Group'], errors='ignore')
y = df_encoded['deficient'].astype(int)

# Force boolean to int (Prevents XGBoost/SHAP errors)
for col in X_full.select_dtypes(include='bool').columns:
    X_full[col] = X_full[col].astype(int)

# Split & Scale
columns_to_scale = ['bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 
                    'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 
                    'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl']

x_train, x_test, y_train, y_test = train_test_split(X_full, y, train_size=0.7, random_state=100, stratify=y)

scaler = StandardScaler()
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()
x_train_scaled[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
x_test_scaled[columns_to_scale] = scaler.transform(x_test[columns_to_scale])

OPTIMAL_THRESHOLD = 0.4

# --- 3. Model Training ---
@st.cache_resource
def train_all_models(_xt, _yt, _xval, _yval):
    cat = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_seed=42, verbose=0).fit(_xt, _yt)
    lr = LogisticRegression(max_iter=1000).fit(_xt, _yt)
    xgb = XGBClassifier(random_state=42).fit(_xt, _yt)
    tab = TabNetClassifier(verbose=0, seed=42)
    tab.fit(X_train=_xt.values, y_train=_yt.values, eval_set=[(_xval.values, _yval.values)], max_epochs=20)
    return cat, lr, xgb, tab

cat_model, lr_model, xgb_model, tab_model = train_all_models(x_train_scaled, y_train, x_test_scaled, y_test)

# --- 4. Streamlit App Interface ---
def main():
    st.set_page_config(layout="wide", page_title="Vitamin D Health Analytics")
    st.title("Vitamin D Deficiency Prediction & Data Insights")

    tab1, tab2, tab3 = st.tabs(["Visualizations", "Model Performance", "Predictor & SHAP"])

    # --- TAB 1: ALL 7 VISUALIZATIONS ---
    with tab1:
        st.header("Exploratory Data Insights")
        
        # Viz 1: Risk Surface
        st.subheader("1. Deficiency Risk Surface")
        risk_surface = df.groupby(['sun_hours_per_day', 'supplement_tier'])['deficient'].mean().unstack().astype(float)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.heatmap(risk_surface * 100, cmap='RdYlGn_r', ax=ax1)
        st.pyplot(fig1)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Vitamin D Levels by Status")
            fig2, ax2 = plt.subplots()
            sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df, palette='coolwarm', ax=ax2)
            ax2.axhline(20, color='red', linestyle='--', label='Deficiency Threshold')
            st.pyplot(fig2)

            st.subheader("4. Body Fat % vs Vitamin D")
            fig4, ax4 = plt.subplots()
            sns.scatterplot(data=df, x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', alpha=0.4, ax=ax4)
            st.pyplot(fig4)

        with col2:
            st.subheader("3. Age Distribution")
            fig3, ax3 = plt.subplots()
            sns.kdeplot(data=df, x='age', hue='deficient', fill=True, ax=ax3)
            st.pyplot(fig3)

            st.subheader("5. Sun Exposure & Skin Tone")
            fig5, ax5 = plt.subplots()
            sns.boxplot(data=df, x='sun_exposure_group', y='vitamin_d_ng_ml', hue='skin_tone', ax=ax5)
            st.pyplot(fig5)

        st.subheader("6. Prevalence by Season & Skin Tone")
        prev = df.groupby(['skin_tone', 'season'])['deficient'].mean().unstack().astype(float)
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        prev.plot(kind='bar', ax=ax6)
        st.pyplot(fig6)

        st.subheader("7. Median Vitamin D Heatmap")
        df['sun_q'] = pd.qcut(df['sun_hours_per_day'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        med = df.groupby(['supplement_tier', 'sun_q'])['vitamin_d_ng_ml'].median().unstack().astype(float)
        fig7, ax7 = plt.subplots()
        sns.heatmap(med, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax7)
        st.pyplot(fig7)

    # --- TAB 2: PERFORMANCE, ROC, DCA ---
    with tab2:
        st.header("Model Evaluation Metrics")
        
        # Performance Table
        def get_row(name, model, is_tab=False):
            p = model.predict_proba(x_test_scaled.values if is_tab else x_test_scaled)[:, 1]
            pred = (p > OPTIMAL_THRESHOLD).astype(int)
            return [name, accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred), roc_auc_score(y_test, p)]

        metrics_df = pd.DataFrame([
            get_row("CatBoost", cat_model), get_row("XGBoost", xgb_model),
            get_row("Logistic Reg", lr_model), get_row("TabNet", tab_model, True)
        ], columns=["Model", "Accuracy", "Precision", "Recall", "AUC"])
        st.table(metrics_df.set_index("Model"))

        # ROC Curve
        st.subheader("ROC Curve Comparison")
        fig_roc, ax_roc = plt.subplots()
        for name, m, is_tab in [("CatBoost", cat_model, False), ("TabNet", tab_model, True)]:
            p = m.predict_proba(x_test_scaled.values if is_tab else x_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, p)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC: {roc_auc_score(y_test, p):.2f})")
        ax_roc.plot([0,1],[0,1], 'k--')
        st.pyplot(fig_roc)

        # DCA
        st.subheader("Decision Curve Analysis")
        thresh = np.linspace(0.01, 0.99, 50)
        p_cat = cat_model.predict_proba(x_test_scaled)[:, 1]
        net_benefit = []
        for t in thresh:
            tp = np.sum((p_cat >= t) & (y_test == 1))
            fp = np.sum((p_cat >= t) & (y_test == 0))
            net_benefit.append((tp / len(y_test)) - (fp / len(y_test)) * (t / (1 - t)))
        
        fig_dca, ax_dca = plt.subplots()
        ax_dca.plot(thresh, net_benefit, label='CatBoost Model')
        ax_dca.axhline(0, color='black', lw=1)
        st.pyplot(fig_dca)

    # --- TAB 3: INTERACTIVE PREDICTION & SHAP ---
    with tab3:
        st.header("Real-Time Prediction")
        
        # Sidebar for User Inputs
        st.sidebar.header("User Settings")
        u_input = {}
        for c in columns_to_scale:
            u_input[c] = st.sidebar.number_input(f"{c}", value=float(df[c].median()))
        
        # Simple One-Hot selection for sidebar
        u_sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
        
        # Construct prediction row
        row = pd.DataFrame(0, index=[0], columns=X_full.columns)
        for c in columns_to_scale: row.loc[0, c] = u_input[c]
        if u_sex == "Male": row.loc[0, 'sex_Male'] = 1
        
        # Scale and Predict
        row_scaled = row.copy()
        row_scaled[columns_to_scale] = scaler.transform(row[columns_to_scale])
        
        prob = cat_model.predict_proba(row_scaled)[0, 1]
        st.metric("Deficiency Risk", f"{prob:.2%}")
        if prob > OPTIMAL_THRESHOLD: st.error("High Risk")
        else: st.success("Low Risk")

        # SHAP
        st.divider()
        st.subheader("Feature Explanation (SHAP)")
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(row_scaled)
        
        fig_shap = plt.figure(figsize=(10, 3))
        shap.force_plot(explainer.expected_value, shap_values[0, :], row.iloc[0, :], matplotlib=True, show=False)
        st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()
