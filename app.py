import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# Suppress warnings for a clean dissertation presentation
warnings.filterwarnings("ignore")

# --- Page Config & Styling ---
st.set_page_config(page_title="Vitamin D Deficiency Analytics", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; background-color: #ffffff; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Global Settings ---
OPTIMAL_THRESHOLD = 0.4
NUMERICAL_COLS = [
    'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 
    'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 
    'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl'
]

# --- 1. Data Loading & Advanced Preprocessing ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.dropna(inplace=True)
    
    # Standardize column names
    df.columns = [
        'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
        'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
        'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'vitamin_d_ng_ml', 'deficient'
    ]
    
    # Feature Engineering for Dashboard Visuals
    df['Sun_Exposure_Group'] = pd.cut(df['sun_hours_per_day'], bins=[0, 2, 4, 6, 8], 
                                      labels=['Low (0-2h)', 'Moderate (2-4h)', 'High (4-6h)', 'V. High (6-8h)'])
    df['Supp_Tier'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 5000], 
                             labels=['None', 'Low (1-400)', 'Med (401-800)', 'High (801-1500)', 'V. High (>1500)'], right=False)
    
    # ML Preprocessing (One-Hot Encoding)
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                'alcohol_use', 'urban_rural']
    
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Age & Supplement Grouping (for model consistency)
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], 
                                labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    df_ml['VitD_Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 10000], 
                                      labels=['0-400', '401-800', '801-1000', '1001-2000', '2000+'])
    
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'VitD_Supp_Group'], drop_first=True)
    
    # Cleanup bool to int
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
        
    return df, df_ml

df_raw, df_ml = load_and_clean_data()

# Prepare ML Tensors
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu', 'Sun_Exposure_Group', 'Supp_Tier'], errors='ignore')
y = df_ml['deficient']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

scaler = StandardScaler()
x_train_sc = x_train.copy()
x_test_sc = x_test.copy()
x_train_sc[NUMERICAL_COLS] = scaler.fit_transform(x_train[NUMERICAL_COLS])
x_test_sc[NUMERICAL_COLS] = scaler.transform(x_test[NUMERICAL_COLS])

# --- 2. Model Training (Cached Resource) ---
@st.cache_resource
def train_models(_xt, _yt, _xv, _yv):
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42).fit(_xt, _yt)
    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42).fit(_xt, _yt)
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(_xt, _yt)
    tab = TabNetClassifier(verbose=0, seed=42)
    tab.fit(X_train=_xt.values, y_train=_yt.values, eval_set=[(_xv.values, _yv.values)], max_epochs=30)
    return cat, xgb, lr, tab

cat_model, xgb_model, lr_model, tab_model = train_models(x_train_sc, y_train, x_test_sc, y_test)

# --- 3. Main Interface Layout ---
st.title("🏛️ Dissertation Project: Vitamin D Deficiency Prediction Framework")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🧪 Model Evaluation Suite", "🩺 Clinical Predictor & SHAP"])

# --- Tab 1: 7 Visualizations in 2-Column Grid ---
with tab1:
    st.header("Epidemiological Insights")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Deficiency Risk Surface")
        risk_map = df_raw.groupby(['Supp_Tier', 'Sun_Exposure_Group'])['deficient'].mean().unstack().astype(float)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.heatmap(risk_map * 100, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax1)
        ax1.set_title("Deficiency Prevalence (%)")
        st.pyplot(fig1)

        st.subheader("2. Serum Vit-D vs Threshold")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_raw, palette="Set2", ax=ax2)
        ax2.axhline(20, color='red', linestyle='--', label='20 ng/mL (Deficiency)')
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("3. Age Distribution by Status")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df_raw, x='age', hue='deficient', fill=True, ax=ax3)
        st.pyplot(fig3)

        st.subheader("7. Median Vitamin D Levels")
        df_raw['Sun_Q'] = pd.qcut(df_raw['sun_hours_per_day'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        med_heat = df_raw.groupby(['Supp_Tier', 'Sun_Q'])['vitamin_d_ng_ml'].median().unstack().astype(float)
        fig7, ax7 = plt.subplots(figsize=(8, 5))
        sns.heatmap(med_heat, annot=True, cmap="YlGnBu", ax=ax7)
        st.pyplot(fig7)

    with c2:
        st.subheader("4. BMI / Body Fat Impact")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_raw, x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', alpha=0.4, ax=ax4)
        st.pyplot(fig4)

        st.subheader("5. Skin Tone & Sun Sensitivity")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_raw, x='Sun_Exposure_Group', y='vitamin_d_ng_ml', hue='skin_tone', ax=ax5)
        st.pyplot(fig5)

        st.subheader("6. Seasonal Prevalence Rates")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        df_raw.groupby('season')['deficient'].mean().sort_values().plot(kind='barh', color='#3498db', ax=ax6)
        ax6.set_xlabel("Mean Deficiency Rate")
        st.pyplot(fig6)

# --- Tab 2: Model Evaluation Suite ---
with tab2:
    st.header("Comparative Model Performance")
    
    # Calculation for ROC and Metrics
    y_prob_cat = cat_model.predict_proba(x_test_sc)[:,1]
    y_prob_xgb = xgb_model.predict_proba(x_test_sc)[:,1]
    y_prob_lr  = lr_model.predict_proba(x_test_sc)[:,1]
    y_prob_tab = tab_model.predict_proba(x_test_sc.values)[:,1]

    # Metrics Table
    def calc_metrics(name, prob, y_true):
        pred = (prob > OPTIMAL_THRESHOLD).astype(int)
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred),
            "F1-Score": f1_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, prob)
        }

    results = pd.DataFrame([
        calc_metrics("CatBoost", y_prob_cat, y_test),
        calc_metrics("XGBoost", y_prob_xgb, y_test),
        calc_metrics("TabNet", y_prob_tab, y_test),
        calc_metrics("Logistic Reg", y_prob_lr, y_test)
    ]).set_index("Model")
    
    st.subheader("Model Performance Comparison")
    st.table(results.style.format("{:.3f}").background_gradient(cmap="Blues"))

    e_col1, e_col2 = st.columns(2)
    
    with e_col1:
        st.subheader("ROC Curve Comparison")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for name, prob in [("CatBoost", y_prob_cat), ("XGBoost", y_prob_xgb), ("TabNet", y_prob_tab), ("LR", y_prob_lr)]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.2f})")
        ax_roc.plot([0,1], [0,1], 'k--')
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
        ax_roc.legend()
        st.pyplot(fig_roc)

    with e_col2:
        st.subheader("Decision Curve Analysis (Clinical)")
        # DCA Logic
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefit = []
        for pt in thresholds:
            tp = np.sum((y_prob_cat >= pt) & (y_test == 1))
            fp = np.sum((y_prob_cat >= pt) & (y_test == 0))
            nb = (tp / len(y_test)) - (fp / len(y_test)) * (pt / (1 - pt))
            net_benefit.append(nb)
        
        fig_dca, ax_dca = plt.subplots(figsize=(8, 6))
        ax_dca.plot(thresholds, net_benefit, label='CatBoost (Clinical NB)', color='blue')
        ax_dca.axhline(0, color='black', lw=1, label='Treat None')
        ax_dca.set_ylim(-0.05, y_test.mean() + 0.1)
        ax_dca.set_xlabel("Threshold Probability")
        ax_dca.set_ylabel("Net Benefit")
        ax_dca.legend()
        st.pyplot(fig_dca)

# --- Tab 3: Interactive Predictor & SHAP ---
with tab3:
    st.header("Patient-Specific Risk Assessment")
    
    p_col1, p_col2 = st.columns([1, 2])
    
    with p_col1:
        st.markdown("### 📋 Input Profile")
        # Numerical Inputs
        u_age = st.slider("Age", 1, 100, 35)
        u_bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
        u_sun = st.slider("Sun Hours/Day", 0.0, 12.0, 3.0)
        u_screen = st.slider("Screen Time (Hrs)", 0.0, 16.0, 4.0)
        u_supp = st.number_input("Vitamin D Supplement (IU)", 0, 5000, 400)
        u_lat = st.number_input("Latitude (Degrees)", 0.0, 90.0, 35.0)
        u_act = st.selectbox("Activity Level", ["Sedentary", "Low", "Moderate", "High"])
        u_skin = st.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        u_season = st.selectbox("Current Season", ["Spring", "Summer", "Monsoon", "Winter"])
        
        # Prepare Input Vector
        input_df = pd.DataFrame(0, index=[0], columns=X.columns)
        input_df.at[0, 'bmi'] = u_bmi
        input_df.at[0, 'sun_hours_per_day'] = u_sun
        input_df.at[0, 'screen_time_hours'] = u_screen
        input_df.at[0, 'latitude_deg'] = u_lat
        
        # Map Categorical selections to OHE columns
        if f'skin_tone_{u_skin}' in input_df.columns: input_df.at[0, f'skin_tone_{u_skin}'] = 1
        if f'season_{u_season}' in input_df.columns: input_df.at[0, f'season_{u_season}'] = 1
        if f'physical_activity_level_{u_act}' in input_df.columns: input_df.at[0, f'physical_activity_level_{u_act}'] = 1
        
        # Scale Input
        input_sc = input_df.copy()
        input_sc[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])
        
    with p_col2:
        st.markdown("### 🔍 Model Diagnostic")
        prob = cat_model.predict_proba(input_sc)[0, 1]
        
        st.metric("Predicted Deficiency Risk", f"{prob*100:.1f}%")
        
        if prob > OPTIMAL_THRESHOLD:
            st.error("Conclusion: HIGH RISK of Vitamin D Deficiency. Clinical consultation recommended.")
        else:
            st.success("Conclusion: LOW RISK of Vitamin D Deficiency. Maintain current lifestyle.")

        st.divider()
        st.subheader("Global Explanations (SHAP)")
        st.markdown("How the model values different features across the population:")
        
        # SHAP Plot
        explainer = shap.TreeExplainer(cat_model)
        shap_vals = explainer.shap_values(x_test_sc)
        
        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals, x_test_sc, plot_type="bar", show=False)
        st.pyplot(fig_shap)
        plt.close()

st.markdown("---")
st.caption("Developed as part of a Master's Dissertation Project in Clinical Health Informatics.")
