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

# Silence warnings for professional presentation
warnings.filterwarnings("ignore")

# --- 1. Decision Curve Analysis Function (Dissertation Logic) ---
def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        # Handle division by zero for threshold 1.0
        if pt == 1.0:
            net_benefit = 0
        else:
            net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(net_benefit)
    return net_benefits

# --- 2. Data Loading & Preprocessing Pipeline ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.dropna(inplace=True)
    
    # Consistent column naming
    df.columns = [
        'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
        'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
        'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'vitamin_d_ng_ml', 'deficient'
    ]
    
    # Visual Groups
    df['Sun_Exposure_Group'] = pd.cut(df['sun_hours_per_day'], bins=[0, 2, 4, 6, 8], labels=['Low', 'Moderate', 'High', 'V. High'])
    df['Supp_Tier'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 10000], labels=['None', 'Low', 'Medium', 'High', 'V. High'], right=False)
    
    # ML Encoding (One-Hot)
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 'alcohol_use', 'urban_rural']
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Advanced Grouping Features
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 20000], labels=['0-400', '401-800', '801-1000', '1001-2000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
        
    return df, df_ml

df_raw, df_ml = load_and_preprocess_data()

# Model Feature Selection (excluding target and leakage columns)
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu', 'Sun_Exposure_Group', 'Supp_Tier'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

scaler = StandardScaler()
x_train_sc = x_train.copy(); x_test_sc = x_test.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc[NUM_COLS] = scaler.transform(x_test[NUM_COLS])

# --- 3. Cached Model Training ---
@st.cache_resource
def train_models(_xt, _yt, _xv, _yv):
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42).fit(_xt, _yt)
    xgb = XGBClassifier(n_estimators=200, max_depth=3, random_state=42).fit(_xt, _yt)
    lr = LogisticRegression(max_iter=1000).fit(_xt, _yt)
    tab = TabNetClassifier(verbose=0, seed=42)
    tab.fit(X_train=_xt.values, y_train=_yt.values, eval_set=[(_xv.values, _yv.values)], max_epochs=20)
    return cat, xgb, lr, tab

cat_m, xgb_m, lr_m, tab_m = train_models(x_train_sc, y_train, x_test_sc, y_test)

# --- 4. Streamlit User Interface ---
st.set_page_config(page_title="Vitamin D Master's Dissertation", layout="wide")
st.title("🎓 Predictive Analytics for Vitamin D Deficiency")

tab_eda, tab_eval, tab_clinical = st.tabs(["📊 Exploratory Data Analysis", "🧪 Model Performance Suite", "🩺 Interactive Prediction & SHAP"])

# --- TAB 1: EDA (2-Column Layout) ---
with tab_eda:
    st.header("Epidemiological Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Risk Surface (Sun vs. Supps)")
        risk_map = df_raw.groupby(['Supp_Tier', 'Sun_Exposure_Group'])['deficient'].mean().unstack().astype(float)
        fig, ax = plt.subplots(figsize=(8, 5)); sns.heatmap(risk_map*100, annot=True, cmap="RdYlGn_r", ax=ax)
        st.pyplot(fig)

        st.subheader("2. Serum Vit-D vs Threshold")
        fig, ax = plt.subplots(figsize=(8, 5)); sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_raw, ax=ax)
        ax.axhline(20, color='red', ls='--'); st.pyplot(fig)

        st.subheader("3. Age Distribution profile")
        fig, ax = plt.subplots(figsize=(8, 5)); sns.kdeplot(data=df_raw, x='age', hue='deficient', fill=True, ax=ax)
        st.pyplot(fig)

        st.subheader("7. Median Levels Matrix")
        df_raw['Sun_Q'] = pd.qcut(df_raw['sun_hours_per_day'], 4, labels=['Q1','Q2','Q3','Q4'])
        med_h = df_raw.groupby(['Supp_Tier', 'Sun_Q'])['vitamin_d_ng_ml'].median().unstack().astype(float)
        fig, ax = plt.subplots(figsize=(8, 5)); sns.heatmap(med_h, annot=True, cmap="YlGnBu", ax=ax); st.pyplot(fig)

    with col2:
        st.subheader("4. BMI / Body Fat Relationship")
        fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_raw, x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', alpha=0.5, ax=ax)
        st.pyplot(fig)

        st.subheader("5. Skin Tone & Sun Exposure")
        fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_raw, x='Sun_Exposure_Group', y='vitamin_d_ng_ml', hue='skin_tone', ax=ax)
        st.pyplot(fig)

        st.subheader("6. Seasonal Prevalence Rates")
        fig, ax = plt.subplots(figsize=(8, 5)); df_raw.groupby('season')['deficient'].mean().sort_values().plot(kind='barh', color='skyblue', ax=ax)
        st.pyplot(fig)

# --- TAB 2: Model Evaluation & DCA ---
with tab_eval:
    st.header("Rigorous Model Evaluation")
    
    y_prob_cat = cat_m.predict_proba(x_test_sc)[:, 1]
    y_prob_xgb = xgb_m.predict_proba(x_test_sc)[:, 1]
    y_prob_lr  = lr_m.predict_proba(x_test_sc)[:, 1]
    y_prob_tab = tab_m.predict_proba(x_test_sc.values)[:, 1]

    # Metrics Summary
    res = []
    for name, prob in [("CatBoost", y_prob_cat), ("XGBoost", y_prob_xgb), ("TabNet", y_prob_tab), ("LogReg", y_prob_lr)]:
        pred = (prob > 0.4).astype(int)
        res.append({"Model": name, "AUC": roc_auc_score(y_test, prob), "F1": f1_score(y_test, pred), "Acc": accuracy_score(y_test, pred)})
    st.table(pd.DataFrame(res).set_index("Model"))

    ev1, ev2 = st.columns(2)
    with ev1:
        st.subheader("ROC Curve Analysis")
        fig, ax = plt.subplots()
        for name, p in [("CatBoost", y_prob_cat), ("XGBoost", y_prob_xgb), ("TabNet", y_prob_tab), ("LR", y_prob_lr)]:
            fpr, tpr, _ = roc_curve(y_test, p)
            ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, p):.2f})")
        ax.plot([0,1], [0,1], 'k--'); ax.legend(); st.pyplot(fig)

    with ev2:
        st.subheader("Decision Curve Analysis (Clinical Utility)")
        thresholds = np.linspace(0.01, 0.99, 100)
        nb_cat = decision_curve(y_test, y_prob_cat, thresholds)
        nb_xgb = decision_curve(y_test, y_prob_xgb, thresholds)
        nb_lr = decision_curve(y_test, y_prob_lr, thresholds)
        
        prevalence = np.mean(y_test)
        treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
        
        fig, ax = plt.subplots()
        ax.plot(thresholds, nb_cat, label='CatBoost', color='blue', lw=2)
        ax.plot(thresholds, nb_xgb, label='XGBoost', alpha=0.6)
        ax.plot(thresholds, nb_lr, label='LR', alpha=0.6)
        ax.plot(thresholds, treat_all, linestyle='--', color='red', label='Treat All')
        ax.axhline(0, color='black', linestyle='--', label='Treat None')
        ax.set_ylim(-0.05, 0.4); ax.set_xlabel("Threshold Prob"); ax.set_ylabel("Net Benefit"); ax.legend(); st.pyplot(fig)

# --- TAB 3: Clinical Prediction & SHAP ---
with tab_clinical:
    st.header("Patient Diagnostic Tool")
    pin, pout = st.columns([1, 2])
    
    with pin:
        st.info("Clinical & Lifestyle Inputs")
        i_age = st.slider("Age", 18, 95, 40)
        i_bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
        i_sun = st.slider("Sun Exposure (Hrs/Day)", 0.0, 8.0, 2.5)
        i_lat = st.number_input("Latitude", 0.0, 90.0, 30.0)
        i_cal = st.number_input("Serum Calcium (mg/dl)", 8.0, 11.0, 9.4)
        i_cho = st.number_input("Cholesterol (mg/dl)", 100, 350, 190)
        i_fat = st.slider("Body Fat %", 5, 50, 25)
        i_skin = st.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        i_supp = st.number_input("Vit D Supplement (IU)", 0, 5000, 400)
        i_sea = st.selectbox("Season", ["Winter", "Spring", "Summer", "Monsoon"])
        i_act = st.selectbox("Activity Level", ["Sedentary", "Low", "Moderate", "High"])
        
        predict_btn = st.button("Generate Diagnostic Report", type="primary")

    with pout:
        if predict_btn:
            # Map Inputs
            input_row = pd.DataFrame(0, index=[0], columns=X.columns)
            input_row.at[0, 'bmi'] = i_bmi
            input_row.at[0, 'sun_hours_per_day'] = i_sun
            input_row.at[0, 'latitude_deg'] = i_lat
            input_row.at[0, 'serum_calcium_mg_dl'] = i_cal
            input_row.at[0, 'cholesterol_mg_dl'] = i_cho
            input_row.at[0, 'body_fat_percentage'] = i_fat
            
            # Map One-Hots
            if f'skin_tone_{i_skin}' in X.columns: input_row.at[0, f'skin_tone_{i_skin}'] = 1
            if f'season_{i_sea}' in X.columns: input_row.at[0, f'season_{i_sea}'] = 1
            if f'physical_activity_level_{i_act}' in X.columns: input_row.at[0, f'physical_activity_level_{i_act}'] = 1
            
            in_sc = input_row.copy()
            in_sc[NUM_COLS] = scaler.transform(input_row[NUM_COLS])
            
            prob = cat_m.predict_proba(in_sc)[0,1]
            st.metric("Predicted Deficiency Risk", f"{prob*100:.1f}%")
            if prob > 0.4: st.error("Diagnosis: High Risk of Deficiency")
            else: st.success("Diagnosis: Normal Vitamin D Status Likely")
            
            st.divider()
            st.subheader("Model Interpretability (SHAP)")
            
            explainer = shap.TreeExplainer(cat_m)
            shap_vals = explainer.shap_values(x_test_sc)
            
            st.write("**Feature Importance (Magnitude):**")
            fig1, ax1 = plt.subplots(); shap.summary_plot(shap_vals, x_test_sc, plot_type='bar', show=False); st.pyplot(fig1)
            
            st.write("**Feature Impact (Directionality):**")
            fig2, ax2 = plt.subplots(); shap.summary_plot(shap_vals, x_test_sc, show=False); st.pyplot(fig2)
