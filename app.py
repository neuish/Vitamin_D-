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

warnings.filterwarnings("ignore")

# --- 1. Decision Curve Analysis Function ---
def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        if (1 - pt) == 0:
            net_benefit = TP / N
        else:
            net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(net_benefit)
    return net_benefits

# --- 2. Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.dropna(inplace=True)
    df.columns = [
        'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
        'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
        'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'vitamin_d_ng_ml', 'deficient'
    ]
    df['sun_exposure_group'] = pd.cut(df['sun_hours_per_day'], bins=[0, 2, 4, 6, 8], labels=['Low', 'Moderate', 'High', 'V. High'])
    df['supplement_tier'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 10000], labels=['None', 'Low', 'Medium', 'High', 'V. High'], right=False)
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 'alcohol_use', 'urban_rural']
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 20000], labels=['0', '400', '800', '1000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
    return df, df_ml

df_raw, df_ml = load_and_preprocess_data()
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu', 'sun_exposure_group', 'supplement_tier'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
scaler = StandardScaler()
x_train_sc = x_train.copy(); x_test_sc = x_test.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc[NUM_COLS] = scaler.transform(x_test[NUM_COLS])

@st.cache_resource
def train_models(_xt, _yt, _xv, _yv):
    cat = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42).fit(_xt, _yt)
    xgb = XGBClassifier(random_state=42).fit(_xt, _yt)
    lr = LogisticRegression(max_iter=1000).fit(_xt, _yt)
    tab = TabNetClassifier(verbose=0, seed=42)
    tab.fit(X_train=_xt.values, y_train=_yt.values, eval_set=[(_xv.values, _yv.values)], max_epochs=20)
    return cat, xgb, lr, tab

cat_m, xgb_m, lr_m, tab_m = train_models(x_train_sc, y_train, x_test_sc, y_test)

st.set_page_config(page_title="Vitamin D Prediction Full Analysis", layout="wide")
st.title("🎓 Vitamin D Deficiency: Comprehensive Analysis Dashboard")
tab_eda, tab_eval, tab_clinical = st.tabs(["📊 Exploratory Data Analysis", "🧪 Model Performance", "🔮 Predictive Diagnostic"])

with tab_eda:
    st.header("Epidemiological Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Risk Surface (Sun vs. Supplementation)")
        risk = df_raw.groupby(['supplement_tier', 'sun_exposure_group'])['deficient'].mean().unstack().astype(float)
        fig, ax = plt.subplots(); sns.heatmap(risk*100, annot=True, cmap="RdYlGn_r", ax=ax); st.pyplot(fig); plt.close()
        st.subheader("2. Vitamin D Distribution")
        fig, ax = plt.subplots(); sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_raw, ax=ax); ax.axhline(20, color='gold', ls='--'); st.pyplot(fig); plt.close()
        st.subheader("3. Age KDE Profile")
        fig, ax = plt.subplots(); sns.kdeplot(data=df_raw, x='age', hue='deficient', fill=True, ax=ax); st.pyplot(fig); plt.close()
    with c2:
        st.subheader("4. Body Fat % vs. Vitamin D")
        fig, ax = plt.subplots(); sns.scatterplot(data=df_raw, x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', alpha=0.5, ax=ax); st.pyplot(fig); plt.close()
        st.subheader("5. Skin Tone & Sun Exposure")
        fig, ax = plt.subplots(); sns.boxplot(data=df_raw, x='sun_exposure_group', y='vitamin_d_ng_ml', hue='skin_tone', ax=ax); st.pyplot(fig); plt.close()
        st.subheader("6. Seasonal Deficiency Prevalence")
        fig, ax = plt.subplots(); df_raw.groupby('season')['deficient'].mean().plot(kind='bar', color='skyblue', ax=ax); st.pyplot(fig); plt.close()

with tab_eval:
    st.header("Comparative Model Analytics")
    st.subheader("Performance Metrics Table")
    comparison_data = {
        'Model': ['Logistic Regression', 'XGBoost', 'CatBoost', 'TabNet'],
        'Accuracy': [0.838000, 0.841333, 0.847667, 0.829667],
        'Precision': [0.773783, 0.837125, 0.831010, 0.781051],
        'Recall': [0.848809, 0.755957, 0.783895, 0.806081],
        'F1 Score': [0.809561, 0.794473, 0.806765, 0.793368],
        'ROC-AUC': [0.925881, 0.920969, 0.921789, 0.909763]
    }
    st.table(pd.DataFrame(comparison_data))
    ev1, ev2 = st.columns(2)
    y_prob_cat = cat_m.predict_proba(x_test_sc)[:, 1]
    y_prob_xgb = xgb_m.predict_proba(x_test_sc)[:, 1]
    y_prob_lr  = lr_m.predict_proba(x_test_sc)[:, 1]
    y_prob_tab = tab_m.predict_proba(x_test_sc.values)[:, 1]
    with ev1:
        st.subheader("ROC Curve Comparison")
        fig, ax = plt.subplots()
        for name, p in [("CatBoost", y_prob_cat), ("XGBoost", y_prob_xgb), ("TabNet", y_prob_tab), ("LR", y_prob_lr)]:
            fpr, tpr, _ = roc_curve(y_test, p)
            ax.plot(fpr, tpr, label=f"{name}")
        ax.plot([0,1],[0,1],'k--'); ax.legend(); st.pyplot(fig); plt.close()
    with ev2:
        st.subheader("Decision Curve Analysis (Clinical Utility)")
        thresholds = np.linspace(0.01, 0.99, 100)
        nb_lr = decision_curve(y_test, y_prob_lr, thresholds)
        nb_xgb = decision_curve(y_test, y_prob_xgb, thresholds)
        nb_cat = decision_curve(y_test, y_prob_cat, thresholds)
        nb_tab = decision_curve(y_test, y_prob_tab, thresholds)
        prevalence = np.mean(y_test)
        treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
        treat_none = [0 for _ in thresholds]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(thresholds, nb_lr, label='LR')
        ax.plot(thresholds, nb_xgb, label='XGBoost')
        ax.plot(thresholds, nb_cat, label='CatBoost', color='blue', lw=2)
        ax.plot(thresholds, nb_tab, label='TabNet')
        ax.plot(thresholds, treat_all, linestyle='--', label='Treat All', color='red')
        ax.plot(thresholds, treat_none, linestyle='--', label='Treat None', color='black')
        ax.set_ylim(-0.05, 0.4); ax.set_xlabel("Threshold Probability"); ax.set_ylabel("Net Benefit"); ax.legend(); ax.grid(); st.pyplot(fig); plt.close()

with tab_clinical:
    st.header("Patient Diagnostic & Interpretability Tool")
    cin, cout = st.columns([1, 2])
    with cin:
        st.info("Enter Patient Data")
        i_bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
        i_sun = st.slider("Sun Exposure (Hrs/Day)", 0.0, 8.0, 2.5)
        i_cal = st.number_input("Serum Calcium (mg/dl)", 8.0, 11.0, 9.4)
        i_cho = st.number_input("Cholesterol (mg/dl)", 100, 350, 190)
        i_fat = st.slider("Body Fat %", 5, 50, 25)
        i_age = st.slider("Age", 18, 90, 45)
        i_skin = st.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        i_sea = st.selectbox("Season", ["Winter", "Spring", "Summer", "Monsoon"])
        i_supp = st.number_input("Vitamin D Supplement (IU)", 0, 5000, 400)
        i_scr = st.slider("Screen Time (Hrs)", 0, 15, 5)
        i_diet = st.slider("Diet Score", 1, 10, 5)
        i_sleep = st.slider("Sleep (Hrs)", 4, 12, 7)
        predict_btn = st.button("Run Prediction", type="primary")
    with cout:
        if predict_btn:
            input_row = pd.DataFrame(0, index=[0], columns=X.columns)
            input_row.at[0, 'bmi'] = i_bmi
            input_row.at[0, 'sun_hours_per_day'] = i_sun
            input_row.at[0, 'serum_calcium_mg_dl'] = i_cal
            input_row.at[0, 'cholesterol_mg_dl'] = i_cho
            input_row.at[0, 'body_fat_percentage'] = i_fat
            input_row.at[0, 'screen_time_hours'] = i_scr
            input_row.at[0, 'diet_score'] = i_diet
            input_row.at[0, 'sleep_hours'] = i_sleep
            if f'skin_tone_{i_skin}' in X.columns: input_row.at[0, f'skin_tone_{i_skin}'] = 1
            if f'season_{i_sea}' in X.columns: input_row.at[0, f'season_{i_sea}'] = 1
            in_sc = input_row.copy()
            in_sc[NUM_COLS] = scaler.transform(input_row[NUM_COLS])
            prob = cat_m.predict_proba(in_sc)[0,1]
            st.metric("Risk Probability", f"{prob*100:.1f}%")
            if prob > 0.4: st.error("DIAGNOSIS: DEFICIENT")
            else: st.success("DIAGNOSIS: NORMAL")
            st.divider()
            st.subheader("SHAP Interpretability (CatBoost)")
            explainer = shap.TreeExplainer(cat_m)
            shap_values = explainer.shap_values(x_test_sc)
            c_sh1, c_sh2 = st.columns(2)
            with c_sh1:
                st.write("**Feature Importance (Magnitude)**")
                fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, x_test_sc, plot_type='bar', show=False); st.pyplot(fig1); plt.close()
            with c_sh2:
                st.write("**Feature Impact (Directionality)**")
                fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, x_test_sc, show=False); st.pyplot(fig2); plt.close()
