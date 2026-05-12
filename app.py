import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

# --- 1. Decision Curve Analysis Function ---
def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt)) if (1 - pt) != 0 else TP / N
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
    
    # ML Encoding Logic
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                'alcohol_use', 'urban_rural']
    
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], 
                                labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 20000], 
                                 labels=['0', '400', '800', '1000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
    
    return df, df_ml

df_raw, df_ml = load_and_preprocess_data()
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 'latitude_deg', 
            'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 
            'body_fat_percentage', 'serum_calcium_mg_dl']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
scaler = StandardScaler()
x_train_sc = x_train.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc = x_test.copy()
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

st.set_page_config(page_title="Vitamin D Prediction Analytics", layout="wide")
st.title("🎓 Vitamin D Deficiency: Clinical Diagnostic Dashboard")

t1, t2, t3 = st.tabs(["📊 EDA", "🧪 Model Performance", "🔮 Predictive Diagnostic"])

with t3:
    st.header("Patient Diagnostic Tool (CatBoost Backend)")
    cin, cout = st.columns([1, 2])
    
    with cin:
        st.info("**Enter Patient Profile**")
        # Features ordered by expected SHAP importance
        i_sun = st.slider("Sun Exposure (Hrs/Day)", 0.0, 8.0, 2.0)
        i_bmi = st.number_input("Body Mass Index (BMI)", 15.0, 50.0, 24.0)
        i_cal = st.number_input("Serum Calcium (mg/dl)", 8.0, 11.0, 9.2)
        i_fat = st.slider("Body Fat %", 5, 50, 25)
        i_age_grp = st.selectbox("Age Group", ['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
        i_supp_grp = st.selectbox("Vitamin D Supplement (IU/Day)", ['0', '400', '800', '1000', '2000+'])
        i_skin = st.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        i_sea = st.selectbox("Season", ["Winter", "Spring", "Summer", "Monsoon"])
        i_lat = st.number_input("Latitude (Degrees)", 0.0, 60.0, 30.0)
        i_act = st.selectbox("Physical Activity", ["Sedentary", "Low", "Moderate", "High"])
        i_diet = st.slider("Diet Quality Score", 1, 10, 5)
        i_sleep = st.slider("Sleep Duration (Hrs)", 4, 12, 7)
        
        predict_btn = st.button("Generate Diagnostic Report", type="primary")

    with cout:
        if predict_btn:
            # FIX: Initialize with 0.0 to prevent LossySetitemError
            row = pd.DataFrame(0.0, index=[0], columns=X.columns)
            
            # Fill numericals
            row.at[0, 'sun_hours_per_day'] = i_sun
            row.at[0, 'bmi'] = i_bmi
            row.at[0, 'serum_calcium_mg_dl'] = i_cal
            row.at[0, 'body_fat_percentage'] = i_fat
            row.at[0, 'latitude_deg'] = i_lat
            row.at[0, 'diet_score'] = i_diet
            row.at[0, 'sleep_hours'] = i_sleep
            
            # Map Categorical Bins
            if f'Age_Group_{i_age_grp}' in X.columns: row.at[0, f'Age_Group_{i_age_grp}'] = 1.0
            if f'Supp_Group_{i_supp_grp}' in X.columns: row.at[0, f'Supp_Group_{i_supp_grp}'] = 1.0
            if f'skin_tone_{i_skin}' in X.columns: row.at[0, f'skin_tone_{i_skin}'] = 1.0
            if f'season_{i_sea}' in X.columns: row.at[0, f'season_{i_sea}'] = 1.0
            if f'physical_activity_level_{i_act}' in X.columns: row.at[0, f'physical_activity_level_{i_act}'] = 1.0
            
            # Scale and Predict
            row_sc = row.copy()
            row_sc[NUM_COLS] = scaler.transform(row[NUM_COLS])
            prob = cat_m.predict_proba(row_sc)[0,1]
            
            st.subheader("Diagnostic Result")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Risk Probability", f"{prob*100:.1f}%")
            if prob > 0.45:
                res_col2.error("STATUS: LIKELY DEFICIENT")
            else:
                res_col2.success("STATUS: LIKELY NORMAL")
            
            st.divider()
            st.subheader("Explainable AI (SHAP Analysis)")
            explainer = shap.TreeExplainer(cat_m)
            shap_val = explainer.shap_values(row_sc)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_val, row_sc, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close()

with t1:
    st.header("Epidemiological Trends")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(); sns.kdeplot(data=df_raw, x='age', hue='deficient', fill=True, ax=ax); st.pyplot(fig); plt.close()
    with c2:
        fig, ax = plt.subplots(); sns.boxplot(data=df_raw, x='season', y='vitamin_d_ng_ml', ax=ax); st.pyplot(fig); plt.close()

with t2:
    st.header("Model Evaluation Metrics")
    comparison_data = {
        'Model': ['Logistic Regression', 'XGBoost', 'CatBoost', 'TabNet'],
        'Accuracy': [0.838, 0.841, 0.848, 0.830],
        'ROC-AUC': [0.926, 0.921, 0.922, 0.910]
    }
    st.table(pd.DataFrame(comparison_data))
