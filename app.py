import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="Vitamin D Predictive Analytics", layout="wide")

# --- CSS for Dissertation Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    h1, h2, h3 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

# --- Global Constants ---
OPTIMAL_THRESHOLD = 0.4
COLUMNS_TO_SCALE = [
    'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 
    'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 
    'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl'
]

# --- Cached Data & Model Pipeline ---
@st.cache_resource
def load_and_process_data():
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.dropna(inplace=True)
    
    # Column Mapping
    df.columns = [
        'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
        'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
        'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'vitamin_d_ng_ml', 'deficient'
    ]

    # Feature Engineering
    df['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    df['VitD_Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 5000], labels=['0-400', '401-800', '801-1000', '1001-2000', '2000+'])
    
    # Encoding
    categorical_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                        'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                        'alcohol_use', 'urban_rural', 'Age_Group', 'VitD_Supp_Group']
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col] = df_encoded[col].astype(int)
        
    return df, df_encoded

@st.cache_resource
def train_models(_x_train, _y_train, _x_test, _y_test):
    # CatBoost
    cat = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
    cat.fit(_x_train, _y_train)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    xgb.fit(_x_train, _y_train)
    
    # TabNet
    tabnet = TabNetClassifier(verbose=0, seed=42)
    tabnet.fit(X_train=_x_train.values, y_train=_y_train.values, max_epochs=20)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(_x_train, _y_train)
    
    return cat, xgb, tabnet, lr

# --- Helper Functions ---
def dca_calculation(y_true, y_prob, thresholds):
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / len(y_true)) - (fp / len(y_true)) * (pt / (1 - pt)) if pt < 1 else 0
        net_benefits.append(nb)
    return net_benefits

# --- Main Logic ---
df_raw, df_encoded = load_and_process_data()

X = df_encoded.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu'])
Y = df_encoded['deficient']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

scaler = StandardScaler()
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()
x_train_scaled[COLUMNS_TO_SCALE] = scaler.fit_transform(x_train[COLUMNS_TO_SCALE])
x_test_scaled[COLUMNS_TO_SCALE] = scaler.transform(x_test[COLUMNS_TO_SCALE])

cat_model, xgb_model, tab_model, lr_model = train_models(x_train_scaled, y_train, x_test_scaled, y_test)

# --- Streamlit UI Tabs ---
st.title("🎓 Master's Dissertation: Vitamin D Deficiency Predictive Framework")
tab_viz, tab_eval, tab_pred = st.tabs(["📊 Exploratory Data Analysis", "🔬 Model Performance", "🔮 Clinical Prediction"])

with tab_viz:
    st.header("Epidemiological Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Risk Surface (Sun vs. Supps)")
        df_raw['sun_bin'] = pd.cut(df_raw['sun_hours_per_day'], bins=5)
        risk = df_raw.groupby(['sun_bin', 'skin_tone'])['deficient'].mean().unstack() * 100
        fig, ax = plt.subplots()
        sns.heatmap(risk, annot=True, cmap="RdYlGn_r", ax=ax)
        st.pyplot(fig)

        st.subheader("2. Serum Vitamin D by Skin Tone")
        fig, ax = plt.subplots()
        sns.violinplot(x='skin_tone', y='vitamin_d_ng_ml', data=df_raw, palette="muted", ax=ax)
        ax.axhline(20, color='red', ls='--', label="Threshold")
        st.pyplot(fig)

    with col2:
        st.subheader("3. Body Fat vs. Vitamin D Correlation")
        fig, ax = plt.subplots()
        sns.scatterplot(x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', data=df_raw, alpha=0.5, ax=ax)
        st.pyplot(fig)

        st.subheader("4. Seasonal Deficiency Prevalence")
        seasonal = df_raw.groupby('season')['deficient'].mean()
        fig, ax = plt.subplots()
        seasonal.plot(kind='bar', color='skyblue', ax=ax)
        st.pyplot(fig)

with tab_eval:
    st.header("Comparative Model Analytics")
    col1, col2 = st.columns([1, 1])

    y_prob_cat = cat_model.predict_proba(x_test_scaled)[:,1]
    y_prob_xgb = xgb_model.predict_proba(x_test_scaled)[:,1]
    y_prob_tab = tab_model.predict_proba(x_test_scaled.values)[:,1]
    y_prob_lr  = lr_model.predict_proba(x_test_scaled)[:,1]

    with col1:
        st.subheader("ROC-AUC Analysis")
        fig, ax = plt.subplots()
        for name, prob in [("CatBoost", y_prob_cat), ("XGBoost", y_prob_xgb), ("TabNet", y_prob_tab), ("LogReg", y_prob_lr)]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC: {roc_auc_score(y_test, prob):.2f})")
        ax.plot([0,1],[0,1], 'k--')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Decision Curve Analysis (Clinical Utility)")
        thresholds = np.linspace(0, 1, 50)
        fig, ax = plt.subplots()
        ax.plot(thresholds, dca_calculation(y_test, y_prob_cat, thresholds), label="CatBoost (Selected)")
        ax.plot(thresholds, dca_calculation(y_test, np.full(len(y_test), y_test.mean()), thresholds), label="Treat All", ls='--')
        ax.set_ylim(-0.05, 0.5)
        ax.legend()
        st.pyplot(fig)

with tab_pred:
    st.header("Patient Risk Assessment Tool")
    col_input, col_res = st.columns([1, 2])
    
    with col_input:
        st.markdown("### Patient Profile")
        in_age = st.number_input("Age", 1, 100, 35)
        in_bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
        in_sun = st.slider("Daily Sun Exposure (Hours)", 0.0, 10.0, 2.0)
        in_lat = st.number_input("Latitude (Degrees)", 0, 90, 30)
        in_skin = st.selectbox("Skin Tone", df_raw['skin_tone'].unique())
        in_supp = st.slider("Vit D Supplement (IU)", 0, 5000, 400)
        
        # Pre-process input for prediction
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        input_data.at[0, 'bmi'] = in_bmi
        input_data.at[0, 'sun_hours_per_day'] = in_sun
        input_data.at[0, 'latitude_deg'] = in_lat
        # ... (In a full app, you would map all categorical inputs here)
        
        input_scaled = input_data.copy()
        input_scaled[COLUMNS_TO_SCALE] = scaler.transform(input_data[COLUMNS_TO_SCALE])
        
    with col_res:
        prob = cat_model.predict_proba(input_scaled)[0,1]
        risk_status = "DEFICIENT" if prob > OPTIMAL_THRESHOLD else "NON-DEFICIENT"
        
        st.metric("Predicted Probability", f"{prob*100:.1f}%", delta=risk_status, delta_color="inverse" if risk_status=="DEFICIENT" else "normal")
        
        st.subheader("Global Feature Importance (SHAP)")
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(x_test_scaled)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, x_test_scaled, plot_type="bar", show=False)
        st.pyplot(fig)
