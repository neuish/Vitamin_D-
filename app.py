import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# --- 1. Clinical Decision Curve Analysis (DCA) ---
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        if pt == 1:
            net_benefit.append(0)
            continue
        # Formula: (TP/N) - (FP/N) * (Pt/(1-Pt))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

# --- 2. Data Loading & Robust Preprocessing ---
@st.cache_data
def load_and_preprocess():
    # Load dataset
    try:
        df = pd.read_csv('Vitamin_D_Dataset.csv')
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'Vitamin_D_Dataset.csv' is in the root directory.")
        st.stop()
    
    # Standardize column names to avoid KeyErrors
    # Removes spaces, parentheses, and units
    df.columns = [
        c.lower()
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '_')
        .strip() 
        for c in df.columns
    ]

    # Map target columns if they were renamed with units in CSV
    # e.g., 'vitamin_d_supplement_iu' might have been 'vitamin_d_supplement'
    if 'age' not in df.columns:
        st.error(f"Missing 'age' column. Available: {list(df.columns)}")
        st.stop()

    # Categorical columns for encoding
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                'alcohol_use', 'urban_rural']
    
    existing_cat = [c for c in cat_cols if c in df.columns]
    df_ml = pd.get_dummies(df, columns=existing_cat, drop_first=True)
    
    # Binned Age and Supplements (Matches Notebook logic)
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], 
                                labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    
    # Find supplement column regardless of exact IU suffix
    supp_col = [c for c in df.columns if 'supplement' in c][0]
    df_ml['Supp_Group'] = pd.cut(df[supp_col], bins=[-1, 400, 800, 1000, 2000, 20000], 
                                 labels=['0', '400', '800', '1000', '2000+'])
    
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    # Ensure all bools are ints for CatBoost/SHAP
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
        
    return df, df_ml

df_raw, df_ml = load_and_preprocess()

# Feature Selection
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu', 'vitamin_d_supplement'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'latitude_deg', 'diet_score', 'sleep_hours']

# Train/Test Scaling
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
scaler = StandardScaler()
x_train_sc = x_train.copy(); x_test_sc = x_test.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc[NUM_COLS] = scaler.transform(x_test[NUM_COLS])

@st.cache_resource
def train_model():
    model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
    model.fit(x_train_sc, y_train)
    return model

cat_model = train_model()

# --- 3. UI Layout ---
st.set_page_config(page_title="VitD Diagnostic AI", layout="wide")
st.title("🔬 Vitamin D Deficiency Predictive Analytics")

tab_pred, tab_eval = st.tabs(["🔮 Patient Risk Prediction", "📊 Clinical DCA Evaluation"])

with tab_pred:
    st.info("Fill out the patient profile below to calculate risk probability.")
    
    # 2-Column Grid for Input
    col1, col2 = st.columns(2)
    
    with col1:
        i_sun = st.slider("Sun Hours per Day", 0.0, 10.0, 2.0)
        i_season = st.selectbox("Current Season", df_raw['season'].unique())
        i_bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 24.5)
        i_age_grp = st.selectbox("Age Range", ['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
        i_supp = st.selectbox("Vitamin D Supplement Level (IU)", ['0', '400', '800', '1000', '2000+'])
        
    with col2:
        i_skin = st.selectbox("Skin Tone Category", df_raw['skin_tone'].unique())
        i_lat = st.number_input("Latitude (Degrees)", -90.0, 90.0, 34.0)
        i_act = st.selectbox("Physical Activity Level", df_raw['physical_activity_level'].unique())
        i_diet = st.slider("Diet Quality Score", 1, 10, 5)
        i_sleep = st.slider("Average Sleep (Hours)", 4, 12, 7)

    predict_btn = st.button("Run Diagnostic & SHAP Analysis", type="primary", use_container_width=True)

    if predict_btn:
        # Initialize as float64 to avoid LossySetitemError
        row = pd.DataFrame(0.0, index=[0], columns=X.columns)
        
        # Map Numeric Data
        row.at[0, 'sun_hours_per_day'] = i_sun
        row.at[0, 'bmi'] = i_bmi
        row.at[0, 'latitude_deg'] = i_lat
        row.at[0, 'diet_score'] = i_diet
        row.at[0, 'sleep_hours'] = i_sleep
        
        # Map Categorical Selections
        for c in [f'season_{i_season}', f'skin_tone_{i_skin}', f'physical_activity_level_{i_act}', 
                    f'Age_Group_{i_age_grp}', f'Supp_Group_{i_supp}']:
            if c in X.columns: row.at[0, c] = 1.0

        # Scale for model
        row_sc = row.copy()
        row_sc[NUM_COLS] = scaler.transform(row[NUM_COLS])
        
        # Predict
        prob = cat_model.predict_proba(row_sc)[0,1]

        # Results Metric
        st.divider()
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Deficiency Probability", f"{prob*100:.1f}%")
        if prob > 0.45:
            res_c2.error("CLINICAL RECOMMENDATION: High Risk. Consider Screening.")
        else:
            res_c2.success("CLINICAL RECOMMENDATION: Low Risk. Normal levels likely.")

        # SHAP Grid Below Results
        st.subheader("Explainable AI: Feature Importance Grid")
        explainer = shap.TreeExplainer(cat_model)
        shap_vals = explainer.shap_values(row_sc)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.write("**Local Impact (Current Case)**")
            fig_l, ax_l = plt.subplots()
            shap.plots.bar(shap.Explanation(values=shap_vals[0], data=row.iloc[0], feature_names=X.columns), show=False)
            st.pyplot(fig_l)
            plt.close()

        with s_col2:
            st.write("**Global Logic (Model Population)**")
            fig_g, ax_g = plt.subplots()
            shap.summary_plot(explainer.shap_values(x_test_sc), x_test_sc, plot_type='bar', show=False)
            st.pyplot(fig_g)
            plt.close()

with tab_eval:
    st.header("Decision Curve Analysis (Clinical Utility)")
    
    y_probs_test = cat_model.predict_proba(x_test_sc)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # Model NB
    nb_model = calculate_net_benefit(y_test, y_probs_test, thresholds)
    # Treat All Baseline
    prevalence = np.mean(y_test)
    nb_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    
    fig_dca, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, nb_model, label='Prediction Model', color='blue', lw=2.5)
    ax.plot(thresholds, nb_all, label='Treat All Strategy', color='red', linestyle='--', alpha=0.6)
    ax.axhline(y=0, color='black', label='Treat None Strategy', lw=1.5)
    
    ax.set_ylim(-0.05, prevalence + 0.1)
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Clinical Utility: Net Benefit Curve')
    ax.legend()
    ax.grid(alpha=0.2)
    
    st.pyplot(fig_dca)
    st.info("The model is clinically superior to treating everyone or no one at thresholds where the blue line is the highest.")
