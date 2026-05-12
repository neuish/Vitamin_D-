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

# --- 1. Clinical Decision Curve Analysis (DCA) Function ---
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        if pt == 1:
            net_benefit.append(0)
            continue
        # Standard Formula: (TP/N) - (FP/N) * (Pt/(1-Pt))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

# --- 2. Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess():
    # Load raw data
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Categorical columns for encoding
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                'alcohol_use', 'urban_rural']
    
    # Matching Notebook Logic: Binned Age and Supplements
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], 
                                labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 20000], 
                                 labels=['0', '400', '800', '1000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    # Cast boolean dummies to int for SHAP compatibility
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
        
    return df, df_ml

df_raw, df_ml = load_and_preprocess()

# Prepare Model Inputs
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'latitude_deg', 'diet_score', 'sleep_hours']

# Scaling based on training set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
scaler = StandardScaler()
x_train_sc = x_train.copy(); x_test_sc = x_test.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc[NUM_COLS] = scaler.transform(x_test[NUM_COLS])

@st.cache_resource
def train_catboost_model():
    model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
    model.fit(x_train_sc, y_train)
    return model

model = train_catboost_model()

# --- 3. Streamlit UI ---
st.set_page_config(page_title="VitD AI Diagnostic", layout="wide")
st.title("☀️ Vitamin D deficiency Prediction Dashboard")

tab_predict, tab_dca = st.tabs(["🔮 Patient Prediction", "📊 Clinical DCA"])

with tab_predict:
    st.subheader("Enter Patient Clinical Parameters")
    
    # 2-Column Input Grid
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            i_sun = st.slider("Sun Exposure (Hours/Day)", 0.0, 10.0, 2.0)
            i_season = st.selectbox("Current Season", df_raw['season'].unique())
            i_bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 24.5)
            i_age_grp = st.selectbox("Age Group", ['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
            i_supp_grp = st.selectbox("Vitamin D Supplement Intake (IU)", ['0', '400', '800', '1000', '2000+'])

        with col2:
            i_skin = st.selectbox("Skin Tone", df_raw['skin_tone'].unique())
            i_lat = st.number_input("Geographic Latitude", -90.0, 90.0, 34.0)
            i_activity = st.selectbox("Physical Activity Level", df_raw['physical_activity_level'].unique())
            i_diet = st.slider("Diet Quality Score (1-10)", 1, 10, 5)
            i_sleep = st.slider("Sleep Duration (Hours)", 4, 12, 7)

    predict_btn = st.button("Run Prediction & Generate SHAP", type="primary", use_container_width=True)

    if predict_btn:
        # Construct prediction row with float64 to avoid LossySetitemError
        row = pd.DataFrame(0.0, index=[0], columns=X.columns)
        
        # Assign Numerical Values
        row.at[0, 'sun_hours_per_day'] = i_sun
        row.at[0, 'bmi'] = i_bmi
        row.at[0, 'latitude_deg'] = i_lat
        row.at[0, 'diet_score'] = i_diet
        row.at[0, 'sleep_hours'] = i_sleep
        
        # Assign One-Hot Encoded Categories
        cat_matches = [
            f'season_{i_season}', f'skin_tone_{i_skin}', f'Age_Group_{i_age_grp}',
            f'Supp_Group_{i_supp_grp}', f'physical_activity_level_{i_activity}'
        ]
        for c in cat_matches:
            if c in X.columns: row.at[0, c] = 1.0

        # Scale and Predict
        row_sc = row.copy()
        row_sc[NUM_COLS] = scaler.transform(row[NUM_COLS])
        prob = model.predict_proba(row_sc)[0, 1]

        # Display Result
        st.divider()
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Risk Probability", f"{prob*100:.2f}%")
        if prob > 0.45:
            res_c2.error("DIAGNOSIS: HIGH RISK OF DEFICIENCY")
        else:
            res_c2.success("DIAGNOSIS: LOW RISK / NORMAL")

        # --- SHAP GRID BELOW PREDICTION ---
        st.subheader("Explainable AI: Feature Importance Grid")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_sc)

        shap_col1, shap_col2 = st.columns(2)
        
        with shap_col1:
            st.write("**Local Impact (Current Patient)**")
            fig_local, ax_local = plt.subplots()
            # Bar plot showing which features pushed the risk up/down for this specific person
            shap.plots.bar(shap.Explanation(values=shap_values[0], data=row.iloc[0], feature_names=X.columns), show=False)
            st.pyplot(fig_local)
            plt.close()

        with shap_col2:
            st.write("**Global Logic (Model Overview)**")
            fig_global, ax_global = plt.subplots()
            shap.summary_plot(explainer.shap_values(x_test_sc), x_test_sc, plot_type='bar', show=False)
            st.pyplot(fig_global)
            plt.close()

with tab_dca:
    st.header("Clinical Decision Curve Analysis")
    
    # Calculate values for DCA
    y_probs_test = model.predict_proba(x_test_sc)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    
    nb_model = calculate_net_benefit(y_test, y_probs_test, thresholds)
    
    # Reference: Treat All
    prevalence = np.mean(y_test)
    nb_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    
    # Plotting
    fig_dca, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, nb_model, label='CatBoost Prediction Model', color='blue', lw=2.5)
    ax.plot(thresholds, nb_all, label='Treat All Strategy', color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', label='Treat None Strategy', lw=1.5)
    
    ax.set_ylim(-0.05, prevalence + 0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Threshold Probability (Clinical Cut-off)')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Clinical Utility: Net Benefit vs Treatment Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig_dca)
    st.info("The DCA illustrates the 'Net Benefit' of using the model. The model is clinically useful when its blue line is higher than both the 'Treat All' (red) and 'Treat None' (black) lines.")
