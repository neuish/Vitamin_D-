import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# --- 1. Data & Model Setup ---
@st.cache_data
def load_and_prep():
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Feature Engineering (Matching your Notebook logic)
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 
                'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 
                'alcohol_use', 'urban_rural']
    
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], 
                                labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 20000], 
                                 labels=['0', '400', '800', '1000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    # Ensure all bools are ints for SHAP/CatBoost
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
        
    return df, df_ml

df_raw, df_ml = load_and_prep()
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'latitude_deg', 'diet_score', 'sleep_hours']

# Standard Scaling
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])

@st.cache_resource
def train_cat():
    model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
    model.fit(X_scaled, y)
    return model

model = train_cat()

# --- 2. UI Layout ---
st.title("🔬 Vitamin D Deficiency Predictor")

with st.expander("Patient Information Input", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        i_sun = st.slider("Sun Exposure (Hrs/Day)", 0.0, 10.0, 2.0)
        i_bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
        i_age_grp = st.selectbox("Age Group", ['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
        i_supp = st.selectbox("Vitamin D Supplement (IU)", ['0', '400', '800', '1000', '2000+'])
        i_sleep = st.slider("Sleep (Hours)", 4, 12, 7)

    with col2:
        i_lat = st.number_input("Latitude (Degrees)", 0.0, 70.0, 34.0)
        i_skin = st.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        i_season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Monsoon"])
        i_activity = st.selectbox("Physical Activity", ["Sedentary", "Low", "Moderate", "High"])
        i_diet = st.slider("Diet Quality Score", 1, 10, 6)

predict_btn = st.button("Run Prediction & SHAP Analysis", type="primary", use_container_width=True)

# --- 3. Prediction & SHAP Grid ---
if predict_btn:
    # Build input row (Initialize as float64 to avoid LossySetitemError)
    input_row = pd.DataFrame(0.0, index=[0], columns=X.columns)
    
    # Map Numerical
    input_row.at[0, 'sun_hours_per_day'] = i_sun
    input_row.at[0, 'bmi'] = i_bmi
    input_row.at[0, 'latitude_deg'] = i_lat
    input_row.at[0, 'diet_score'] = i_diet
    input_row.at[0, 'sleep_hours'] = i_sleep
    
    # Map Categorical (One-Hot)
    if f'Age_Group_{i_age_grp}' in X.columns: input_row.at[0, f'Age_Group_{i_age_grp}'] = 1.0
    if f'Supp_Group_{i_supp}' in X.columns: input_row.at[0, f'Supp_Group_{i_supp}'] = 1.0
    if f'skin_tone_{i_skin}' in X.columns: input_row.at[0, f'skin_tone_{i_skin}'] = 1.0
    if f'season_{i_season}' in X.columns: input_row.at[0, f'season_{i_season}'] = 1.0
    if f'physical_activity_level_{i_activity}' in X.columns: input_row.at[0, f'physical_activity_level_{i_activity}'] = 1.0
    
    # Scale
    input_scaled = input_row.copy()
    input_scaled[NUM_COLS] = scaler.transform(input_row[NUM_COLS])
    
    # Result
    prob = model.predict_proba(input_scaled)[0, 1]
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Deficiency Risk", f"{prob*100:.2f}%")
    if prob > 0.5:
        res_col2.error("Result: High Risk of Deficiency")
    else:
        res_col2.success("Result: Low Risk / Normal")

    # --- SHAP GRID ---
    st.subheader("Feature Impact (SHAP Interpretability)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    
    g1, g2 = st.columns(2)
    
    with g1:
        st.write("**Local Impact (Force Plot)**")
        # Creating a bar chart as a surrogate for force plot in Streamlit for better rendering
        shap_df = pd.DataFrame({
            'Feature': X.columns,
            'SHAP Value': shap_values[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)
        
        fig_local, ax_local = plt.subplots()
        colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Value']]
        ax_local.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        ax_local.set_title("Top 10 Factors for THIS Patient")
        st.pyplot(fig_local)

    with g2:
        st.write("**Global Importance (Summary)**")
        fig_global, ax_global = plt.subplots()
        shap.summary_plot(explainer.shap_values(X_scaled), X_scaled, plot_type="bar", show=False)
        st.pyplot(fig_global)
