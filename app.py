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

# --- 1. Data Loading ---
@st.cache_data
def load_and_preprocess():
    # Loading exactly as named in the original project
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    
    # Simple feature engineering for age/supplements
    df_ml = pd.get_dummies(df, columns=[
        'Sex', 'Skin Tone', 'Clothing Coverage', 'Season', 
        'Physical Activity Level', 'Diet Type', 'Socioeconomic Status', 
        'Education Level', 'Smoking Status', 'Alcohol Use', 'Urban/Rural'
    ], drop_first=True)
    
    df_ml['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['Vitamin D Supplement (IU)'], bins=[-1, 400, 800, 1000, 2000, 20000], labels=['0', '400', '800', '1000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
        
    return df, df_ml

df_raw, df_ml = load_and_preprocess()

# --- 2. Model Setup ---
X = df_ml.drop(columns=['Deficient', 'Vitamin D (ng/ml)', 'Age', 'Vitamin D Supplement (IU)'], errors='ignore')
y = df_ml['Deficient']
NUM_COLS = ['BMI', 'Sun Hours Per Day', 'Latitude (Deg)', 'Diet Score', 'Sleep Hours']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
scaler = StandardScaler()
x_train_sc = x_train.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc = x_test.copy()
x_test_sc[NUM_COLS] = scaler.transform(x_test[NUM_COLS])

@st.cache_resource
def train_model():
    model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
    model.fit(x_train_sc, y_train)
    return model

model = train_model()

# --- 3. Streamlit UI Layout ---
st.set_page_config(page_title="Vitamin D Predictor", layout="wide")
st.title("☀️ Vitamin D Deficiency Prediction")

st.subheader("Patient Diagnostic Profile")
with st.container():
    # 2-Column Layout for Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        i_sun = st.slider("Sun Exposure (Hrs/Day)", 0.0, 10.0, 2.0)
        i_season = st.selectbox("Season", df_raw['Season'].unique())
        i_bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
        i_age_grp = st.selectbox("Age Group", ['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
        i_supp = st.selectbox("Vitamin D Supplement (IU)", ['0', '400', '800', '1000', '2000+'])

    with col2:
        i_skin = st.selectbox("Skin Tone", df_raw['Skin Tone'].unique())
        i_lat = st.number_input("Latitude", -90.0, 90.0, 34.0)
        i_act = st.selectbox("Physical Activity", df_raw['Physical Activity Level'].unique())
        i_diet = st.slider("Diet Quality Score", 1, 10, 5)
        i_sleep = st.slider("Sleep (Hours)", 4, 12, 7)

predict_btn = st.button("Predict Risk & Generate SHAP Analysis", type="primary", use_container_width=True)

# --- 4. Prediction Results & SHAP Grid ---
if predict_btn:
    # Initialize with floats to avoid assignment errors
    input_row = pd.DataFrame(0.0, index=[0], columns=X.columns)
    
    # Map Numerical Values
    input_row.at[0, 'Sun Hours Per Day'] = i_sun
    input_row.at[0, 'BMI'] = i_bmi
    input_row.at[0, 'Latitude (Deg)'] = i_lat
    input_row.at[0, 'Diet Score'] = i_diet
    input_row.at[0, 'Sleep Hours'] = i_sleep
    
    # Map One-Hot Categories
    for c in [f'Season_{i_season}', f'Skin Tone_{i_skin}', f'Physical Activity Level_{i_act}', 
              f'Age_Group_{i_age_grp}', f'Supp_Group_{i_supp}']:
        if c in X.columns:
            input_row.at[0, c] = 1.0

    # Scale inputs
    input_scaled = input_row.copy()
    input_scaled[NUM_COLS] = scaler.transform(input_row[NUM_COLS])
    
    # Get Probability
    prob = model.predict_proba(input_scaled)[0, 1]

    st.divider()
    res1, res2 = st.columns(2)
    res1.metric("Deficiency Risk", f"{prob*100:.2f}%")
    if prob > 0.45:
        res2.error("Result: High Probability of Deficiency")
    else:
        res2.success("Result: Normal Range Expected")

    # SHAP Grid Section
    st.subheader("Feature Interpretation (SHAP Analysis)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    g1, g2 = st.columns(2)
    with g1:
        st.write("**Local Factor Impact**")
        fig_l, ax_l = plt.subplots()
        # Create a basic bar summary for the local prediction
        shap.plots.bar(shap.Explanation(values=shap_values[0], data=input_row.iloc[0], feature_names=X.columns), show=False)
        st.pyplot(fig_l)
        plt.close()

    with g2:
        st.write("**Global Feature Importance**")
        fig_g, ax_g = plt.subplots()
        shap.summary_plot(explainer.shap_values(x_test_sc), x_test_sc, plot_type='bar', show=False)
        st.pyplot(fig_g)
        plt.close()
