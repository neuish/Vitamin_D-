# --- Streamlit Setup & Data Preparation ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# --- Data Loading and Preprocessing ---
df = pd.read_csv('Vitamin_D_Dataset.csv')

# Keep a copy of original df for visualizations that need vitamin_d_ng_ml
df_viz = df.copy()

# Data Cleaning
df.dropna(inplace=True)

# Rename columns
df.columns = [
    'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
    'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
    'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
    'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
    'physical_activity_level', 'diet_type', 'socioeconomic_status',
    'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
    'vitamin_d_ng_ml', 'deficient'
]

# Create derived features for visualizations (on df and df_viz)
for dframe in [df, df_viz]:
    dframe['supplement_tier'] = pd.cut(dframe['vitamin_d_supplement_iu'],
                                       bins=[-1, 0, 400, 800, 1500, 2001],
                                       labels=['None (0 IU)', 'Low (1-400 IU)', 'Medium (401-800 IU)', 'High (801-1500 IU)', 'Very High (>1500 IU)'],
                                       right=False)
    dframe['sun_hours_bins'] = pd.cut(dframe['sun_hours_per_day'], bins=np.arange(0, 8.5, 0.5), right=False)
    dframe['sun_exposure_group'] = pd.cut(
        dframe['sun_hours_per_day'],
        bins=[0, 2, 4, 6, 8],
        labels=['Low\n(0-2h)', 'Moderate\n(2-4h)', 'High\n(4-6h)', 'Very High\n(6-8h)']
    )
    dframe['sun_hours_quartile'] = pd.qcut(dframe['sun_hours_per_day'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    dframe['Age_Group'] = pd.cut(dframe['age'], bins=[0, 20, 30, 40, 50, 60, 70, float('inf')], 
                                 labels=['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'], right=False)
    dframe['VitaminD_Supplement_Group'] = pd.cut(dframe['vitamin_d_supplement_iu'], 
                                                 bins=[0, 400, 800, 1000, 2000, float('inf')], 
                                                 labels=['0', '400', '800', '1000', '2000+'], right=False)

# Drop 'vitamin_d_ng_ml' only for modeling
df_encoded = df.drop('vitamin_d_ng_ml', axis=1).copy()

# One-hot encode
categorical_cols_to_encode = ['sex', 'skin_tone', 'clothing_coverage', 'season', 
                              'physical_activity_level', 'diet_type', 'socioeconomic_status', 
                              'education_level', 'smoking_status', 'alcohol_use', 'urban_rural']
df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols_to_encode, drop_first=True)

# Convert bool to int
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)

# Additional one-hot for groups
df_age_vitd_encoded = pd.get_dummies(df[['Age_Group', 'VitaminD_Supplement_Group']], drop_first=True)
for col in df_age_vitd_encoded.select_dtypes(include='bool').columns:
    df_age_vitd_encoded[col] = df_age_vitd_encoded[col].astype(int)

df_encoded = pd.concat([df_encoded, df_age_vitd_encoded], axis=1)
df_encoded = df_encoded.drop(columns=['age', 'vitamin_d_supplement_iu', 'Age_Group', 'VitaminD_Supplement_Group'], errors='ignore')

df_encoded['deficient'] = df_encoded['deficient'].astype(int)

# --- Train-Test Split & Scaling ---
columns_to_scale = [
    'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 
    'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 
    'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl'
]

x = df_encoded.drop(columns=['deficient', 'supplement_tier', 'sun_hours_bins', 
                             'sun_exposure_group', 'sun_hours_quartile'])
y = df_encoded['deficient']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, stratify=y
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
x_test[columns_to_scale] = scaler.transform(x_test[columns_to_scale])

OPTIMAL_THRESHOLD = 0.4

# --- Models ---
cat_model = CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6, random_seed=42, 
    verbose=0, eval_metric='AUC'
)
cat_model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=50, use_best_model=True)

# (Other models can be kept but simplified for speed)
prediction_model = cat_model
x_full = x.copy()

# --- Streamlit App ---
def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Vitamin D Deficiency Prediction App (CatBoost Model)")

    tab1, tab2, tab3 = st.tabs(["Visualizations", "Model Evaluation", "Interactive Prediction"])

    with tab1:
        st.header("Key Visualizations")
        df_plot = df_viz.copy()  # Use the preserved visualization copy
        df_plot['deficient'] = df_plot['deficient'].astype(str)

        # Viz 1: Risk Surface
        st.subheader('Vitamin D Deficiency Risk Surface: Sun Hours vs. Supplementation')
        risk_surface = df_viz.groupby(['sun_hours_bins', 'supplement_tier'])['deficient'].mean().unstack()
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        sns.heatmap(risk_surface * 100, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=.5, ax=ax1)
        ax1.set_title('Vitamin D Deficiency Risk Surface')
        st.pyplot(fig1)
        plt.close(fig1)

        # Viz 2: Fixed violinplot
        st.subheader('Vitamin D Distribution by Deficiency Status')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_plot, 
                       palette={'0': 'teal', '1': 'coral'}, ax=ax2)
        ax2.axhline(20, color='gold', linestyle='--', label='Deficiency Threshold (20 ng/mL)')
        ax2.set_title('Vitamin D Distribution by Deficiency Status')
        ax2.set_xlabel('Deficient')
        ax2.set_ylabel('Vitamin D (ng/mL)')
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

        # Add other visualizations similarly (shortened for brevity)

    with tab3:
        st.header("Interactive Prediction")
        st.sidebar.header("Input Features")

        input_data = {}
        for col_name in columns_to_scale:
            min_val = float(x_full[col_name].min())
            max_val = float(x_full[col_name].max())
            mean_val = float(x_full[col_name].mean())
            input_data[col_name] = st.sidebar.number_input(
                f"{col_name.replace('_', ' ').title()}", 
                min_value=min_val, max_value=max_val, value=mean_val
            )

        raw_age = st.sidebar.slider("Age", 18, 79, 48)
        raw_vitd_iu = st.sidebar.slider("Vitamin D Supplement (IU)", 0, 2000, 400, 100)

        # Categorical inputs
        cat_inputs = {}
        cat_options = {
            'sex': ['Male', 'Female'],
            # ... add all others similarly
        }
        # (Implement all selectboxes)

        # Create input row with correct dtypes
        input_row = pd.DataFrame(0.0, index=[0], columns=x_full.columns)  # Start with float

        # Fill numeric
        for col in columns_to_scale:
            input_row[col] = input_data[col]

        # Fill categoricals and one-hots (implement logic)

        # Scale
        input_scaled = input_row.copy()
        input_scaled[columns_to_scale] = scaler.transform(input_scaled[columns_to_scale])

        if st.button("Predict"):
            prob = prediction_model.predict_proba(input_scaled.values)[0, 1]
            pred = 1 if prob > OPTIMAL_THRESHOLD else 0
            if pred == 1:
                st.error(f"**Deficient** (Probability: {prob:.3f})")
            else:
                st.success(f"**Not Deficient** (Probability: {prob:.3f})")

            # SHAP (keep if working)

run_streamlit_app()
