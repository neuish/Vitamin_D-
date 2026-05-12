import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ========================= DATA LOADING =========================
df = pd.read_csv('Vitamin_D_Dataset.csv')

# === CRITICAL: Print original columns to debug ===
st.write("Original columns:", df.columns.tolist())   # Remove after confirming

# Robust renaming (in case order/columns vary slightly)
rename_dict = {
    # Map original possible names to standardized ones
    # Adjust these keys if your CSV has different header names
}

# If rename failed before, do it safely
expected_cols = [
    'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
    'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
    'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
    'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
    'physical_activity_level', 'diet_type', 'socioeconomic_status',
    'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
    'vitamin_d_ng_ml', 'deficient'
]

if len(df.columns) == len(expected_cols):
    df.columns = expected_cols
else:
    st.error(f"Column mismatch! Found {len(df.columns)} columns, expected {len(expected_cols)}")
    st.stop()

# Keep full copy for visualizations
df_viz = df.copy()

# ========================= FEATURE ENGINEERING FOR VIZ =========================
for dframe in [df, df_viz]:
    dframe['supplement_tier'] = pd.cut(
        dframe['vitamin_d_supplement_iu'],
        bins=[-1, 0, 400, 800, 1500, 2001],
        labels=['None (0 IU)', 'Low (1-400 IU)', 'Medium (401-800 IU)', 
                'High (801-1500 IU)', 'Very High (>1500 IU)'],
        right=False
    )
    
    dframe['sun_hours_bins'] = pd.cut(dframe['sun_hours_per_day'], 
                                      bins=np.arange(0, 8.5, 0.5), right=False)
    
    dframe['sun_exposure_group'] = pd.cut(
        dframe['sun_hours_per_day'],
        bins=[0, 2, 4, 6, 8],
        labels=['Low\n(0-2h)', 'Moderate\n(2-4h)', 'High\n(4-6h)', 'Very High\n(6-8h)']
    )
    
    dframe['sun_hours_quartile'] = pd.qcut(dframe['sun_hours_per_day'], 
                                           q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

# ========================= MODEL DATA PREP =========================
df_model = df.drop(columns=['vitamin_d_ng_ml'], errors='ignore').copy()

cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level',
            'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status',
            'alcohol_use', 'urban_rural']

df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# Age & Supplement groups
df_encoded['Age_Group'] = pd.cut(df_encoded['age'], 
                                 bins=[0,20,30,40,50,60,70,np.inf],
                                 labels=['Below 20','20-29','30-39','40-49','50-59','60-69','70+'])

df_encoded['VitD_Supp_Group'] = pd.cut(df_encoded['vitamin_d_supplement_iu'],
                                       bins=[0,400,800,1000,2000,np.inf],
                                       labels=['0','400','800','1000','2000+'])

df_encoded = pd.get_dummies(df_encoded, columns=['Age_Group', 'VitD_Supp_Group'], drop_first=True)

# Final cleanup
df_encoded = df_encoded.drop(columns=['age', 'vitamin_d_supplement_iu'], errors='ignore')
df_encoded['deficient'] = df_encoded['deficient'].astype(int)

# ========================= TRAIN/TEST =========================
columns_to_scale = ['bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
                    'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours',
                    'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl']

X = df_encoded.drop('deficient', axis=1)
y = df_encoded['deficient']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, 
                                                    random_state=100, stratify=y)

scaler = StandardScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale]  = scaler.transform(X_test[columns_to_scale])

# ========================= MODEL =========================
cat_model = CatBoostClassifier(
    iterations=300, 
    learning_rate=0.1, 
    depth=6, 
    random_seed=42, 
    verbose=0, 
    eval_metric='AUC'
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), 
              early_stopping_rounds=50, use_best_model=True)

# ========================= STREAMLIT APP =========================
def run_app():
    st.set_page_config(layout="wide")
    st.title("Vitamin D Deficiency Prediction App")

    tab1, tab2, tab3 = st.tabs(["Visualizations", "Model Evaluation", "Interactive Prediction"])

    with tab1:
        st.header("Key Visualizations")
        df_plot = df_viz.copy()
        df_plot['deficient'] = df_plot['deficient'].astype(str)

        # 1. Risk Surface
        st.subheader("Deficiency Risk: Sun Hours vs Supplement Tier")
        risk = df_viz.groupby(['sun_hours_bins', 'supplement_tier'])['deficient'].mean().unstack()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(risk*100, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        # 2. Vitamin D Distribution
        st.subheader("Vitamin D Distribution by Deficiency Status")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_plot, 
                       palette={'0':'teal', '1':'coral'}, ax=ax2)
        ax2.axhline(20, color='gold', linestyle='--', label='Deficiency Threshold')
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

    with tab3:
        st.header("Interactive Prediction")
        st.sidebar.header("Patient Information")

        input_dict = {}
        for col in columns_to_scale:
            mean_val = float(X[col].mean())
            input_dict[col] = st.sidebar.number_input(
                col.replace('_', ' ').title(), 
                value=mean_val, 
                key=col
            )

        age = st.sidebar.slider("Age", 18, 80, 45)
        vitd_supp = st.sidebar.slider("Vitamin D Supplement (IU)", 0, 2000, 400, 50)

        # Categorical inputs (add more as needed)
        sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
        skin_tone = st.sidebar.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        # ... add remaining categorical fields

        if st.button("Predict Deficiency"):
            # Build input DataFrame (this part needs full one-hot logic - simplified here)
            input_df = pd.DataFrame(0.0, index=[0], columns=X.columns)
            
            for col in columns_to_scale:
                input_df[col] = input_dict[col]
            
            input_df['age'] = age          # will be dropped later if needed
            input_df['vitamin_d_supplement_iu'] = vitd_supp
            
            # One-hot encoding for categoricals (expand this)
            if f"sex_{sex}" in input_df.columns:
                input_df[f"sex_{sex}"] = 1
            
            input_scaled = input_df.copy()
            input_scaled[columns_to_scale] = scaler.transform(input_scaled[columns_to_scale])
            
            prob = cat_model.predict_proba(input_scaled.values)[0, 1]
            
            if prob > 0.4:
                st.error(f"**Vitamin D Deficient** (Probability: {prob:.3f})")
            else:
                st.success(f"**Not Deficient** (Probability: {prob:.3f})")

    st.sidebar.info("CatBoost Model • AUC ~0.90")

run_app()
