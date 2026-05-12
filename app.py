import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import shap
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# --- Helper function to compute metrics (already defined in notebook) ---
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

# --- Decision Curve Analysis (already defined in notebook) ---
def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(net_benefit)

    return net_benefits


# --- Data Loading and Preprocessing Function ---
@st.cache_data
def load_data():
    # Load the dataset (replace with your actual path if running locally)
    # In Colab, we can assume the file is mounted, for a local app, ensure path is correct.
    df = pd.read_csv('/content/drive/MyDrive/Logistic Regression/Vitamin_D_Dataset.csv')

    # Rename columns
    df.columns = [
        'age',
        'bmi',
        'sun_hours_per_day',
        'screen_time_hours',
        'calcium_intake_mg',
        'vitamin_d_supplement_iu',
        'latitude_deg',
        'outdoor_activity_minutes',
        'diet_score',
        'sleep_hours',
        'cholesterol_mg_dl',
        'body_fat_percentage',
        'serum_calcium_mg_dl',
        'sex',
        'skin_tone',
        'clothing_coverage',
        'season',
        'physical_activity_level',
        'diet_type',
        'socioeconomic_status',
        'education_level',
        'smoking_status',
        'alcohol_use',
        'urban_rural',
        'vitamin_d_ng_ml',
        'deficient_label'
    ]
    df.rename(columns={'deficient_label': 'deficient'}, inplace=True)

    # Drop 'vitamin_d_ng_ml' as it's directly related to the target 'deficient'
    df_processed = df.drop('vitamin_d_ng_ml', axis=1)

    # Convert target to object before one-hot encoding other categorical columns
    df_processed['deficient'] = df_processed['deficient'].astype('object')

    # Create Age Group and Vitamin D Supplement Group
    age_bins = [0, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = ['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df_processed['Age_Group'] = pd.cut(df_processed['age'], bins=age_bins, labels=age_labels, right=False)

    delay_bins = [0, 400, 800, 1000, 2000, float('inf')]
    delay_labels = ['0', '400', '800', '1000', '2000+']
    df_processed['VitaminD_Supplement_Group'] = pd.cut(df_processed['vitamin_d_supplement_iu'], bins=delay_bins, labels=delay_labels, right=False)

    # Columns for one-hot encoding
    categorical_cols_to_encode = [
        'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'Age_Group', 'VitaminD_Supplement_Group'
    ]

    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols_to_encode, drop_first=True)

    # Convert boolean columns to integers
    for col_bool in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col_bool] = df_encoded[col_bool].astype(int)

    # Drop original age and vitamin_d_supplement_iu as their grouped versions are encoded
    df_encoded = df_encoded.drop(columns=['age', 'vitamin_d_supplement_iu'])

    # Define features (x) and target (y)
    # Ensure 'deficient' is int for model training
    y = df_encoded['deficient'].astype(int)
    x = df_encoded.drop(columns=['deficient'])

    # Remove original visualization-specific columns if they exist
    # These columns ('supplement_tier', 'sun_hours_bins', 'sun_exposure_group', 'sun_hours_quartile')
    # were created on the 'df' DataFrame before it was copied to df_processed and encoded.
    # They are not present in 'x' after the above preprocessing steps.
    # However, if you explicitly add them back to df_encoded for other purposes, they might need to be dropped.
    # For this current flow, they are not part of 'x'.


    # Split data to fit scaler and RFE
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.7, random_state=100, stratify=y
    )

    # Identify continuous columns for scaling
    columns_to_scale = [
        'bmi', 'sun_hours_per_day', 'screen_time_hours',
        'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl',
        'body_fat_percentage', 'serum_calcium_mg_dl'
    ]

    # Scale continuous features
    scaler = StandardScaler()
    x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
    x_test[columns_to_scale] = scaler.transform(x_test[columns_to_scale])

    # RFE for feature selection (Logistic Regression)
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(estimator=logreg, n_features_to_select=15) # n_features_to_select from previous notebook cells
    rfe = rfe.fit(x_train, y_train)
    selected_features_rfe = x_train.columns[rfe.support_]

    # Prepare final x_train, x_test with selected features
    x_train_rfe = x_train[selected_features_rfe]
    x_test_rfe = x_test[selected_features_rfe]

    return df, x, y, x_train, y_train, x_test, y_test, scaler, selected_features_rfe, x_train_rfe, x_test_rfe

# --- Model Training (Caching models) ---
@st.cache_resource
def train_models(x_train, y_train, x_test, y_test, selected_features_rfe):

    # --- Logistic Regression ---
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    # Fit on RFE selected features
    logreg_model.fit(x_train[selected_features_rfe], y_train)

    # --- XGBoost ---
    xgb_base = XGBClassifier(colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }
    grid_search = GridSearchCV(xgb_base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(x_train, y_train)
    xgb_model = grid_search.best_estimator_

    # --- CatBoost ---
    cat_model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_seed=42, verbose=0, eval_metric='AUC')
    cat_model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=50, use_best_model=True)

    # --- TabNet ---
    tabnet_model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.3, n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=1e-2),
        scheduler_params={"step_size": 30, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='sparsemax',
        seed=42, verbose=0
    )
    tabnet_model.fit(
        X_train=x_train.values, y_train=y_train.values,
        eval_set=[(x_test.values, y_test.values)],
        eval_name=['test'], eval_metric=['auc'],
        max_epochs=50, patience=10, batch_size=256, virtual_batch_size=128
    )

    return logreg_model, xgb_model, cat_model, tabnet_model


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Vitamin D Deficiency Prediction")
st.title("💊 Vitamin D Deficiency Prediction")

# Load data and models
df_original, x_full, y_full, x_train_scaled, y_train_orig, x_test_scaled, y_test_orig, scaler, selected_features_rfe, x_train_rfe, x_test_rfe = load_data()
logreg_model, xgb_model, cat_model, tabnet_model = train_models(x_train_scaled, y_train_orig, x_test_scaled, y_test_orig, selected_features_rfe)


# Re-convert y_test_orig to int for consistent use in metrics
y_test_orig = y_test_orig.astype(int)

# Precompute predictions and probabilities for evaluation tab
OPTIMAL_THRESHOLD = 0.4 # From notebook analysis

# LR predictions on RFE selected features
y_prob_lr = logreg_model.predict_proba(x_test_rfe)[:, 1]
y_pred_lr = (y_prob_lr > OPTIMAL_THRESHOLD).astype(int)

# XGBoost predictions
y_prob_xgb = xgb_model.predict_proba(x_test_scaled)[:, 1]
y_pred_xgb = xgb_model.predict(x_test_scaled)

# CatBoost predictions
y_prob_cat = cat_model.predict_proba(x_test_scaled)[:, 1]
y_pred_cat = cat_model.predict(x_test_scaled)

# TabNet predictions
y_prob_tab = tabnet_model.predict_proba(x_test_scaled.values)[:, 1] # TabNet expects numpy
y_pred_tab = tabnet_model.predict(x_test_scaled.values)


# Create tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Model Evaluation", "Prediction"])

with tab1:
    st.header("Dashboard: Data Exploration")

    # Vitamin D Distribution & Deficiency Threshold
    st.subheader("1. Vitamin D Distribution & Deficiency Threshold")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_original, palette={0: 'teal', 1: 'coral'}, hue='deficient', legend=False, ax=ax)
    ax.axhline(20, color='gold', linestyle='--', label='Deficiency Threshold')
    ax.set_title('Vitamin D Distribution by Deficiency Status', fontsize=16)
    ax.set_xlabel('Deficient (0: No, 1: Yes)', fontsize=12)
    ax.set_ylabel('Vitamin D (ng/mL)', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Deficient', 'Deficient'])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Age Distribution
    st.subheader("2. Age Distribution for Deficient vs. Non-Deficient")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_original, x='age', hue='deficient', kde=True, palette={0: 'teal', 1: 'coral'}, stat='density', common_norm=False, ax=ax)
    ax.set_title('Age Distribution for Vitamin D Deficient vs. Non-Deficient', fontsize=16)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Body Fat Percentage vs. Vitamin D Levels
    st.subheader("3. Body Fat Percentage vs. Vitamin D Levels")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', data=df_original, palette={0: 'teal', 1: 'coral'}, alpha=0.6, ax=ax)
    ax.axhline(20, linestyle='--', color='gold', label='Deficiency Threshold')
    ax.set_title('Body Fat Percentage vs. Vitamin D Levels', fontsize=16)
    ax.set_xlabel('Body Fat Percentage', fontsize=12)
    ax.set_ylabel('Vitamin D (ng/mL)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Deficient')
    st.pyplot(fig)

    # Sun Exposure vs. Vitamin D Level by Skin Tone
    st.subheader("4. Sun Exposure vs. Vitamin D Level by Skin Tone")
    df_original['sun_exposure_group'] = pd.cut(
        df_original['sun_hours_per_day'],
        bins=[0, 2, 4, 6, 8],
        labels=['Low\n(0-2h)', 'Moderate\n(2-4h)', 'High\n(4-6h)', 'Very High\n(6-8h)']
    )
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df_original, x='sun_exposure_group', y='vitamin_d_ng_ml', hue='skin_tone', palette='magma', split=False, inner='quartile', linewidth=1.5, ax=ax)
    ax.axhline(20, color='red', linestyle='--', linewidth=2, label='Deficiency Threshold')
    ax.set_title('Vitamin D Distribution Across Sun Exposure Levels and Skin Tone', fontsize=20, weight='bold', pad=20)
    ax.set_xlabel('Daily Sun Exposure')
    ax.set_ylabel('Serum Vitamin D (ng/mL)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(title='Skin Tone', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # Stacked Bar Chart of Deficiency Prevalence by Skin Tone and Season
    st.subheader("5. Deficiency Prevalence by Skin Tone and Season")
    deficiency_crosstab = df_original.groupby(['skin_tone', 'season'])['deficient'].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 7))
    deficiency_crosstab.plot(kind='bar', stacked=True, cmap='viridis', ax=ax)
    ax.set_title('Vitamin D Deficiency Prevalence by Skin Tone and Season', fontsize=16)
    ax.set_xlabel('Skin Tone', fontsize=12)
    ax.set_ylabel('Proportion Deficient', fontsize=12)
    ax.tick_params(axis='x', rotation=45, ha='right')
    ax.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    # Heatmap of Median Vitamin D by Supplementation Tier and Sun Hours Quartile
    st.subheader("6. Median Vitamin D by Supplementation Tier and Sun Hours Quartile")
    df_original['supplement_tier'] = pd.cut(df_original['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 2001], labels=['None', 'Low (<=400)', 'Medium (401-800)', 'High (801-1500)', 'Very High (>1500)'], right=False)
    df_original['sun_hours_quartile'] = pd.qcut(df_original['sun_hours_per_day'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    median_vd_heatmap = df_original.groupby(['supplement_tier', 'sun_hours_quartile'])['vitamin_d_ng_ml'].median().unstack()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(median_vd_heatmap, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, linecolor='lightgrey', ax=ax)
    ax.set_title('Median Vitamin D (ng/mL) by Supplementation Tier and Sun Hours Quartile', fontsize=16)
    ax.set_xlabel('Sun Hours Per Day Quartile', fontsize=12)
    ax.set_ylabel('Vitamin D Supplementation Tier', fontsize=12)
    ax.tick_params(axis='x', rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("7. Correlation Heatmap")
    # Ensure df_encoded is recreated as in load_data to get all cols for correlation
    _, df_for_corr, _, _, _, _, _, _, _, _, _ = load_data()
    df_for_corr['deficient'] = df_for_corr['deficient'].astype(int)
    df_numeric_for_corr = df_for_corr.drop(columns=['supplement_tier', 'sun_hours_bins', 'sun_exposure_group', 'sun_hours_quartile'], errors='ignore')

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(df_numeric_for_corr.corr(), cmap='coolwarm', annot=False, fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap of Features', fontsize=16)
    st.pyplot(fig)


with tab2:
    st.header("Model Evaluation")

    st.subheader("1. Model Comparison Table")
    model_comparison = pd.DataFrame([
        {'Model': 'Logistic Regression', **get_metrics(y_test_orig, y_pred_lr, y_prob_lr)},
        {'Model': 'XGBoost', **get_metrics(y_test_orig, y_pred_xgb, y_prob_xgb)},
        {'Model': 'CatBoost', **get_metrics(y_test_orig, y_pred_cat, y_prob_cat)},
        {'Model': 'TabNet', **get_metrics(y_test_orig, y_pred_tab, y_prob_tab)}
    ])
    st.dataframe(model_comparison.set_index('Model'))

    st.subheader("2. ROC Curves Comparison")
    fig, ax = plt.subplots(figsize=(7, 7))
    fpr_lr, tpr_lr, _ = roc_curve(y_test_orig, y_prob_lr)
    ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test_orig, y_prob_lr):.3f})')

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test_orig, y_prob_xgb)
    ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y_test_orig, y_prob_xgb):.3f})')

    fpr_cat, tpr_cat, _ = roc_curve(y_test_orig, y_prob_cat)
    ax.plot(fpr_cat, tpr_cat, label=f'CatBoost (AUC={roc_auc_score(y_test_orig, y_prob_cat):.3f})')

    fpr_tab, tpr_tab, _ = roc_curve(y_test_orig, y_prob_tab)
    ax.plot(fpr_tab, tpr_tab, label=f'TabNet (AUC={roc_auc_score(y_test_orig, y_prob_tab):.3f})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison of All Models')
    ax.legend(loc='lower right')
    st.pyplot(fig)

    st.subheader("3. Decision Curve Analysis")
    thresholds = np.linspace(0.01, 0.99, 100)

    nb_lr = decision_curve(y_test_orig, y_prob_lr, thresholds)
    nb_xgb = decision_curve(y_test_orig, y_prob_xgb, thresholds)
    nb_cat = decision_curve(y_test_orig, y_prob_cat, thresholds)
    nb_tab = decision_curve(y_test_orig, y_prob_tab, thresholds)

    prevalence = np.mean(y_test_orig)
    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    treat_none = [0 for _ in thresholds]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(thresholds, nb_lr, label='Logistic Regression')
    ax.plot(thresholds, nb_xgb, label='XGBoost')
    ax.plot(thresholds, nb_cat, label='CatBoost')
    ax.plot(thresholds, nb_tab, label='TabNet')

    ax.plot(thresholds, treat_all, linestyle='--', label='Treat All')
    ax.plot(thresholds, treat_none, linestyle='--', label='Treat None')

    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

with tab3:
    st.header("Interactive Prediction")
    st.write("Enter patient details to predict Vitamin D deficiency.")

    # Collect user inputs dynamically based on features used by the model
    # Use the x_full DataFrame for feature names and min/max/unique values

    input_data = {}
    # Continuous features
    continuous_cols_for_input = [
        'bmi', 'sun_hours_per_day', 'screen_time_hours',
        'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl',
        'body_fat_percentage', 'serum_calcium_mg_dl'
    ]

    for col in continuous_cols_for_input:
        min_val = float(x_full[col].min())
        max_val = float(x_full[col].max())
        mean_val = float(x_full[col].mean())
        input_data[col] = st.sidebar.number_input(f"Enter {col.replace('_', ' ').title()}", min_value=min_val, max_value=max_val, value=mean_val)

    # Categorical features - handle grouped Age and VitaminD Supplement separately to get raw input
    # Then reconstruct the one-hot encoded columns
    raw_age = st.sidebar.slider("Age", min_value=18, max_value=79, value=48)
    raw_vitamin_d_supplement_iu = st.sidebar.slider("Vitamin D Supplement (IU)", min_value=0, max_value=2000, value=400, step=100)

    # Map raw age to Age_Group
    age_bins_raw = [0, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels_raw = ['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    age_group_raw = pd.cut([raw_age], bins=age_bins_raw, labels=age_labels_raw, right=False)[0]
    
    # Map raw vitamin D supplement to VitaminD_Supplement_Group
    delay_bins_raw = [0, 400, 800, 1000, 2000, float('inf')]
    delay_labels_raw = ['0', '400', '800', '1000', '2000+']
    vitamin_d_group_raw = pd.cut([raw_vitamin_d_supplement_iu], bins=delay_bins_raw, labels=delay_labels_raw, right=False)[0]

    # Original categorical columns that were one-hot encoded
    categorical_cols_original = {
        'sex': ['Female', 'Male'],
        'skin_tone': ['Dark', 'Light', 'Medium'],
        'clothing_coverage': ['High', 'Low', 'Medium'],
        'season': ['Monsoon', 'Spring', 'Summer', 'Winter'],
        'physical_activity_level': ['High', 'Low', 'Moderate', 'Sedentary'],
        'diet_type': ['Mixed', 'Non-veg', 'Veg'],
        'socioeconomic_status': ['High', 'Low', 'Middle'],
        'education_level': ['Graduate', 'Postgraduate', 'Secondary', 'Undergraduate'],
        'smoking_status': ['Non-smoker', 'Smoker'],
        'alcohol_use': ['No', 'Yes'],
        'urban_rural': ['Rural', 'Urban']
    }

    for col, options in categorical_cols_original.items():
        input_data[col] = st.sidebar.selectbox(f"Select {col.replace('_', ' ').title()}", options)

    # Create a DataFrame for the single prediction input
    input_df = pd.DataFrame(columns=x_full.columns)

    # Fill continuous columns
    for col in continuous_cols_for_input:
        input_df.loc[0, col] = input_data[col]

    # Fill one-hot encoded columns for original categorical features
    for col, options in categorical_cols_original.items():
        for option in options:
            if option != options[0]: # drop_first=True equivalent
                col_name = f"{col}_{option}"
                if col_name in x_full.columns:
                    input_df.loc[0, col_name] = 1 if input_data[col] == option else 0

    # Fill one-hot encoded columns for Age_Group
    for label in age_labels_raw:
        if label != age_labels_raw[0]: # drop_first=True equivalent
            col_name = f"Age_Group_{label}"
            if col_name in x_full.columns:
                input_df.loc[0, col_name] = 1 if age_group_raw == label else 0

    # Fill one-hot encoded columns for VitaminD_Supplement_Group
    for label in delay_labels_raw:
        if label != delay_labels_raw[0]: # drop_first=True equivalent
            col_name = f"VitaminD_Supplement_Group_{label}"
            if col_name in x_full.columns:
                input_df.loc[0, col_name] = 1 if vitamin_d_group_raw == label else 0

    # Ensure all columns are present and in the correct order for scaling and prediction
    # Fill any missing columns (due to drop_first=True on some categories not selected) with 0
    for col in x_full.columns:
        if col not in input_df.columns:
            input_df.loc[0, col] = 0
    input_df = input_df[x_full.columns] # Reorder columns

    # Scale continuous features in the input DataFrame
    input_df_scaled = input_df.copy()
    input_df_scaled[continuous_cols_for_input] = scaler.transform(input_df_scaled[continuous_cols_for_input])

    # Select features for Logistic Regression (RFE selected features)
    input_df_rfe = input_df_scaled[selected_features_rfe]


    st.subheader("Prediction Result (Logistic Regression)")
    if st.button("Predict"):    
        prediction_prob = logreg_model.predict_proba(input_df_rfe)[0, 1]
        prediction_label = (prediction_prob > OPTIMAL_THRESHOLD).astype(int)

        if prediction_label == 1:
            st.error(f"The model predicts: **Vitamin D Deficient** (Probability: {prediction_prob:.2f})")
        else:
            st.success(f"The model predicts: **Not Vitamin D Deficient** (Probability: {prediction_prob:.2f})")

        st.subheader("Feature Explanations (SHAP for Logistic Regression)")
        st.write("How each feature contributes to this specific prediction:")

        # SHAP Explanation for Logistic Regression (Linear Model)
        # It's important to use the original scaled features for SHAP if the model was trained on them.
        # logreg_model was trained on x_train_rfe (scaled and RFE selected)

        # Extract coefficients and intercept for LinearExplainer
        # Need to reconstruct statsmodels GLM for proper explainer
        x_train_sm_for_shap = sm.add_constant(x_train_rfe)
        y_train_sm_for_shap = y_train_orig.astype(int) # Ensure int type
        sm_model = sm.GLM(y_train_sm_for_shap, x_train_sm_for_shap, family=sm.families.Binomial())
        sm_pred = sm_model.fit()

        model_coef = sm_pred.params[1:].values # Coefficients for features
        model_intercept = sm_pred.params[0] # Intercept

        # Create SHAP explainer (if not already created)
        explainer_lr = shap.LinearExplainer((model_coef, model_intercept), x_train_rfe)

        # Compute SHAP values for the single input
        shap_values_lr = explainer_lr.shap_values(input_df_rfe)

        # Visualize with force plot
        fig_force = shap.force_plot(
            explainer_lr.expected_value,
            shap_values_lr,
            input_df_rfe,
            matplotlib=True, 
            show=False # Prevent immediate display
        )
        st.pyplot(fig_force, bbox_inches='tight')

        st.write("--- ")
        st.write("Global Feature Importance (SHAP Summary Plot - Bar):")
        fig_bar, ax_bar = plt.subplots()
        shap.summary_plot(explainer_lr.shap_values(x_test_rfe), x_test_rfe, plot_type='bar', max_display=15, show=False)
        st.pyplot(fig_bar, bbox_inches='tight')

        st.write("--- ")
        st.write("Global Feature Impact (SHAP Summary Plot - Beeswarm):")
        fig_beeswarm, ax_beeswarm = plt.subplots()
        shap.summary_plot(explainer_lr.shap_values(x_test_rfe), x_test_rfe, max_display=15, show=False)
        st.pyplot(fig_beeswarm, bbox_inches='tight')
