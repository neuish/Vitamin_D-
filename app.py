import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ML Libraries
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

import torch
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(layout="wide")
st.title("Vitamin D Deficiency Prediction App")


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("Vitamin_D_Dataset.csv")

# Remove missing values
df.dropna(inplace=True)

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
    'deficient'
]

# =========================================================
# KEEP ORIGINAL DF FOR VISUALIZATION
# =========================================================
df_plot = df.copy()

# =========================================================
# CREATE MODEL DATAFRAME
# =========================================================
df_model = df.copy()

# Create supplement tiers
df_model['supplement_tier'] = pd.cut(
    df_model['vitamin_d_supplement_iu'],
    bins=[-1, 0, 400, 800, 1500, 2001],
    labels=[
        'None (0 IU)',
        'Low (1-400 IU)',
        'Medium (401-800 IU)',
        'High (801-1500 IU)',
        'Very High (>1500 IU)'
    ],
    right=False
)

# Create sun bins
df_model['sun_hours_bins'] = pd.cut(
    df_model['sun_hours_per_day'],
    bins=np.arange(0, 8.5, 0.5),
    right=False
)

# Sun exposure groups
df_model['sun_exposure_group'] = pd.cut(
    df_model['sun_hours_per_day'],
    bins=[0, 2, 4, 6, 8],
    labels=[
        'Low (0-2h)',
        'Moderate (2-4h)',
        'High (4-6h)',
        'Very High (6-8h)'
    ]
)

# Quartiles
df_model['sun_hours_quartile'] = pd.qcut(
    df_model['sun_hours_per_day'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

# =========================================================
# AGE GROUPS
# =========================================================
age_bins = [0, 20, 30, 40, 50, 60, 70, float('inf')]
age_labels = [
    'Below 20',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70+'
]

df_model['Age_Group'] = pd.cut(
    df_model['age'],
    bins=age_bins,
    labels=age_labels,
    right=False
)

# =========================================================
# SUPPLEMENT GROUPS
# =========================================================
supp_bins = [0, 400, 800, 1000, 2000, float('inf')]
supp_labels = ['0', '400', '800', '1000', '2000+']

df_model['VitaminD_Supplement_Group'] = pd.cut(
    df_model['vitamin_d_supplement_iu'],
    bins=supp_bins,
    labels=supp_labels,
    right=False
)

# =========================================================
# DROP TARGET LEAKAGE COLUMN
# =========================================================
df_model = df_model.drop(columns=['vitamin_d_ng_ml'])

# =========================================================
# ENCODING
# =========================================================
categorical_cols = [
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
    'Age_Group',
    'VitaminD_Supplement_Group'
]

df_encoded = pd.get_dummies(
    df_model,
    columns=categorical_cols,
    drop_first=True
)

# Convert bools to ints
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)

df_encoded['deficient'] = df_encoded['deficient'].astype(int)

# =========================================================
# FEATURES
# =========================================================
columns_to_scale = [
    'bmi',
    'sun_hours_per_day',
    'screen_time_hours',
    'calcium_intake_mg',
    'latitude_deg',
    'outdoor_activity_minutes',
    'diet_score',
    'sleep_hours',
    'cholesterol_mg_dl',
    'body_fat_percentage',
    'serum_calcium_mg_dl'
]

drop_cols = [
    'deficient',
    'supplement_tier',
    'sun_hours_bins',
    'sun_exposure_group',
    'sun_hours_quartile'
]

X = df_encoded.drop(columns=drop_cols)
y = df_encoded['deficient']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=42,
    stratify=y
)

# =========================================================
# SCALING
# =========================================================
scaler = StandardScaler()

x_train[columns_to_scale] = scaler.fit_transform(
    x_train[columns_to_scale]
)

x_test[columns_to_scale] = scaler.transform(
    x_test[columns_to_scale]
)

# =========================================================
# MODELS
# =========================================================

# Logistic Regression
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

lr_model.fit(x_train, y_train)

y_prob_lr = lr_model.predict_proba(x_test)[:, 1]

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(x_train, y_train)

y_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]

# CatBoost
cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)

cat_model.fit(x_train, y_train)

y_prob_cat = cat_model.predict_proba(x_test)[:, 1]

# TabNet
tabnet_model = TabNetClassifier(
    seed=42,
    verbose=0
)

tabnet_model.fit(
    X_train=x_train.values,
    y_train=y_train.values,
    eval_set=[(x_test.values, y_test.values)],
    eval_name=['test'],
    eval_metric=['auc'],
    max_epochs=50,
    patience=10
)

y_prob_tab = tabnet_model.predict_proba(x_test.values)[:, 1]

# =========================================================
# SHAP
# =========================================================
explainer = shap.TreeExplainer(cat_model)

# =========================================================
# VISUALIZATION TAB
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Visualizations",
    "Model Evaluation",
    "Interactive Prediction"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    st.header("Key Visualizations")

    # =====================================================
    # Violin Plot
    # =====================================================
    st.subheader("Vitamin D Distribution by Deficiency Status")

    df_plot['deficient_label'] = df_plot['deficient'].map({
        0: 'Non-Deficient',
        1: 'Deficient'
    })

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    sns.violinplot(
        x='deficient_label',
        y='vitamin_d_ng_ml',
        data=df_plot,
        palette=['teal', 'coral'],
        ax=ax1
    )

    ax1.axhline(
        20,
        color='gold',
        linestyle='--',
        label='Threshold'
    )

    ax1.set_title("Vitamin D Distribution")
    ax1.legend()

    st.pyplot(fig1)

    # =====================================================
    # Scatter Plot
    # =====================================================
    st.subheader("Body Fat Percentage vs Vitamin D")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x='body_fat_percentage',
        y='vitamin_d_ng_ml',
        hue='deficient_label',
        data=df_plot,
        ax=ax2
    )

    ax2.axhline(
        20,
        color='red',
        linestyle='--'
    )

    st.pyplot(fig2)

# =========================================================
# TAB 2
# =========================================================
with tab2:

    st.header("Model Evaluation")

    def metrics(y_true, y_pred, y_prob):

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_prob)
        }

    model_table = pd.DataFrame([
        {
            'Model': 'Logistic Regression',
            **metrics(
                y_test,
                (y_prob_lr > 0.5).astype(int),
                y_prob_lr
            )
        },
        {
            'Model': 'XGBoost',
            **metrics(
                y_test,
                xgb_model.predict(x_test),
                y_prob_xgb
            )
        },
        {
            'Model': 'CatBoost',
            **metrics(
                y_test,
                cat_model.predict(x_test),
                y_prob_cat
            )
        },
        {
            'Model': 'TabNet',
            **metrics(
                y_test,
                tabnet_model.predict(x_test.values),
                y_prob_tab
            )
        }
    ])

    st.dataframe(model_table)

    # =====================================================
    # ROC CURVE
    # =====================================================
    st.subheader("ROC Curve")

    fig3, ax3 = plt.subplots(figsize=(8, 8))

    # LR
    fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
    ax3.plot(
        fpr,
        tpr,
        label=f'LR AUC={roc_auc_score(y_test, y_prob_lr):.3f}'
    )

    # XGB
    fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
    ax3.plot(
        fpr,
        tpr,
        label=f'XGB AUC={roc_auc_score(y_test, y_prob_xgb):.3f}'
    )

    # CAT
    fpr, tpr, _ = roc_curve(y_test, y_prob_cat)
    ax3.plot(
        fpr,
        tpr,
        label=f'CatBoost AUC={roc_auc_score(y_test, y_prob_cat):.3f}'
    )

    # TABNET
    fpr, tpr, _ = roc_curve(y_test, y_prob_tab)
    ax3.plot(
        fpr,
        tpr,
        label=f'TabNet AUC={roc_auc_score(y_test, y_prob_tab):.3f}'
    )

    ax3.plot([0, 1], [0, 1], 'k--')

    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")

    ax3.legend()

    st.pyplot(fig3)

# =========================================================
# TAB 3
# =========================================================
with tab3:

    st.header("Interactive Prediction")

    input_data = {}

    st.sidebar.header("Patient Inputs")

    # =====================================================
    # NUMERIC INPUTS
    # =====================================================
    for col in columns_to_scale:

        input_data[col] = st.sidebar.number_input(
            col.replace('_', ' ').title(),
            value=float(X[col].mean())
        )

    # =====================================================
    # DATAFRAME FIXED
    # =====================================================
    input_df = pd.DataFrame(
        0.0,
        index=[0],
        columns=X.columns
    )

    # Numeric values
    for col in columns_to_scale:

        if col in input_df.columns:
            input_df.loc[0, col] = input_data[col]

    # Scale
    input_df[columns_to_scale] = scaler.transform(
        input_df[columns_to_scale]
    )

    # =====================================================
    # PREDICT
    # =====================================================
    if st.button("Predict"):

        pred_prob = cat_model.predict_proba(input_df)[0, 1]

        pred = cat_model.predict(input_df)[0]

        if pred == 1:
            st.error(
                f"Vitamin D Deficient "
                f"(Probability: {pred_prob:.2f})"
            )
        else:
            st.success(
                f"Not Vitamin D Deficient "
                f"(Probability: {pred_prob:.2f})"
            )

        # =================================================
        # SHAP
        # =================================================
        st.subheader("SHAP Explanation")

        shap_values = explainer.shap_values(input_df)

        fig4, ax4 = plt.subplots(figsize=(10, 6))

        shap.summary_plot(
            shap_values,
            input_df,
            show=False
        )

        st.pyplot(fig4)
