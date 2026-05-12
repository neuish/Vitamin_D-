import sys
!{sys.executable} -m pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import shap
import warnings
warnings.filterwarnings("ignore")

# Set page config for Streamlit
st.set_page_config(layout="wide", page_title="Vitamin D Deficiency Prediction")

st.title("Vitamin D Deficiency Prediction App")

# --- 1. Data Loading and Preprocessing (Encapsulated) ---

@st.cache_data
def load_and_preprocess_data():
    df = None # Initialize df to None
    try:
        df = pd.read_csv('Vitamin_D_Dataset.csv')
    except FileNotFoundError:
        st.error("Vitamin_D_Dataset.csv not found. Please ensure it's in the same directory as the app.")
        st.stop() # Stop the Streamlit app if file not found
        # Return None for all expected outputs if file not found
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    # Data Cleaning and treating
    df.dropna(inplace=True)
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

    # Store df_raw for visualizations that use 'vitamin_d_ng_ml'
    df_raw_viz = df.copy()

    # Feature Engineering for visualizations
    df_raw_viz['supplement_tier'] = pd.cut(df_raw_viz['vitamin_d_supplement_iu'],
                                   bins=[-1, 0, 400, 800, 1500, 2001],
                                   labels=['None (0 IU)', 'Low (1-400 IU)', 'Medium (401-800 IU)', 'High (801-1500 IU)', 'Very High (>1500 IU)'],
                                   right=False)
    df_raw_viz['sun_hours_bins'] = pd.cut(df_raw_viz['sun_hours_per_day'], bins=np.arange(0, 8.5, 0.5), right=False)
    df_raw_viz['sun_exposure_group'] = pd.cut(
        df_raw_viz['sun_hours_per_day'],
        bins=[0, 2, 4, 6, 8],
        labels=[
            'Low\n(0-2h)',
            'Moderate\n(2-4h)',
            'High\n(4-6h)',
            'Very High\n(6-8h)'
        ]
    )
    df_raw_viz['sun_hours_quartile'] = pd.qcut(df_raw_viz['sun_hours_per_day'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    # Drop vitamin_d_ng_ml as 'deficient' is the target for modeling
    df = df.drop('vitamin_d_ng_ml', axis=1)

    # Convert target to object before encoding, then to int for modeling
    df['deficient'] = df['deficient'].astype('object')

    categorical_cols_to_encode = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 'alcohol_use', 'urban_rural']
    df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

    for col_bool in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col_bool] = df_encoded[col_bool].astype(int)

    # Grouping Age and VitaminD Supplement IU
    age_bins = [0, 20, 30, 40, 50, 60, 70, float('inf')]
    age_labels = ['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    delay_bins = [0, 400, 800, 1000, 2000, float('inf')]
    delay_labels = ['0', '400', '800', '1000', '2000+']

    df['Age_Group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df['VitaminD_Supplement_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=delay_bins, labels=delay_labels, right=False)

    df_age_vitd_encoded = pd.get_dummies(df[['Age_Group', 'VitaminD_Supplement_Group']], drop_first=True)
    for col_bool in df_age_vitd_encoded.select_dtypes(include='bool').columns:
        df_age_vitd_encoded[col_bool] = df_age_vitd_encoded[col_bool].astype(int)

    df_encoded = pd.concat([df_encoded, df_age_vitd_encoded], axis=1)

    # Drop original age and vitamin_d_supplement_iu after creating groups
    df_encoded = df_encoded.drop(columns=['age', 'vitamin_d_supplement_iu'])

    # Ensure target variable is numeric for modeling
    df_encoded['deficient'] = df_encoded['deficient'].astype(int)

    # Drop visualization-specific columns from df_encoded if they are still present
    columns_to_drop_from_encoded = ['supplement_tier', 'sun_hours_bins', 'sun_exposure_group', 'sun_hours_quartile']
    df_encoded = df_encoded.drop(columns=columns_to_drop_from_encoded, errors='ignore')

    x = df_encoded.drop(['deficient'], axis=1)
    y = df_encoded['deficient']

    # Identify continuous columns for scaling
    continuous_cols_to_scale = [
        'bmi', 'sun_hours_per_day', 'screen_time_hours',
        'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl',
        'body_fat_percentage', 'serum_calcium_mg_dl'
    ]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.7, random_state=100, stratify=y
    )

    # Scale continuous features
    scaler = StandardScaler()
    x_train[continuous_cols_to_scale] = scaler.fit_transform(x_train[continuous_cols_to_scale])
    x_test[continuous_cols_to_scale] = scaler.transform(x_test[continuous_cols_to_scale])

    # --- RFE for Logistic Regression feature selection ---
    logreg_rfe = LogisticRegression()
    rfe = RFE(estimator=logreg_rfe, n_features_to_select=15)
    rfe.fit(x_train, y_train)
    selected_lr_features = x_train.columns[rfe.support_]

    # --- Train Logistic Regression Model ---
    # Ensure y_train is integer type for sm.GLM
    y_train_int = y_train.astype(int)
    x_train_lr_sm = sm.add_constant(x_train[selected_lr_features])
    lr_model_sm = sm.GLM(y_train_int, x_train_lr_sm, family=sm.families.Binomial())
    lr_pred_model = lr_model_sm.fit()

    # --- Optimal Threshold from previous analysis ---
    OPTIMAL_THRESHOLD = 0.4

    # --- Train XGBoost Model ---
    xgb_base = XGBClassifier(
        colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    # Simplified training for Streamlit without GridSearchCV for speed
    xgb_model = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=200,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    xgb_model.fit(x_train, y_train)

    # --- Train CatBoost Model ---
    cat_model = CatBoostClassifier(
        iterations=300, learning_rate=0.1, depth=6, random_seed=42, verbose=0, eval_metric='AUC'
    )
    cat_model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=50, use_best_model=True)

    # --- Train TabNet Model ---
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

    return df_raw_viz, df_encoded, x_train, x_test, y_train, y_test, scaler, continuous_cols_to_scale, selected_lr_features, lr_pred_model, OPTIMAL_THRESHOLD, xgb_model, cat_model, tabnet_model

df_raw_viz, df_encoded, x_train, x_test, y_train, y_test, scaler, continuous_cols_to_scale, selected_lr_features, lr_pred_model, OPTIMAL_THRESHOLD, xgb_model, cat_model, tabnet_model = load_and_preprocess_data()

# Check if data loading was successful
if df_raw_viz is None:
    st.warning("Data could not be loaded. Some features of the app might not be available.")


# --- Helper function to compute metrics ---
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

# --- Helper function for ROC plot ---
def draw_roc(actual, probs, title="Receiver Operating Characteristic"):
    fpr, tpr, thresholds = roc_curve(actual, probs, drop_intermediate=False)
    auc_score = roc_auc_score(actual, probs)
    fig, ax = plt.subplots(figsize=(6, 4)) # Adjusted figure size
    ax.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig

# --- Helper function for Decision Curve Analysis ---
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

# --- Tabbed Interface ---
dashboard_tab, model_eval_tab, prediction_tab = st.tabs(["Dashboard", "Model Evaluation", "Prediction"])

# --- Dashboard Tab ---
with dashboard_tab:
    if df_raw_viz is not None:
        st.header("Exploratory Data Analysis")

        # 1. Vitamin D Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Vitamin D Distribution by Deficiency Status")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df_raw_viz, palette={0: 'teal', 1: 'coral'}, hue='deficient', legend=False, ax=ax)
            ax.axhline(20, color='gold', linestyle='--', label='Deficiency Threshold')
            ax.set_title('Vitamin D Distribution by Deficiency Status', fontsize=14)
            ax.set_xlabel('Deficient (0: No, 1: Yes)', fontsize=12)
            ax.set_ylabel('Vitamin D (ng/mL)', fontsize=12)
            ax.set_xticks([0, 1], ['Non-Deficient', 'Deficient'])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

        # 2. Age Distribution for Deficient vs. Non-Deficient
        with col2:
            st.subheader("2. Age Distribution for Vitamin D Deficient vs. Non-Deficient")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df_raw_viz, x='age', hue='deficient', kde=True, palette={0: 'teal', 1: 'coral'}, stat='density', common_norm=False, ax=ax)
            ax.set_title('Age Distribution for Vitamin D Deficient vs. Non-Deficient', fontsize=14)
            ax.set_xlabel('Age', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

        # 3. Body Fat Percentage vs. Vitamin D Levels
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. Body Fat Percentage vs. Vitamin D Levels")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', data=df_raw_viz, palette={0: 'teal', 1: 'coral'}, alpha=0.6, ax=ax)
            ax.axhline(20, linestyle='--', color='gold', label='Deficiency Threshold')
            ax.set_title('Body Fat Percentage vs. Vitamin D Levels', fontsize=14)
            ax.set_xlabel('Body Fat Percentage', fontsize=12)
            ax.set_ylabel('Vitamin D (ng/mL)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title='Deficient')
            plt.tight_layout()
            st.pyplot(fig)

        # 4. Sun Exposure vs Vitamin D Level by Skin Tone
        with col4:
            st.subheader("4. Sun Exposure vs Vitamin D Level by Skin Tone")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.violinplot(
                data=df_raw_viz,
                x='sun_exposure_group',
                y='vitamin_d_ng_ml',
                hue='skin_tone',
                palette='magma',
                split=False,
                inner='quartile',
                linewidth=1.5,
                ax=ax
            )
            ax.axhline(
                20,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Deficiency Threshold'
            )
            ax.set_title(
                'Vitamin D Distribution Across Sun Exposure Levels and Skin Tone',
                fontsize=14,
                weight='bold',
                pad=10
            )
            ax.set_xlabel('Daily Sun Exposure')
            ax.set_ylabel('Serum Vitamin D (ng/mL)')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(title='Skin Tone', bbox_to_anchor=(1.02, 1), loc='upper left')
            sns.despine(ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

        # 5. Deficiency Prevalence by Skin Tone and Season
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("5. Deficiency Prevalence by Skin Tone and Season")
            deficiency_crosstab = df_raw_viz.groupby(['skin_tone', 'season'])['deficient'].mean().unstack()
            fig, ax = plt.subplots(figsize=(7, 5))
            deficiency_crosstab.plot(kind='bar', stacked=True, cmap='viridis', ax=ax)
            ax.set_title('Vitamin D Deficiency Prevalence by Skin Tone and Season', fontsize=14)
            ax.set_xlabel('Skin Tone', fontsize=12)
            ax.set_ylabel('Proportion Deficient', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            ax.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

        # 6. Vitamin D Deficiency Risk Surface: Sun Hours vs. Supplementation
        with col6:
            st.subheader("6. Vitamin D Deficiency Risk Surface: Sun Hours vs. Supplementation")
            risk_surface = df_raw_viz.groupby(['sun_hours_bins', 'supplement_tier'])['deficient'].mean().unstack()
            overall_deficiency = df_raw_viz['deficient'].mean()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(risk_surface * 100, annot=True, fmt='.1f', cmap='RdYlGn_r',
                        center=overall_deficiency * 100, linewidths=.5, linecolor='lightgrey',
                        cbar_kws={'label': 'Deficiency Rate (%)'}, ax=ax)
            ax.set_title('Vitamin D Deficiency Risk Surface: Sun Hours vs. Supplementation', fontsize=14)
            ax.set_xlabel('Sun Hours Per Day Bins', fontsize=12)
            ax.set_ylabel('Vitamin D Supplement (IU) Tier', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        # 7. Heatmap of Median Vitamin D by Supplementation Tier and Sun Hours Quartile
        st.subheader("7. Median Vitamin D by Supplementation Tier and Sun Hours Quartile")
        median_vd_heatmap = df_raw_viz.groupby(['supplement_tier', 'sun_hours_quartile'])['vitamin_d_ng_ml'].median().unstack()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(median_vd_heatmap, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, linecolor='lightgrey', ax=ax)
        ax.set_title('Median Vitamin D (ng/mL) by Supplementation Tier and Sun Hours Quartile', fontsize=14)
        ax.set_xlabel('Sun Hours Per Day Quartile', fontsize=12)
        ax.set_ylabel('Vitamin D Supplementation Tier', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        # Correlation Heatmap for encoded features
        st.subheader("Correlation Heatmap of Encoded Features")
        # Ensure df_encoded has 'deficient' as numeric for correlation
        df_numeric_for_corr = df_encoded.copy()
        df_numeric_for_corr['deficient'] = df_numeric_for_corr['deficient'].astype(int) # Make sure it's int for corr

        fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted figure size for heatmap
        sns.heatmap(df_numeric_for_corr.corr(), cmap='coolwarm', annot=False, fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap of Encoded Features')
        st.pyplot(fig)
    else:
        st.write("Please upload the `Vitamin_D_Dataset.csv` file to view the dashboard.")


# --- Model Evaluation Tab ---
with model_eval_tab:
    if df_raw_viz is not None:
        st.header("Model Evaluation and Comparison")

        # --- Make predictions for all models on the test set ---
        # Logistic Regression
        x_test_lr_sm = sm.add_constant(x_test[selected_lr_features])
        y_prob_lr = lr_pred_model.predict(x_test_lr_sm)
        y_pred_lr = (y_prob_lr > OPTIMAL_THRESHOLD).astype(int)

        # XGBoost
        # x_test needs to be reindexed to match x_train for xgb_model
        x_test_xgb = x_test.reindex(columns=x_train.columns, fill_value=0)
        y_pred_xgb = xgb_model.predict(x_test_xgb)
        y_prob_xgb = xgb_model.predict_proba(x_test_xgb)[:, 1]

        # CatBoost
        y_pred_cat = cat_model.predict(x_test)
        y_prob_cat = cat_model.predict_proba(x_test)[:, 1]

        # TabNet
        y_pred_tab = tabnet_model.predict(x_test.values)
        y_prob_tab = tabnet_model.predict_proba(x_test.values)[:, 1]

        # Ensemble (LR + XGBoost)
        y_prob_ensemble = (y_prob_lr.values + y_prob_xgb) / 2
        y_pred_ensemble = (y_prob_ensemble > OPTIMAL_THRESHOLD).astype(int)


        # --- Model Comparison Table ---
        st.subheader("Model Performance on Test Set")
        model_comparison = pd.DataFrame([
            {'Model': 'Logistic Regression', **get_metrics(y_test, y_pred_lr, y_prob_lr)},
            {'Model': 'XGBoost', **get_metrics(y_test, y_pred_xgb, y_prob_xgb)},
            {'Model': 'CatBoost', **get_metrics(y_test, y_pred_cat, y_prob_cat)},
            {'Model': 'TabNet', **get_metrics(y_test, y_pred_tab, y_prob_tab)},
            {'Model': 'Ensemble (LR+XGB)', **get_metrics(y_test, y_pred_ensemble, y_prob_ensemble)}
        ])
        st.dataframe(model_comparison.set_index('Model'))

        # --- ROC Curves Comparison ---
        col_roc, col_dca = st.columns(2)
        with col_roc:
            st.subheader("ROC Curve Comparison")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4)) # Adjusted figure size
            ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

            fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
            ax_roc.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.3f})')

            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
            ax_roc.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y_test, y_prob_xgb):.3f})')

            fpr_cat, tpr_cat, _ = roc_curve(y_test, y_prob_cat)
            ax_roc.plot(fpr_cat, tpr_cat, label=f'CatBoost (AUC={roc_auc_score(y_test, y_prob_cat):.3f})')

            fpr_tab, tpr_tab, _ = roc_curve(y_test, y_prob_tab)
            ax_roc.plot(fpr_tab, tpr_tab, label=f'TabNet (AUC={roc_auc_score(y_test, y_prob_tab):.3f})')

            fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ensemble)
            ax_roc.plot(fpr_ens, tpr_ens, label=f'Ensemble (LR+XGB) (AUC={roc_auc_score(y_test, y_prob_ensemble):.3f})')

            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve Comparison of All Models')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)

        # --- Decision Curve Analysis ---
        with col_dca:
            st.subheader("Decision Curve Analysis")
            thresholds_dca = np.linspace(0.01, 0.99, 100)

            nb_lr = decision_curve(y_test, y_prob_lr, thresholds_dca)
            nb_xgb = decision_curve(y_test, y_prob_xgb, thresholds_dca)
            nb_cat = decision_curve(y_test, y_prob_cat, thresholds_dca)
            nb_tab = decision_curve(y_test, y_prob_tab, thresholds_dca) # Corrected: Use y_prob_tab
            nb_ens = decision_curve(y_test, y_prob_ensemble, thresholds_dca) # Use ensemble probabilities

            prevalence = np.mean(y_test)
            treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds_dca if (1-pt) !=0] # Prevent div by zero
            treat_none = [0 for _ in thresholds_dca if (1-pt) !=0]

            # Filter thresholds to match length of treat_all/none after div by zero handling
            valid_thresholds_dca = [pt for pt in thresholds_dca if (1-pt) !=0]

            fig_dca, ax_dca = plt.subplots(figsize=(6, 4)) # Adjusted figure size
            ax_dca.plot(valid_thresholds_dca, nb_lr, label='Logistic Regression')
            ax_dca.plot(valid_thresholds_dca, nb_xgb, label='XGBoost')
            ax_dca.plot(valid_thresholds_dca, nb_cat, label='CatBoost')
            ax_dca.plot(valid_thresholds_dca, nb_tab, label='TabNet') # Corrected: Added TabNet plot
            ax_dca.plot(valid_thresholds_dca, nb_ens, label='Ensemble (LR+XGB)')

            ax_dca.plot(valid_thresholds_dca, treat_all, linestyle='--', label='Treat All')
            ax_dca.plot(valid_thresholds_dca, treat_none, linestyle='--', label='Treat None')

            ax_dca.set_xlabel('Threshold Probability')
            ax_dca.set_ylabel('Net Benefit')
            ax_dca.set_title('Decision Curve Analysis')
            ax_dca.legend()
            ax_dca.grid()
            st.pyplot(fig_dca)
    else:
        st.write("Data not loaded. Cannot display model evaluation.")


# --- Prediction Tab ---
with prediction_tab:
    if df_raw_viz is not None:
        st.header("Predict Vitamin D Deficiency")
        st.write("Enter the patient's characteristics to predict their vitamin D deficiency status.")

        # Input widgets
        col1_input, col2_input = st.columns(2)
        with col1_input:
            bmi_val = st.slider("BMI", min_value=15.0, max_value=40.0, value=25.0)
            sun_hours_per_day_val = st.slider("Sun Hours per Day", min_value=0.0, max_value=10.0, value=4.0)
            calcium_intake_mg_val = st.slider("Calcium Intake (mg)", min_value=200.0, max_value=1600.0, value=700.0)
            outdoor_activity_minutes_val = st.slider("Outdoor Activity (minutes/day)", min_value=0.0, max_value=180.0, value=60.0)
            diet_score_val = st.slider("Diet Score (0-10)", min_value=0.0, max_value=10.0, value=6.0)
        with col2_input:
            skin_tone_val = st.selectbox("Skin Tone", ['Light', 'Medium', 'Dark'])
            season_val = st.selectbox("Season", ['Spring', 'Summer', 'Monsoon', 'Winter'])
            physical_activity_level_val = st.selectbox("Physical Activity Level", ['Low', 'Moderate', 'High'])
            vitamin_d_supplement_iu_val = st.selectbox("Vitamin D Supplement Dose (IU)", [0, 400, 800, 1000, 2000])


        def predict_deficiency_quiz_streamlit(
            bmi_val, sun_hours_per_day_val, calcium_intake_mg_val, outdoor_activity_minutes_val, diet_score_val,
            skin_tone_val, season_val, physical_activity_level_val, vitamin_d_supplement_iu_val,
            scaler, selected_lr_features, lr_pred_model, OPTIMAL_THRESHOLD, continuous_cols_to_scale, x_train_columns
        ):
            feature_template = {f: 0 for f in x_train_columns} # Use x_train_columns for full set

            feature_template['bmi'] = bmi_val
            feature_template['sun_hours_per_day'] = sun_hours_per_day_val
            feature_template['calcium_intake_mg'] = calcium_intake_mg_val
            feature_template['outdoor_activity_minutes'] = outdoor_activity_minutes_val
            feature_template['diet_score'] = diet_score_val

            if skin_tone_val.lower() == 'light':
                if 'skin_tone_Light' in feature_template: feature_template['skin_tone_Light'] = 1
            elif skin_tone_val.lower() == 'medium':
                if 'skin_tone_Medium' in feature_template: feature_template['skin_tone_Medium'] = 1

            if season_val.lower() == 'spring':
                if 'season_Spring' in feature_template: feature_template['season_Spring'] = 1
            elif season_val.lower() == 'summer':
                if 'season_Summer' in feature_template: feature_template['season_Summer'] = 1
            elif season_val.lower() == 'winter':
                if 'season_Winter' in feature_template: feature_template['season_Winter'] = 1

            if physical_activity_level_val.lower() == 'low':
                if 'physical_activity_level_Low' in feature_template: feature_template['physical_activity_level_Low'] = 1
            elif physical_activity_level_val.lower() == 'moderate':
                if 'physical_activity_level_Moderate' in feature_template: feature_template['physical_activity_level_Moderate'] = 1

            if vitamin_d_supplement_iu_val >= 400 and vitamin_d_supplement_iu_val < 800:
                if 'VitaminD_Supplement_Group_400' in feature_template: feature_template['VitaminD_Supplement_Group_400'] = 1
            elif vitamin_d_supplement_iu_val >= 800 and vitamin_d_supplement_iu_val < 1000:
                if 'VitaminD_Supplement_Group_800' in feature_template: feature_template['VitaminD_Supplement_Group_800'] = 1
            elif vitamin_d_supplement_iu_val >= 1000 and vitamin_d_supplement_iu_val < 2000:
                if 'VitaminD_Supplement_Group_1000' in feature_template: feature_template['VitaminD_Supplement_Group_1000'] = 1
            elif vitamin_d_supplement_iu_val >= 2000:
                if 'VitaminD_Supplement_Group_2000+' in feature_template: feature_template['VitaminD_Supplement_Group_2000+'] = 1

            input_df_full_unscaled = pd.DataFrame([feature_template], columns=x_train_columns)

            # Apply scaling to the continuous features in the input dataframe
            input_df_full_scaled = input_df_full_unscaled.copy()
            input_df_full_scaled[continuous_cols_to_scale] = scaler.transform(input_df_full_scaled[continuous_cols_to_scale])

            # Select only the RFE-selected features for prediction
            input_df_for_prediction = input_df_full_scaled[selected_lr_features]

            # Add constant for statsmodels GLM model
            input_df_sm = sm.add_constant(input_df_for_prediction, has_constant='add', prepend=True)

            # Ensure columns match training order, fill missing with 0 if any
            # This is crucial because RFE might select a subset of dummy variables
            # that are not present in the current input due to user choices.
            missing_cols_in_input = set(lr_pred_model.params.index) - set(input_df_sm.columns)
            for c in missing_cols_in_input:
                input_df_sm[c] = 0
            input_df_sm = input_df_sm[lr_pred_model.params.index] # Reorder to match model


            risk_probability = lr_pred_model.predict(input_df_sm)[0]
            prediction_label = 'Deficient' if risk_probability > OPTIMAL_THRESHOLD else 'Non-Deficient'

            return {
                'risk_probability': round(risk_probability, 4),
                'prediction': prediction_label
            }

        if st.button("Predict Deficiency"):
            prediction_result = predict_deficiency_quiz_streamlit(
                bmi_val, sun_hours_per_day_val, calcium_intake_mg_val, outdoor_activity_minutes_val, diet_score_val,
                skin_tone_val, season_val, physical_activity_level_val, vitamin_d_supplement_iu_val,
                scaler, selected_lr_features, lr_pred_model, OPTIMAL_THRESHOLD, continuous_cols_to_scale, x_train.columns
            )
            st.subheader("Prediction Results:")
            st.write(f"**Predicted Probability of Deficiency:** {prediction_result['risk_probability']:.2f}")
            st.write(f"**Predicted Status:** {prediction_result['prediction']}")

        # --- SHAP Integration for Logistic Regression ---
        st.subheader("Feature Importance Explanation (SHAP - Logistic Regression Model)")
        st.write("This section provides insights into which features influence the prediction for a specific example using the Logistic Regression model.")

        lr_example_idx = st.slider("Select an example from test set for SHAP explanation:", 0, len(x_test) - 1, 0, key='lr_shap_slider')
        if st.button("Generate SHAP Explanation for LR Example", key='lr_shap_button'):
            # Extract coefficients and intercept from the statsmodels result object
            model_coef = lr_pred_model.params.drop('const').values
            model_intercept = lr_pred_model.params['const']

            # Create SHAP explainer for Logistic Regression
            explainer_lr = shap.LinearExplainer((model_coef, model_intercept), x_train[selected_lr_features])

            # Compute SHAP values for the selected example
            shap_values_lr_single = explainer_lr.shap_values(x_test[selected_lr_features].iloc[lr_example_idx])

            st.write(f"**Actual Status for Example {lr_example_idx}:** {y_test.iloc[lr_example_idx]}")

            col_force, col_waterfall = st.columns(2)
            with col_force:
                # Force plot for a single instance
                st.write("**Force Plot:**")
                fig_force_lr, ax_force_lr = plt.subplots(figsize=(6, 3)) # Adjusted figure size
                shap.force_plot(
                    explainer_lr.expected_value,
                    shap_values_lr_single,
                    x_test[selected_lr_features].iloc[lr_example_idx],
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig_force_lr)

            with col_waterfall:
                st.write("**Waterfall Plot:**")
                # Bar plot of feature contributions for the selected instance
                fig_bar_single_lr, ax_bar_single_lr = plt.subplots(figsize=(6, 4)) # Adjusted figure size
                shap.waterfall_plot(
                shap.Explanation(
                        values=shap_values_lr_single,
                        base_values=explainer_lr.expected_value,
                        data=x_test[selected_lr_features].iloc[lr_example_idx],
                        feature_names=x_test[selected_lr_features].columns
                    ),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig_bar_single_lr)


        st.write("--- Recommended for local development/exploration ---")
        if st.checkbox("Show global SHAP summary plots (computationally intensive)", key='lr_global_shap_checkbox'):
            # Compute SHAP values for the whole test set for global plots
            model_coef = lr_pred_model.params.drop('const').values
            model_intercept = lr_pred_model.params['const']
            explainer_lr_global = shap.LinearExplainer((model_coef, model_intercept), x_train[selected_lr_features])
            shap_values_lr_global = explainer_lr_global.shap_values(x_test[selected_lr_features])

            col_summary_bar, col_beeswarm = st.columns(2)
            with col_summary_bar:
                st.subheader("Global Feature Importance (Logistic Regression)")
                fig_summary_bar_lr, ax_summary_bar_lr = plt.subplots(figsize=(6, 4)) # Adjusted figure size
                shap.summary_plot(shap_values_lr_global, x_test[selected_lr_features], plot_type='bar', max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig_summary_bar_lr)

            with col_beeswarm:
                st.subheader("Global Feature Impact (Beeswarm Plot - Logistic Regression)")
                fig_beeswarm_lr, ax_beeswarm_lr = plt.subplots(figsize=(6, 4)) # Adjusted figure size
                shap.summary_plot(shap_values_lr_global, x_test[selected_lr_features], max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig_beeswarm_lr)
    else:
        st.write("Data not loaded. Cannot provide predictions or SHAP explanations.")

# End of Streamlit App Script
