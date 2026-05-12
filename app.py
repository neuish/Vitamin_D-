%%writefile app.py
# --- Streamlit Setup & Data Preparation --- #
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# --- Data Loading and Preprocessing ---
# Assuming the CSV is placed in the same directory as app.py for standalone execution
df = pd.read_csv('Vitamin_D_Dataset.csv')

# Data Cleaning and treating (from notebook cells)
# Drop rows with any null values (already checked in notebook, but included for robustness)
df.dropna(inplace=True)

# Rename columns (from Wv6_aSb6KlqV)
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

# Create tiers for vitaminD_supplement_dose_IU for visualizations (from KnSMTJaR7sjw)
df['supplement_tier'] = pd.cut(df['vitamin_d_supplement_iu'],
                               bins=[-1, 0, 400, 800, 1500, 2001],
                               labels=['None (0 IU)', 'Low (1-400 IU)', 'Medium (401-800 IU)', 'High (801-1500 IU)', 'Very High (>1500 IU)'],
                               right=False)

# Create bins for sun_hours_per_day (from KnSMTJaR7sjw)
df['sun_hours_bins'] = pd.cut(df['sun_hours_per_day'], bins=np.arange(0, 8.5, 0.5), right=False)

# Create sun_exposure_group (from J5-0pCDd8DC3)
df['sun_exposure_group'] = pd.cut(
    df['sun_hours_per_day'],
    bins=[0, 2, 4, 6, 8],
    labels=[
        'Low\n(0-2h)',
        'Moderate\n(2-4h)',
        'High\n(4-6h)',
        'Very High\n(6-8h)'
    ]
)

# Create quartiles for sun_hours_per_day (from Q10JqJUIjSiE)
df['sun_hours_quartile'] = pd.qcut(df['sun_hours_per_day'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

# Drop 'vitamin_d_ng_ml' (from GXUyx7xsVy55)
df = df.drop('vitamin_d_ng_ml', axis=1)

# Convert 'deficient' to object type for some operations, then back to int (from 6ab3e379, then back to int later)
df['deficient'] = df['deficient'].astype('object')

# One-hot encode categorical columns (from ffca819f)
categorical_cols_to_encode = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 'alcohol_use', 'urban_rural']
df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

# Convert boolean columns to integers (from b8682bbd)
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)

# Grouping Age and VitaminD Supplement IU (from e21reFM3aJOL)
age_bins = [0, 20, 30, 40, 50, 60, 70, float('inf')]
age_labels = ['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

delay_bins = [0, 400, 800, 1000, 2000, float('inf')]
delay_labels = ['0', '400', '800', '1000', '2000+']

df['Age_Group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
df['VitaminD_Supplement_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=delay_bins, labels=delay_labels, right=False)

# One-hot encode 'Age_Group' and 'VitaminD_Supplement_Group' (from 2a0bcc8f)
df_age_vitd_encoded = pd.get_dummies(df[['Age_Group', 'VitaminD_Supplement_Group']], drop_first=True)
for col in df_age_vitd_encoded.select_dtypes(include='bool').columns:
    df_age_vitd_encoded[col] = df_age_vitd_encoded[col].astype(int)

df_encoded = pd.concat([df_encoded, df_age_vitd_encoded], axis=1)

# Drop 'age' and 'vitamin_d_supplement_iu' (from xtqGTowfdnhc)
df_encoded = df_encoded.drop(columns=['age', 'vitamin_d_supplement_iu'])

# Convert 'deficient' back to int before train-test split (needed for model training)
df_encoded['deficient'] = df_encoded['deficient'].astype(int)

# --- Train-Test Split & Scaling --- #
columns_to_scale = [
    'bmi', 'sun_hours_per_day', 'screen_time_hours',
    'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes',
    'diet_score', 'sleep_hours', 'cholesterol_mg_dl',
    'body_fat_percentage', 'serum_calcium_mg_dl'
]

x = df_encoded.drop(columns=['deficient', 'supplement_tier', 'sun_hours_bins', 'sun_exposure_group', 'sun_hours_quartile'])
y = df_encoded['deficient']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, stratify=y
)

scaler = StandardScaler()
x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
x_test[columns_to_scale]  = scaler.transform(x_test[columns_to_scale])

# Optimal threshold from Logistic Regression analysis (from iQaRjKzOfqbe)
OPTIMAL_THRESHOLD = 0.4

# --- Model Training ---
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scoring = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']

# Logistic Regression (for comparison)
pipe_lr = Pipeline(
    [('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))]
)
# Fit the LR model for obtaining y_prob_lr (not cross-validated pipeline for direct prediction)
logreg_model_for_predict = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))])
logreg_model_for_predict.fit(x_train, y_train)
y_prob_lr = logreg_model_for_predict.predict_proba(x_test)[:, 1]

# XGBoost (for comparison)
pipe_xgb = Pipeline(
    [('model', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, eval_metric='logloss'))]
)
xgb_model = XGBClassifier(
    colsample_bytree=0.8, random_state=42, eval_metric='logloss',
    learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8 # Best params from notebook cell lwwMej5Q5nLJ
)
xgb_model.fit(x_train, y_train)
y_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]

# CatBoost (the primary model for the app)
cat_model = CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6, random_seed=42, verbose=0, eval_metric='AUC'
)
cat_model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=50, use_best_model=True)
y_prob_cat = cat_model.predict_proba(x_test)[:, 1]

# TabNet (for comparison)
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
y_prob_tab = tabnet_model.predict_proba(x_test.values)[:, 1]

# --- Cross-Validation Results (from _FpNxYVSbBZW) ---
results = {}
for name, pipe in [('Logistic Regression', pipe_lr),
                   ('XGBoost', pipe_xgb),
                   ('CatBoost', cat_model)]:
    # For CatBoost, if already trained, avoid retraining with cross_validate
    # Here we'll use a fresh pipeline for cross_validate if pipe is a trained model
    if isinstance(pipe, CatBoostClassifier):
        cv_pipe = Pipeline([('model', CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_seed=42, verbose=0))])
    else:
        cv_pipe = pipe
    cv_res = cross_validate(cv_pipe, x, y, cv=cv, scoring=scoring)
    results[name] = {metric: f"{cv_res[f'test_{metric}'].mean():.3f} \u00b1 {cv_res[f'test_{metric}'].std():.3f}"
                     for metric in scoring}
cv_df = pd.DataFrame(results).T

# --- Model Comparison Table (from 02g3447UIqaE) ---
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

y_test_pred_lr = (y_prob_lr > OPTIMAL_THRESHOLD).astype(int) # Recompute for consistency with notebook logic
y_pred_xgb = xgb_model.predict(x_test)
y_pred_cat = cat_model.predict(x_test)
y_pred_tab = tabnet_model.predict(x_test.values)

model_comparison = pd.DataFrame([
    {'Model': 'Logistic Regression', **get_metrics(y_test, y_test_pred_lr, y_prob_lr)},
    {'Model': 'XGBoost', **get_metrics(y_test, y_pred_xgb, y_prob_xgb)},
    {'Model': 'CatBoost', **get_metrics(y_test, y_pred_cat, y_prob_cat)},
    {'Model': 'TabNet', **get_metrics(y_test, y_pred_tab, y_prob_tab)}
])

# --- y_test_final for LR Evaluation (from 2pvnwRhEgnMm) ---
y_test_final = pd.DataFrame({'Actual' : y_test.values, 'Predicted_Prob' : y_prob_lr})
y_test_final['Predicted'] = y_test_final.Predicted_Prob.map(lambda x: 1 if x > OPTIMAL_THRESHOLD else 0)

# --- SHAP Explainer (from ee8e0578) ---
explainer_cat = shap.TreeExplainer(cat_model)

# --- Global variables for Streamlit App --- #
x_full = df_encoded.drop(columns=['deficient', 'supplement_tier', 'sun_hours_bins', 'sun_exposure_group', 'sun_hours_quartile']).copy()
prediction_model = cat_model
selected_features_for_prediction = x_train.columns
x_train_full = x_train.copy()
x_test_full = x_test.copy()
explainer_for_shap = explainer_cat

# --- Helper function for Decision Curve Analysis (from AHt9OY4No8sY) ---
def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        if (1 - pt) == 0:
            net_benefit = (TP / N)
        else:
            net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(net_benefit)

    return net_benefits

# --- Streamlit App Definition --- #
def run_streamlit_app():
    st.set_page_config(layout="wide")

    st.title("Vitamin D Deficiency Prediction App (CatBoost Model)")

    tab1, tab2, tab3 = st.tabs(["Visualizations", "Model Evaluation", "Interactive Prediction"])

    with tab1:
        st.header("Key Visualizations")

        # Visualization 1: Vitamin D Deficiency Risk Surface
        st.subheader('Vitamin D Deficiency Risk Surface: Sun Hours vs. Supplementation')
        df_viz1 = df.copy()
        df_viz1['supplement_tier'] = pd.cut(df_viz1['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 2001], labels=['None (0 IU)', 'Low (1-400 IU)', 'Medium (401-800 IU)', 'High (801-1500 IU)', 'Very High (>1500 IU)'], right=False)
        df_viz1['sun_hours_bins'] = pd.cut(df_viz1['sun_hours_per_day'], bins=np.arange(0, 8.5, 0.5), right=False)
        risk_surface = df_viz1.groupby(['sun_hours_bins', 'supplement_tier'])['deficient'].mean().unstack()
        overall_deficiency = df_viz1['deficient'].mean()
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        sns.heatmap(risk_surface * 100, annot=True, fmt='.1f', cmap='RdYlGn_r', center=overall_deficiency * 100, linewidths=.5, linecolor='lightgrey', cbar_kws={'label': 'Deficiency Rate (%)'}, ax=ax1)
        ax1.set_title('Vitamin D Deficiency Risk Surface: Sun Hours vs. Supplementation', fontsize=16)
        ax1.set_xlabel('Sun Hours Per Day Bins', fontsize=12)
        ax1.set_ylabel('Vitamin D Supplement (IU) Tier', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig1)
        plt.close(fig1)

        # Visualization 2: Vitamin D Distribution & Deficiency Threshold
        st.subheader('Vitamin D Distribution by Deficiency Status')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # 'vitamin_d_ng_ml' is dropped from df_encoded, but df still has it for visualization
        sns.violinplot(x='deficient', y='vitamin_d_ng_ml', data=df, palette={0: 'teal', 1: 'coral'}, hue='deficient', legend=False, ax=ax2)
        ax2.axhline(20, color='gold', linestyle='--', label='Deficiency Threshold')
        ax2.set_title('Vitamin D Distribution by Deficiency Status', fontsize=16)
        ax2.set_xlabel('Deficient (0: No, 1: Yes)', fontsize=12)
        ax2.set_ylabel('Vitamin D (ng/mL)', fontsize=12)
        ax2.set_xticks([0, 1], ['Non-Deficient', 'Deficient'])
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)
        plt.close(fig2)

        # Visualization 3: Age Distribution for Deficient vs. Non-Deficient
        st.subheader('Age Distribution for Vitamin D Deficient vs. Non-Deficient')
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='age', hue='deficient', kde=True, palette={0: 'teal', 1: 'coral'}, stat='density', common_norm=False, ax=ax3)
        ax3.set_title('Age Distribution for Vitamin D Deficient vs. Non-Deficient', fontsize=16)
        ax3.set_xlabel('Age', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)
        plt.close(fig3)

        # Visualization 4: Body Fat Percentage vs. Vitamin D Levels
        st.subheader('Body Fat Percentage vs. Vitamin D Levels')
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='body_fat_percentage', y='vitamin_d_ng_ml', hue='deficient', data=df, palette={0: 'teal', 1: 'coral'}, alpha=0.6, ax=ax4)
        ax4.axhline(20, linestyle='--', color='gold', label='Deficiency Threshold')
        ax4.set_title('Body Fat Percentage vs. Vitamin D Levels', fontsize=16)
        ax4.set_xlabel('Body Fat Percentage', fontsize=12)
        ax4.set_ylabel('Vitamin D (ng/mL)', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(title='Deficient')
        st.pyplot(fig4)
        plt.close(fig4)

        # Visualization 5: Sun Exposure vs. Vitamin D Level by Skin Tone
        st.subheader('Vitamin D Distribution Across Sun Exposure Levels and Skin Tone')
        df_viz5 = df.copy()
        df_viz5['sun_exposure_group'] = pd.cut(
            df_viz5['sun_hours_per_day'],
            bins=[0, 2, 4, 6, 8],
            labels=['Low\n(0-2h)', 'Moderate\n(2-4h)', 'High\n(4-6h)', 'Very High\n(6-8h)'])
        fig5, ax5 = plt.subplots(figsize=(14, 8))
        sns.violinplot(data=df_viz5, x='sun_exposure_group', y='vitamin_d_ng_ml', hue='skin_tone', palette='magma', split=False, inner='quartile', linewidth=1.5, ax=ax5)
        ax5.axhline(20, color='red', linestyle='--', linewidth=2, label='Deficiency Threshold')
        ax5.set_title('Vitamin D Distribution Across Sun Exposure Levels and Skin Tone', fontsize=20, weight='bold', pad=20)
        ax5.set_xlabel('Daily Sun Exposure')
        ax5.set_ylabel('Serum Vitamin D (ng/mL)')
        ax5.grid(axis='y', linestyle='--', alpha=0.4)
        ax5.legend(title='Skin Tone', bbox_to_anchor=(1.02, 1), loc='upper left')
        sns.despine(ax=ax5)
        st.pyplot(fig5)
        plt.close(fig5)

        # Visualization 6: Stacked Bar Chart of Deficiency Prevalence by Skin Tone and Season
        st.subheader('Vitamin D Deficiency Prevalence by Skin Tone and Season')
        deficiency_crosstab = df.groupby(['skin_tone', 'season'])['deficient'].mean().unstack()
        fig6, ax6 = plt.subplots(figsize=(12, 7))
        deficiency_crosstab.plot(kind='bar', stacked=True, cmap='viridis', ax=ax6)
        ax6.set_title('Vitamin D Deficiency Prevalence by Skin Tone and Season', fontsize=16)
        ax6.set_xlabel('Skin Tone', fontsize=12)
        ax6.set_ylabel('Proportion Deficient', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        ax6.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig6)
        plt.close(fig6)

        # Visualization 7: Heatmap of Median Vitamin D by Supplementation Tier and Sun Hours Quartile
        st.subheader('Heatmap of Median Vitamin D by Supplementation Tier and Sun Hours Quartile')
        df_viz7 = df.copy()
        df_viz7['supplement_tier'] = pd.cut(df_viz7['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 2001], labels=['None', 'Low (<=400)', 'Medium (401-800)', 'High (801-1500)', 'Very High (>1500)'], right=False)
        df_viz7['sun_hours_quartile'] = pd.qcut(df_viz7['sun_hours_per_day'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        median_vd_heatmap = df_viz7.groupby(['supplement_tier', 'sun_hours_quartile'])['vitamin_d_ng_ml'].median().unstack()
        fig7, ax7 = plt.subplots(figsize=(10, 8))
        sns.heatmap(median_vd_heatmap, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, linecolor='lightgrey', ax=ax7)
        ax7.set_title('Median Vitamin D (ng/mL) by Supplementation Tier and Sun Hours Quartile', fontsize=16)
        ax7.set_xlabel('Sun Hours Per Day Quartile', fontsize=12)
        ax7.set_ylabel('Vitamin D Supplementation Tier', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig7)
        plt.close(fig7)

    with tab2:
        st.header("Model Evaluation")

        st.subheader("Cross-Validation Results")
        st.dataframe(cv_df)

        st.subheader("Model Comparison Table")
        st.dataframe(model_comparison)

        st.subheader("ROC Curve Comparison of All Models")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 7))

        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        ax_roc.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.3f})')

        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        ax_roc.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y_test, y_prob_xgb):.3f})')

        fpr_cat, tpr_cat, _ = roc_curve(y_test, y_prob_cat)
        ax_roc.plot(fpr_cat, tpr_cat, label=f'CatBoost (AUC={roc_auc_score(y_test, y_prob_cat):.3f})')

        fpr_tab, tpr_tab, _ = roc_curve(y_test, y_prob_tab)
        ax_roc.plot(fpr_tab, tpr_tab, label=f'TabNet (AUC={roc_auc_score(y_test, y_prob_tab):.3f})')

        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve Comparison of All Models')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        st.subheader("Decision Curve Analysis")
        fig_dca, ax_dca = plt.subplots(figsize=(8,6))
        thresholds = np.linspace(0.01, 0.99, 100)

        # Baselines
        prevalence = np.mean(y_test)
        treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) if (1-pt) != 0 else prevalence for pt in thresholds]
        treat_none = [0 for _ in thresholds]

        nb_lr = decision_curve(y_test, y_prob_lr, thresholds)
        nb_xgb = decision_curve(y_test, y_prob_xgb, thresholds)
        nb_cat = decision_curve(y_test, y_prob_cat, thresholds)
        nb_tab = decision_curve(y_test, y_prob_tab, thresholds) # Use y_prob_tab from above

        ax_dca.plot(thresholds, nb_lr, label='LR')
        ax_dca.plot(thresholds, nb_xgb, label='XGBoost')
        ax_dca.plot(thresholds, nb_cat, label='CatBoost')
        ax_dca.plot(thresholds, nb_tab, label='TabNet')

        ax_dca.plot(thresholds, treat_all, linestyle='--', label='Treat All')
        ax_dca.plot(thresholds, treat_none, linestyle='--', label='Treat None')

        ax_dca.set_xlabel('Threshold Probability')
        ax_dca.set_ylabel('Net Benefit')
        ax_dca.set_title('Decision Curve Analysis')
        ax_dca.legend()
        ax_dca.grid()
        st.pyplot(fig_dca)
        plt.close(fig_dca)

    with tab3:
        st.header("Interactive Prediction")
        st.write("Enter patient details to predict Vitamin D deficiency.")

        # Collect user inputs dynamically based on features used by the model
        input_data = {}

        st.sidebar.header("Input Features")
        for col_name in columns_to_scale:
            min_val = float(x_full[col_name].min())
            max_val = float(x_full[col_name].max())
            mean_val = float(x_full[col_name].mean())
            input_data[col_name] = st.sidebar.number_input(f"Enter {col_name.replace('_', ' ').title()}", min_value=min_val, max_value=max_val, value=mean_val, key=f"num_{col_name}")

        raw_age = st.sidebar.slider("Age", min_value=18, max_value=79, value=48, key="age_slider")
        raw_vitamin_d_supplement_iu = st.sidebar.slider("Vitamin D Supplement (IU)", min_value=0, max_value=2000, value=400, step=100, key="vitd_supp_slider")

        age_bins_raw = [0, 20, 30, 40, 50, 60, 70, float('inf')]
        age_labels_raw = ['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        age_group_raw = pd.cut([raw_age], bins=age_bins_raw, labels=age_labels_raw, right=False)[0]

        delay_bins_raw = [0, 400, 800, 1000, 2000, float('inf')]
        delay_labels_raw = ['0', '400', '800', '1000', '2000+']
        vitamin_d_group_raw = pd.cut([raw_vitamin_d_supplement_iu], bins=delay_bins_raw, labels=delay_labels_raw, right=False)[0]

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

        input_df = pd.DataFrame(0, index=[0], columns=x_full.columns)

        for col_name in columns_to_scale:
            input_df.loc[0, col_name] = input_data[col_name]

        for col, options in categorical_cols_original.items():
            for option in options:
                col_name_ohe = f"{col}_{option}"
                if col_name_ohe in x_full.columns:
                    input_df.loc[0, col_name_ohe] = 1 if input_data[col] == option else 0
                elif input_data[col] == option and option == options[0]:
                    pass

        for label in age_labels_raw:
            if label != age_labels_raw[0]:
                col_name_ohe = f"Age_Group_{label}"
                if col_name_ohe in x_full.columns:
                    input_df.loc[0, col_name_ohe] = 1 if age_group_raw == label else 0
            elif age_group_raw == label and label == age_labels_raw[0]:
                pass

        for label in delay_labels_raw:
            if label != delay_labels_raw[0]:
                col_name_ohe = f"VitaminD_Supplement_Group_{label}"
                if col_name_ohe in x_full.columns:
                    input_df.loc[0, col_name_ohe] = 1 if vitamin_d_group_raw == label else 0
            elif vitamin_d_group_raw == label and label == delay_labels_raw[0]:
                pass

        input_df = input_df[x_full.columns]

        for col_name in input_df.select_dtypes(include='bool').columns:
            input_df[col_name] = input_df[col_name].astype(int)

        input_df_scaled = input_df.copy()
        input_df_scaled[columns_to_scale] = scaler.transform(input_df_scaled[columns_to_scale])

        input_df_for_prediction = input_df_scaled[selected_features_for_prediction]

        st.subheader("Prediction Result (CatBoost)")
        if st.button("Predict Vitamin D Deficiency"):
            prediction_prob = prediction_model.predict_proba(input_df_for_prediction.values)[0, 1]
            prediction_label = prediction_model.predict(input_df_for_prediction.values)[0]

            if prediction_label == 1:
                st.error(f"The model predicts: **Vitamin D Deficient** (Probability: {prediction_prob:.2f})")
            else:
                st.success(f"The model predicts: **Not Vitamin D Deficient** (Probability: {prediction_prob:.2f})")

            st.subheader("Feature Explanations (SHAP for CatBoost)")
            st.write("How each feature contributes to this specific prediction:")

            shap_values_cat_individual = explainer_for_shap.shap_values(input_df_for_prediction.values)[0]

            st.write("**Individual Prediction Explanation (Force Plot):**")
            fig_force = shap.force_plot(
                explainer_for_shap.expected_value[0],
                shap_values_cat_individual,
                input_df_for_prediction.iloc[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_force, bbox_inches='tight')
            plt.close(fig_force)

            st.write("--- ")
            st.write("**Global Feature Importance (SHAP Summary Plot - Bar):**")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            shap.summary_plot(explainer_for_shap.shap_values(x_test_full.values), x_test_full, plot_type='bar', max_display=15, show=False)
            st.pyplot(fig_bar, bbox_inches='tight')
            plt.close(fig_bar)

            st.write("--- ")
            st.write("**Global Feature Impact (SHAP Summary Plot - Beeswarm):**")
            fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
            shap.summary_plot(explainer_for_shap.shap_values(x_test_full.values), x_test_full, max_display=15, show=False)
            st.pyplot(fig_beeswarm, bbox_inches='tight')
            plt.close(fig_beeswarm)

# Call the Streamlit app function
run_streamlit_app()
