import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

# --- 1. CORRECTED Decision Curve Analysis Function ---
def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []
    for pt in thresholds:
        if pt == 1.0:
            net_benefits.append(0.0)
            continue
        y_pred = (y_prob >= pt).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        # Standard Formula: (TP/N) - (FP/N) * (Pt / (1-Pt))
        nb = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(nb)
    return net_benefits

# --- 2. Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('Vitamin_D_Dataset.csv')
    df.dropna(inplace=True)
    df.columns = [
        'age', 'bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg',
        'vitamin_d_supplement_iu', 'latitude_deg', 'outdoor_activity_minutes',
        'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage',
        'serum_calcium_mg_dl', 'sex', 'skin_tone', 'clothing_coverage', 'season',
        'physical_activity_level', 'diet_type', 'socioeconomic_status',
        'education_level', 'smoking_status', 'alcohol_use', 'urban_rural',
        'vitamin_d_ng_ml', 'deficient'
    ]
    
    cat_cols = ['sex', 'skin_tone', 'clothing_coverage', 'season', 'physical_activity_level', 'diet_type', 'socioeconomic_status', 'education_level', 'smoking_status', 'alcohol_use', 'urban_rural']
    df_ml = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Feature Engineering for Model
    df_ml['Age_Group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], labels=['<20', '20s', '30s', '40s', '50s', '60s', '70+'])
    df_ml['Supp_Group'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 400, 800, 1000, 2000, 20000], labels=['0', '400', '800', '1000', '2000+'])
    df_ml = pd.get_dummies(df_ml, columns=['Age_Group', 'Supp_Group'], drop_first=True)
    
    for col in df_ml.select_dtypes(include='bool').columns:
        df_ml[col] = df_ml[col].astype(int)
    
    # Preprocessing for EDA heatmap
    df['sun_exposure_group'] = pd.cut(df['sun_hours_per_day'], bins=[0, 2, 4, 6, 8], labels=['Low', 'Moderate', 'High', 'V. High'])
    df['supplement_tier'] = pd.cut(df['vitamin_d_supplement_iu'], bins=[-1, 0, 400, 800, 1500, 10000], labels=['None', 'Low', 'Medium', 'High', 'V. High'], right=False)
    
    return df, df_ml

df_raw, df_ml = load_and_preprocess_data()
X = df_ml.drop(columns=['deficient', 'vitamin_d_ng_ml', 'age', 'vitamin_d_supplement_iu', 'sun_exposure_group', 'supplement_tier'], errors='ignore')
y = df_ml['deficient']
NUM_COLS = ['bmi', 'sun_hours_per_day', 'screen_time_hours', 'calcium_intake_mg', 'latitude_deg', 'outdoor_activity_minutes', 'diet_score', 'sleep_hours', 'cholesterol_mg_dl', 'body_fat_percentage', 'serum_calcium_mg_dl']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
scaler = StandardScaler()
x_train_sc = x_train.copy(); x_test_sc = x_test.copy()
x_train_sc[NUM_COLS] = scaler.fit_transform(x_train[NUM_COLS])
x_test_sc[NUM_COLS] = scaler.transform(x_test[NUM_COLS])

@st.cache_resource
def train_models(_xt, _yt, _xv, _yv):
    cat = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42).fit(_xt, _yt)
    xgb = XGBClassifier(random_state=42).fit(_xt, _yt)
    lr = LogisticRegression(max_iter=1000).fit(_xt, _yt)
    tab = TabNetClassifier(verbose=0, seed=42)
    tab.fit(X_train=_xt.values, y_train=_yt.values, eval_set=[(_xv.values, _yv.values)], max_epochs=20)
    return cat, xgb, lr, tab

cat_m, xgb_m, lr_m, tab_m = train_models(x_train_sc, y_train, x_test_sc, y_test)

# --- 3. UI TABS ---
st.set_page_config(page_title="Vitamin D Full Analysis", layout="wide")
st.title("🎓 Vitamin D Deficiency Dashboard")
tab_eda, tab_eval, tab_clinical = st.tabs(["📊 Exploratory Analysis", "🧪 Model Performance", "🔮 Predictive Diagnostic"])

with tab_eda:
    st.header("Epidemiological Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Surface")
        risk = df_raw.groupby(['supplement_tier', 'sun_exposure_group'])['deficient'].mean().unstack().astype(float)
        fig, ax = plt.subplots(); sns.heatmap(risk*100, annot=True, cmap="RdYlGn_r", ax=ax); st.pyplot(fig); plt.close()

with tab_eval:
    st.header("Comparative Model Analytics")
    ev1, ev2 = st.columns(2)
    
    y_prob_cat = cat_m.predict_proba(x_test_sc)[:, 1]
    y_prob_xgb = xgb_m.predict_proba(x_test_sc)[:, 1]
    y_prob_lr  = lr_m.predict_proba(x_test_sc)[:, 1]
    y_prob_tab = tab_m.predict_proba(x_test_sc.values)[:, 1]

    with ev1:
        st.subheader("ROC Curve Comparison")
        fig_roc, ax_roc = plt.subplots()
        for name, p in [("CatBoost", y_prob_cat), ("XGBoost", y_prob_xgb), ("TabNet", y_prob_tab), ("LR", y_prob_lr)]:
            fpr, tpr, _ = roc_curve(y_test, p)
            ax_roc.plot(fpr, tpr, label=f"{name}")
        ax_roc.plot([0,1],[0,1],'k--'); ax_roc.legend(); st.pyplot(fig_roc); plt.close()

    with ev2:
        st.subheader("Decision Curve Analysis")
        thresholds = np.linspace(0.01, 0.99, 100)
        nb_cat = decision_curve(y_test, y_prob_cat, thresholds)
        nb_lr = decision_curve(y_test, y_prob_lr, thresholds)
        
        prevalence = np.mean(y_test)
        treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
        treat_none = [0 for _ in thresholds]
        
        fig_dca, ax_dca = plt.subplots()
        ax_dca.plot(thresholds, nb_cat, label='CatBoost (Model)', color='blue', lw=2)
        ax_dca.plot(thresholds, treat_all, linestyle='--', label='Treat All', color='red', alpha=0.7)
        ax_dca.plot(thresholds, treat_none, color='black', label='Treat None', lw=1)
        ax_dca.set_ylim(-0.05, 0.4); ax_dca.legend(); st.pyplot(fig_dca); plt.close()

with tab_clinical:
    st.header("Diagnostic Tool")
    col1, col2 = st.columns(2)
    with col1:
        i_sun = st.slider("Sun Exposure", 0.0, 8.0, 2.5)
        i_sea = st.selectbox("Season", ["Winter", "Spring", "Summer", "Monsoon"])
        i_bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
        i_age = st.slider("Age", 1, 100, 45)
        i_supp = st.number_input("Supplement (IU)", 0, 5000, 400)
    with col2:
        i_skin = st.selectbox("Skin Tone", ["Light", "Medium", "Dark"])
        i_lat = st.number_input("Latitude", 0.0, 70.0, 34.0)
        i_act = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
        i_diet = st.slider("Diet Score", 1, 10, 5)
        i_sleep = st.slider("Sleep", 4, 12, 7)

    predict_btn = st.button("Run Diagnostic Analysis", type="primary")

    if predict_btn:
        input_row = pd.DataFrame(0.0, index=[0], columns=X.columns)
        input_row.at[0, 'bmi'] = float(i_bmi)
        input_row.at[0, 'sun_hours_per_day'] = float(i_sun)
        input_row.at[0, 'latitude_deg'] = float(i_lat)
        input_row.at[0, 'diet_score'] = float(i_diet)
        input_row.at[0, 'sleep_hours'] = float(i_sleep)
        
        if f'skin_tone_{i_skin}' in X.columns: input_row.at[0, f'skin_tone_{i_skin}'] = 1.0
        if f'season_{i_sea}' in X.columns: input_row.at[0, f'season_{i_sea}'] = 1.0
        
        in_sc = input_row.copy()
        in_sc[NUM_COLS] = scaler.transform(input_row[NUM_COLS])
        prob = cat_m.predict_proba(in_sc)[0,1]
        
        st.divider()
        st.metric("Deficiency Risk", f"{prob*100:.1f}%")
        
        # SHAP Grid
        st.subheader("Interpretability Analysis Grid")
        explainer = shap.TreeExplainer(cat_m)
        shap_vals_local = explainer.shap_values(in_sc)
        g1, g2 = st.columns(2)
        with g1:
            st.write("**Local Impact**")
            fig_l, ax_l = plt.subplots()
            shap.plots.bar(shap.Explanation(values=shap_vals_local[0], data=input_row.iloc[0], feature_names=X.columns), show=False)
            st.pyplot(fig_l); plt.close()
        with g2:
            st.write("**Global Logic**")
            fig_g, ax_g = plt.subplots()
            shap_vals_global = explainer.shap_values(x_test_sc)
            shap.summary_plot(shap_vals_global, x_test_sc, plot_type='bar', show=False)
            st.pyplot(fig_g); plt.close()
