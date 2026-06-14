# ============================================================
# DATASPRINT ROUND 1 — COMPLETE NOTEBOOK
# Digital Media Ad Campaign Revenue Prediction
# ============================================================

# %%\n# ── CELL 1: Install ──────────────────────────────────────────
"""
!pip install -q optuna shap lime lightgbm xgboost missingno ydata-profiling \
             joblib scikit-learn imbalanced-learn category_encoders \
             matplotlib seaborn plotly kaleido
"""

# %%\n# ── CELL 2: Imports ──────────────────────────────────────────
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
import os, gc, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
np.random.seed(42)

from sklearn.model_selection import KFold, train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap
import lime
import lime.lime_tabular
import joblib

print("✅ All imports done.")

# %%\n# ── CELL 3: Load Dataset ─────────────────────────────────────
FNAME = 'digital_media_dataset.csv'

if 'df' not in globals():
    if not os.path.exists(FNAME):
        raise FileNotFoundError(f"Upload {FNAME} to Colab first.")
    df = pd.read_csv(FNAME)
    print(f"Loaded: {df.shape}")

df_clean = df.copy(deep=True)
print("df_clean ready. Shape:", df_clean.shape)
print(df_clean.head(3))
print("\nDtypes:\n", df_clean.dtypes.value_counts())

# %%\n# ── CELL 4: EDA ───────────────────────────────────────────────
print("=== MISSING VALUES ===")
missing = df_clean.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

fig, ax = plt.subplots(figsize=(14, 3))
msno.bar(df_clean, ax=ax, color='#2563EB', fontsize=9)
plt.title('Missing Value Profile', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

print("\n=== DUPLICATES:", df_clean.duplicated().sum(), "===")
print("Shape:", df_clean.shape)

# Revenue distribution
if 'revenue_usd' in df_clean.columns:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(df_clean['revenue_usd'].dropna(), bins=60, color='#2563EB', edgecolor='white')
    axes[0].set_title('Revenue (raw)')
    axes[1].hist(np.log1p(df_clean['revenue_usd'].fillna(0)), bins=60, color='#16A34A', edgecolor='white')
    axes[1].set_title('Revenue (log1p)')
    # Box plot
    df_clean['revenue_usd'].dropna().plot(kind='box', ax=axes[2], color='#DC2626')
    axes[2].set_title('Revenue Boxplot')
    plt.suptitle('Revenue Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()

# Correlation heatmap
num_cols = df_clean.select_dtypes(include=np.number)
if num_cols.shape[1] > 1:
    plt.figure(figsize=(12, 9))
    corr = num_cols.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, linewidths=0.5,
                annot_kws={'size': 8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()

# Channel-level summary
if 'channel' in df_clean.columns and 'revenue_usd' in df_clean.columns:
    channel_summary = df_clean.groupby('channel').agg(
        avg_revenue=('revenue_usd', 'mean'),
        total_revenue=('revenue_usd', 'sum'),
        count=('revenue_usd', 'count')
    ).sort_values('avg_revenue', ascending=False)
    print("\n=== CHANNEL PERFORMANCE ===")
    print(channel_summary)

# %%\n# ── CELL 5: Robust Cleaning ───────────────────────────────────
numeric_candidates = [
    'impressions','clicks','spend_usd','conversions','revenue_usd',
    'ctr_pct','conversion_rate_pct','bounce_rate_pct','session_duration_sec',
    'audience_age','ad_quality_score','roas'
]
for col in numeric_candidates:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop duplicates
dupes = df_clean.duplicated().sum()
if dupes > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"Dropped {dupes} duplicate rows.")

# Remove logically invalid rows
if {'clicks', 'impressions'}.issubset(df_clean.columns):
    mask = df_clean['clicks'] > df_clean['impressions']
    print(f"Removing {mask.sum()} rows where clicks > impressions")
    df_clean = df_clean.loc[~mask]

if 'impressions' in df_clean.columns:
    mask = df_clean['impressions'] < 0
    print(f"Removing {mask.sum()} rows with negative impressions")
    df_clean = df_clean.loc[~mask]

if 'spend_usd' in df_clean.columns:
    mask = df_clean['spend_usd'] < 0
    print(f"Removing {mask.sum()} rows with negative spend")
    df_clean = df_clean.loc[~mask]

# Cap outliers at 99th percentile
cap_cols = ['revenue_usd', 'spend_usd', 'impressions', 'clicks', 'conversions']
for col in cap_cols:
    if col in df_clean.columns:
        q99 = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=q99)
        print(f"Capped {col} at 99th pct = {q99:.2f}")

# Impute with median
impute_cols = ['clicks','spend_usd','conversions','ad_quality_score',
               'bounce_rate_pct','session_duration_sec','audience_age']
for c in impute_cols:
    if c in df_clean.columns:
        nulls = df_clean[c].isnull().sum()
        if nulls > 0:
            med = df_clean[c].median()
            df_clean[c] = df_clean[c].fillna(med)
            print(f"Imputed {nulls} nulls in {c} with median {med:.3f}")

# Date features
if 'date' in df_clean.columns:
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    df_clean['month'] = df_clean['date'].dt.month.fillna(0).astype(int)
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek.fillna(0).astype(int)
    df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)
    df_clean['quarter'] = df_clean['date'].dt.quarter.fillna(0).astype(int)
    df_clean['week_of_year'] = df_clean['date'].dt.isocalendar().week.fillna(0).astype(int)
    print("Date features extracted.")

print("\n✅ After cleaning shape:", df_clean.shape)

# %%\n# ── CELL 6: Feature Engineering ──────────────────────────────
df_clean = df_clean.copy()

def safe_div(a, b, fill=0.0):
    """Safe division avoiding ZeroDivisionError."""
    return np.where(b == 0, fill, a / b)

# --- FUNNEL METRICS ---
imp = df_clean.get('impressions', pd.Series(0, index=df_clean.index))
clk = df_clean.get('clicks', pd.Series(0, index=df_clean.index))
conv = df_clean.get('conversions', pd.Series(0, index=df_clean.index))
spend = df_clean.get('spend_usd', pd.Series(0, index=df_clean.index))
rev = df_clean.get('revenue_usd', pd.Series(0, index=df_clean.index))
aq = df_clean.get('ad_quality_score', pd.Series(5, index=df_clean.index))
bounce = df_clean.get('bounce_rate_pct', pd.Series(50, index=df_clean.index))
sess = df_clean.get('session_duration_sec', pd.Series(60, index=df_clean.index))
cvr = df_clean.get('conversion_rate_pct', pd.Series(0, index=df_clean.index))

# Core ratios
df_clean['ctr_calc'] = safe_div(clk, imp + 1)
df_clean['cvr_calc'] = safe_div(conv, clk + 1)
df_clean['cpc'] = safe_div(spend, clk + 1)
df_clean['cpa'] = safe_div(spend, conv + 1)
df_clean['roas_calc'] = safe_div(rev, spend + 1)
df_clean['cpm'] = safe_div(spend * 1000, imp + 1)

# Efficiency composites
df_clean['revenue_per_click'] = safe_div(rev, clk + 1)
df_clean['revenue_per_impression'] = safe_div(rev, imp + 1)
df_clean['revenue_per_conversion'] = safe_div(rev, conv + 1)
df_clean['conversion_efficiency'] = safe_div(conv, clk + 1)
df_clean['funnel_velocity'] = safe_div(clk * (cvr / 100.0), spend + 1)
df_clean['spend_efficiency'] = safe_div(rev, np.log1p(spend) + 1e-9)

# Fatigue / waste signals
df_clean['overexposure_index'] = safe_div(imp, clk + 1)
df_clean['cost_waste_ratio'] = spend * (1 - cvr / 100.0)
df_clean['campaign_fatigue'] = safe_div(bounce, df_clean['ctr_calc'] + 1e-9)
df_clean['drop_off_rate'] = df_clean['ctr_calc'] - df_clean['cvr_calc']

# Behavioral / intent
df_clean['intent_score'] = safe_div((cvr / 100.0) * sess, bounce + 1)
df_clean['engagement_depth'] = sess * (1 - bounce / 100.0)
df_clean['attention_index'] = safe_div(sess * df_clean['ctr_calc'], bounce + 1)

# Quality composites
aq_norm = safe_div(aq, aq.max() + 1e-9)
ed_norm = safe_div(df_clean['engagement_depth'], df_clean['engagement_depth'].max() + 1e-9)
df_clean['user_quality_score'] = (aq_norm + ed_norm) / 2

intent_norm = safe_div(df_clean['intent_score'], df_clean['intent_score'].max() + 1e-9)
ad_eff_norm = safe_div(df_clean['spend_efficiency'], df_clean['spend_efficiency'].max() + 1e-9)
df_clean['campaign_health_score'] = (intent_norm + ad_eff_norm + df_clean['user_quality_score']) / 3

# Marginal ROAS (quintile-based)
if 'spend_usd' in df_clean.columns:
    df_clean['spend_rank'] = spend.rank(method='first')
    try:
        df_clean['spend_quintile'] = pd.qcut(df_clean['spend_rank'], 5, labels=False, duplicates='drop')
    except Exception:
        df_clean['spend_quintile'] = pd.cut(df_clean['spend_rank'], 5, labels=False)
    _grp = df_clean.groupby('spend_quintile')[['spend_usd', 'revenue_usd']].mean().reset_index()
    _grp['marginal_roas'] = _grp['revenue_usd'].diff() / (_grp['spend_usd'].diff().replace(0, np.nan))
    _map = _grp.set_index('spend_quintile')['marginal_roas'].to_dict()
    df_clean['marginal_roas'] = df_clean['spend_quintile'].map(_map).fillna(0)

# Budget saturation (log concavity signal)
df_clean['log_spend'] = np.log1p(spend)
df_clean['log_impressions'] = np.log1p(imp)
df_clean['log_clicks'] = np.log1p(clk)
df_clean['log_revenue'] = np.log1p(rev)
df_clean['spend_log_ratio'] = safe_div(rev, df_clean['log_spend'] + 1)

# Channel-level reliability prior
if 'channel' in df_clean.columns:
    ch_mean = df_clean.groupby('channel')['revenue_usd'].mean().to_dict()
    ch_std = df_clean.groupby('channel')['revenue_usd'].std().fillna(1).to_dict()
    df_clean['channel_mean_revenue'] = df_clean['channel'].map(ch_mean)
    df_clean['channel_cv'] = df_clean['channel'].map(ch_std) / (df_clean['channel_mean_revenue'] + 1e-9)

# Audience segment signal
if 'audience_segment' in df_clean.columns:
    seg_mean = df_clean.groupby('audience_segment')['revenue_usd'].mean().to_dict()
    df_clean['segment_revenue_prior'] = df_clean['audience_segment'].map(seg_mean)

# Risk flag (low intent + high waste)
median_intent = df_clean['intent_score'].median()
q75_waste = df_clean['cost_waste_ratio'].quantile(0.75)
df_clean['risk_flag'] = ((df_clean['intent_score'] < median_intent) &
                          (df_clean['cost_waste_ratio'] > q75_waste)).astype(int)

# Revenue tier (for classification bonus)
df_clean['revenue_tier'] = pd.qcut(
    df_clean['revenue_usd'].fillna(0), q=4,
    labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
    duplicates='drop'
)

print("✅ Feature engineering done. New shape:", df_clean.shape)
new_feats = ['ctr_calc','cvr_calc','cpc','cpa','roas_calc','cpm',
             'intent_score','campaign_health_score','risk_flag',
             'user_quality_score','marginal_roas','engagement_depth']
print(df_clean[new_feats].describe().round(4).to_string())

# %%\n# ── CELL 7: Encode & Build Feature Matrix ─────────────────────
cat_cols = [c for c in ['channel','region','device_type','audience_segment',
                         'campaign_objective','ad_format'] if c in df_clean.columns]
for c in cat_cols:
    df_clean[c] = df_clean[c].fillna('Unknown').astype(str)

df_model = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False)

EXCLUDE = {
    'campaign_id', 'date', 'revenue_usd', 'log_revenue',
    'revenue_tier', 'spend_rank', 'spend_quintile',
    # leakage cols
    'revenue_per_conversion', 'roas_calc', 'roas',
    'revenue_per_click', 'revenue_per_impression',
    'spend_log_ratio'
}
feature_cols = [c for c in df_model.columns if c not in EXCLUDE]
X = df_model[feature_cols].fillna(0)
y = np.log1p(df_model['revenue_usd'].fillna(0))

print(f"X shape: {X.shape}, y shape: {y.shape}")
print("Sample features:", X.columns[:25].tolist())

# %%\n# ── CELL 8: Baseline Models with 5-fold CV ────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

baseline_models = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             random_state=42, verbosity=0, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                    random_state=42, n_jobs=-1, verbose=-1),
}

results = {}
for name, model in baseline_models.items():
    try:
        r2 = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
        rmse = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        mae = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
        results[name] = {'R2_mean': r2.mean(), 'R2_std': r2.std(),
                         'RMSE_mean': rmse.mean(), 'MAE_mean': mae.mean()}
        print(f"{name}: R2={r2.mean():.4f} ± {r2.std():.4f}")
    except Exception as e:
        results[name] = {'error': str(e)}

print("\n=== BASELINE RESULTS ===")
print(df_clean[new_feats].describe().round(4).to_string())

# %%\n# ── CELL 9: Optuna Tuning (XGBoost + LightGBM) ───────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost tuning
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=2, show_progress_bar=True)
print(f"\n✅ XGBoost Best R2: {study_xgb.best_value:.4f}")
print("Best params:", study_xgb.best_params)

# LightGBM tuning
def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42, 'verbose': -1, 'n_jobs': -1
    }
    model = lgb.LGBMRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    return scores.mean()

study_lgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=2, show_progress_bar=True)
print(f"\n✅ LightGBM Best R2: {study_lgb.best_value:.4f}")

# Optuna plots
try:
    optuna.visualization.matplotlib.plot_optimization_history(study_xgb)
    plt.title('XGBoost Optimization History'); plt.show()
    optuna.visualization.matplotlib.plot_param_importances(study_xgb)
    plt.title('XGBoost Hyperparameter Importance'); plt.show()
except Exception:
    pass  # plotly not always available

# %%\n# ── CELL 10: Train Best Models, Ensemble ─────────────────────
best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0, n_jobs=-1)
best_xgb.fit(X_train, y_train)

best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1, n_jobs=-1)
best_lgb.fit(X_train, y_train)

# Simple ensemble (equal weight)
y_pred_xgb = best_xgb.predict(X_test)
y_pred_lgb = best_lgb.predict(X_test)
y_pred_ensemble = (y_pred_xgb * 0.55 + y_pred_lgb * 0.45)

# Evaluate on real scale
y_test_real = np.expm1(y_test)
y_pred_real_xgb = np.expm1(y_pred_xgb)
y_pred_real_lgb = np.expm1(y_pred_lgb)
y_pred_real_ens = np.expm1(y_pred_ensemble)

def eval_metrics(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    print(f"{name}: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

m_xgb = eval_metrics(y_test_real, y_pred_real_xgb, "XGBoost")
m_lgb = eval_metrics(y_test_real, y_pred_real_lgb, "LightGBM")
m_ens = eval_metrics(y_test_real, y_pred_real_ens, "Ensemble")

# Pick best single model for SHAP (usually XGBoost)
final_model = best_xgb
final_preds = y_pred_real_xgb

# Residual plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test_real, final_preds, alpha=0.4, s=10, color='#2563EB')
max_val = max(y_test_real.max(), final_preds.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Revenue'); axes[0].set_ylabel('Predicted Revenue')
axes[0].set_title('Actual vs Predicted')
residuals = y_test_real.values - final_preds
axes[1].hist(residuals, bins=50, color='#16A34A', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Residual Distribution')
plt.suptitle('Model Performance', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# %%\n# ── CELL 11: SHAP Explainability ──────────────────────────────
explainer = shap.TreeExplainer(final_model)
X_test_sample = X_test.sample(n=min(500, X_test.shape[0]), random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Global bar chart
plt.figure()
shap.summary_plot(shap_values, X_test_sample, plot_type='bar', max_display=15, show=False)
plt.title('SHAP Feature Importance (Global)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# Beeswarm
plt.figure()
shap.summary_plot(shap_values, X_test_sample, max_display=15, show=False)
plt.title('SHAP Value Distribution', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# Top feature dependence plots
top_feats = pd.Series(np.abs(shap_values).mean(0),
                      index=X_test_sample.columns).sort_values(ascending=False)
for feat in top_feats.index[:3]:
    shap.dependence_plot(feat, shap_values, X_test_sample, interaction_index=None, show=True)

# Local explanation (single prediction)
print("\n=== LOCAL EXPLANATION (first test sample) ===")
local_idx = 0
shap_exp = explainer(X_test_sample.iloc[[local_idx]])
shap.plots.waterfall(shap_exp[0], max_display=10, show=True)

# Business insight narrative
print("\n=== TOP 5 REVENUE DRIVERS (SHAP) ===")
for feat, val in top_feats.head(5).items():
    direction = "↑ increases" if shap_values[:, X_test_sample.columns.get_loc(feat)].mean() > 0 else "↓ decreases"
    print(f"  {feat}: avg |SHAP| = {val:.4f} → {direction} predicted revenue")

# %%\n# ── CELL 12: LIME Local Explanation ───────────────────────────
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode='regression',
    random_state=42
)

idx = X_test.index[0]
exp = lime_explainer.explain_instance(
    X_test.loc[idx].values,
    final_model.predict,
    num_features=12
)
print("LIME local explanation (top 12):")
for feat, weight in sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True):
    sign = "+" if weight > 0 else ""
    print(f"  {feat}: {sign}{weight:.4f}")
exp.as_pyplot_figure()
plt.title('LIME Local Explanation', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# %%\n# ── CELL 13: Clustering & Campaign Segments ───────────────────
cluster_feats = [c for c in [
    'spend_usd','revenue_usd','conversions','ad_quality_score',
    'intent_score','campaign_health_score','ctr_calc','cvr_calc'
] if c in df_clean.columns]

if cluster_feats:
    cluster_data = df_clean[cluster_feats].fillna(0)
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    # Elbow method
    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cluster_scaled)
        inertias.append(km.inertia_)
    plt.figure(figsize=(7, 4))
    plt.plot(range(2, 8), inertias, 'bo-', linewidth=2)
    plt.xlabel('K'); plt.ylabel('Inertia')
    plt.title('KMeans Elbow Method'); plt.show()

    # Final clustering with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_clean['campaign_cluster'] = kmeans.fit_predict(cluster_scaled)
    joblib.dump({'kmeans': kmeans, 'scaler': scaler, 'features': cluster_feats},
                'cluster_model.joblib')

    cluster_profiles = df_clean.groupby('campaign_cluster')[cluster_feats].mean().round(3)
    print("\n=== CAMPAIGN CLUSTER PROFILES ===")
    print(cluster_profiles.to_string())

    # Cluster labels (business-friendly)
    rev_by_cluster = cluster_profiles['revenue_usd']
    labels = {
        rev_by_cluster.idxmax(): '🏆 High Performers',
        rev_by_cluster.idxmin(): '⚠️ Underperformers',
    }
    df_clean['cluster_label'] = df_clean['campaign_cluster'].map(
        lambda x: labels.get(x, f'Segment {x}')
    )

"""# Install Prophet if not already
# !pip install prophet

from prophet import Prophet

# Monthly aggregation
ts_month = df_clean.groupby(pd.Grouper(key='date', freq='M'))['revenue_usd'].sum().reset_index()
ts_month = ts_month.rename(columns={'date':'ds','revenue_usd':'y'})

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(ts_month)

future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

# Plot with Matplotlib
plt.figure(figsize=(10,5))
plt.plot(ts_month['ds'], ts_month['y'], label='Historical Revenue', color='black')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Revenue', color='blue')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='lightblue', alpha=0.3, label='Confidence Interval')
plt.title("Monthly Revenue Forecast", fontsize=14)
plt.xlabel("Date"); plt.ylabel("Revenue")
plt.legend()
plt.show()
"""

# %%\n# ── CELL 14: Save All Artifacts ───────────────────────────────
"""joblib.dump(final_model, 'best_xgb.joblib')
joblib.dump(best_lgb, 'best_lgb.joblib')
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')
joblib.dump(explainer, 'shap_explainer.joblib')
# LIME explainer cannot be pickled easily due to lambda functions.
# We save its configuration/data instead so it can be re-instantiated.
joblib.dump({
    'training_data': X_train.values,
    'feature_names': X_train.columns.tolist(),
    'mode': 'regression',
    'random_state': 42
}, 'lime_config.joblib')
joblib.dump({
    'xgb': m_xgb, 'lgb': m_lgb, 'ensemble': m_ens
}, 'model_metrics.joblib')

df_clean.to_csv('df_clean_full_with_features.csv', index=False)
joblib.dump(df_clean, 'df_clean_full.joblib')

# Save SHAP values for UI
joblib.dump({
    'columns': X_test_sample.columns.tolist(),
    'shap_values': shap_values,
    'top_features': top_feats.head(20).to_dict()
}, 'shap_data.joblib')

# Save model metadata JSON (for UI)
meta = {
    'model': 'XGBoost + LightGBM Ensemble',
    'n_features': len(X.columns),
    'n_samples': len(df_clean),
    'metrics': {'xgb': m_xgb, 'lgb': m_lgb, 'ensemble': m_ens},
    'top_features': top_feats.head(10).index.tolist(),
    'tuning': 'Optuna TPE (80 XGB + 60 LGB trials)',
    'xai': ['SHAP TreeExplainer', 'LIME TabularExplainer'],
    'clusters': 4
}
with open('model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✅ ALL ARTIFACTS SAVED:")
print("  best_xgb.joblib | best_lgb.joblib | feature_columns.joblib")
print("  shap_explainer.joblib | lime_config.joblib")
print("  df_clean_full_with_features.csv | model_meta.json")
print("  cluster_model.joblib | shap_data.joblib")
"""
# ── CELL 14: Save All Artifacts ───────────────────────────────
import os
os.makedirs('artifacts', exist_ok=True)

joblib.dump(final_model, 'artifacts/best_xgb.joblib')
joblib.dump(best_lgb, 'artifacts/best_lgb.joblib')
joblib.dump(X.columns.tolist(), 'artifacts/feature_columns.joblib')
joblib.dump(explainer, 'artifacts/shap_explainer.joblib')
joblib.dump({
    'training_data': X_train.values,
    'feature_names': X_train.columns.tolist(),
    'mode': 'regression',
    'random_state': 42
}, 'artifacts/lime_config.joblib')
joblib.dump({
    'xgb': m_xgb, 'lgb': m_lgb, 'ensemble': m_ens
}, 'artifacts/model_metrics.joblib')

df_clean.to_csv('artifacts/df_clean_full_with_features.csv', index=False)
joblib.dump(df_clean, 'artifacts/df_clean_full.joblib')

joblib.dump({
    'columns': X_test_sample.columns.tolist(),
    'shap_values': shap_values,
    'top_features': top_feats.head(20).to_dict()
}, 'artifacts/shap_data.joblib')

meta = {
    'model': 'XGBoost + LightGBM Ensemble',
    'n_features': len(X.columns),
    'n_samples': len(df_clean),
    'metrics': {'xgb': m_xgb, 'lgb': m_lgb, 'ensemble': m_ens},
    'top_features': top_feats.head(10).index.tolist(),
    'tuning': 'Optuna TPE (80 XGB + 60 LGB trials)',
    'xai': ['SHAP TreeExplainer', 'LIME TabularExplainer'],
    'clusters': 4
}
with open('artifacts/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✅ ALL ARTIFACTS SAVED to artifacts/ folder:")
print("  best_xgb.joblib | best_lgb.joblib | feature_columns.joblib")
print("  shap_explainer.joblib | lime_config.joblib | model_metrics.joblib")
print("  df_clean_full_with_features.csv | model_meta.json | shap_data.joblib")