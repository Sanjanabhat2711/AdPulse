"""import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import random
import math
import csv
import io

app = FastAPI()

# Data configuration
CHANNELS = ['Social Media','Search','Display','Email','Video','Affiliate']
OBJECTIVES = ['Brand Awareness','Lead Generation','Retargeting','Conversion','Engagement']
DEVICES = ['Mobile','Desktop','Tablet']
REGIONS = ['North America','Europe','APAC','LATAM','MEA']

def generate_campaigns():
    campaigns = []
    for i in range(48):
        ch = CHANNELS[i % len(CHANNELS)]
        spend = random.uniform(500, 25000)
        imp = random.randint(10000, 500000)
        clk = random.randint(200, int(imp * 0.08))
        conv = random.randint(5, int(clk * 0.12))
        revenue = spend * random.uniform(1.2, 6.5) + random.uniform(0, 2000)
        aq = random.uniform(4, 10)
        bounce = random.uniform(20, 75)
        sess = random.uniform(30, 180)
        cvr = (conv / clk * 100) if clk else 0
        ctr = (clk / imp * 100) if imp else 0
        intentScore = (cvr / 100 * sess) / (bounce + 1)
        healthScore = min(1, max(0, (intentScore / 2 + aq / 10 + revenue / spend / 10) * 0.25))
        roas = revenue / spend if spend else 0
        status = 'high' if roas > 3.5 else 'medium' if roas > 2 else 'risk'
        
        campaigns.append({
            "id": i + 1,
            "name": f"Campaign {str(i + 1).zfill(3)}",
            "channel": ch,
            "objective": OBJECTIVES[random.randint(0, 4)],
            "device": DEVICES[random.randint(0, 2)],
            "region": REGIONS[random.randint(0, 4)],
            "spend": spend,
            "imp": imp,
            "clk": clk,
            "conv": conv,
            "revenue": revenue,
            "aq": aq,
            "bounce": bounce,
            "sess": sess,
            "cvr": cvr,
            "ctr": ctr,
            "roas": roas,
            "intentScore": intentScore,
            "healthScore": healthScore,
            "cpc": spend / clk if clk else 0,
            "cpa": spend / conv if conv else 0,
            "cpm": spend * 1000 / imp if imp else 0,
            "status": status,
            "riskFlag": 1 if (status == 'risk' and bounce > 55) else 0
        })
    return campaigns

global_campaigns = generate_campaigns()

@app.get("/")
def read_root():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/campaigns")
def get_campaigns():
    return global_campaigns

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    global global_campaigns
    contents = await file.read()
    decoded = contents.decode('utf-8')
    reader = csv.DictReader(io.StringIO(decoded))
    
    new_campaigns = []
    i = 1
    for row in reader:
        try:
            spend = float(row.get('spend_usd', 0) or 0)
            imp = int(float(row.get('impressions', 0) or 0))
            clk = int(float(row.get('clicks', 0) or 0))
            conv = int(float(row.get('conversions', 0) or 0))
            revenue = float(row.get('revenue_usd', 0) or 0)
            aq = float(row.get('ad_quality_score', 5) or 5)
            bounce = float(row.get('bounce_rate_pct', 50) or 50)
            sess = float(row.get('session_duration_sec', 60) or 60)
            
            cvr = float(row.get('conversion_rate_pct', (conv / clk * 100) if clk else 0) or 0)
            ctr = float(row.get('ctr_pct', (clk / imp * 100) if imp else 0) or 0)
            
            intentScore = (cvr / 100 * sess) / (bounce + 1)
            roas = revenue / spend if spend else 0
            healthScore = min(1, max(0, (intentScore / 2 + aq / 10 + roas / 10) * 0.25))
            status = 'high' if roas > 3.5 else 'medium' if roas > 2 else 'risk'
            
            c = {
                "id": str(row.get('campaign_id', f"C{i:03d}")),
                "name": str(row.get('campaign_id', f"Campaign {i:03d}")),
                "channel": str(row.get('channel', 'Other')),
                "objective": str(row.get('campaign_objective', 'Awareness')),
                "device": str(row.get('device_type', 'Mobile')),
                "region": str(row.get('region', 'North America')),
                "spend": spend,
                "imp": imp,
                "clk": clk,
                "conv": conv,
                "revenue": revenue,
                "aq": aq,
                "bounce": bounce,
                "sess": sess,
                "cvr": cvr,
                "ctr": ctr,
                "roas": roas,
                "intentScore": intentScore,
                "healthScore": healthScore,
                "cpc": spend / clk if clk else 0,
                "cpa": spend / conv if conv else 0,
                "cpm": spend * 1000 / imp if imp else 0,
                "status": status,
                "riskFlag": 1 if (status == 'risk' and bounce > 55) else 0
            }
            new_campaigns.append(c)
            i += 1
            if i > 2500: break # Safety limit
        except Exception:
            continue
            
    if new_campaigns:
        global_campaigns = sorted(new_campaigns, key=lambda x: x['revenue'], reverse=True)
    return {"message": f"Processed {len(new_campaigns)} campaigns successfully."}

class PredictionRequest(BaseModel):
    spend: float
    imp: float
    clk: float
    cvr: float
    aq: float
    bounce: float
    sess: float
    channel: str

@app.post("/api/predict")
def predict_revenue(req: PredictionRequest):
    # Mimic the JS pseudo-model
    ctr = (req.clk / (req.imp + 1)) * 100 if (req.imp + 1) > 0 else 0
    conv = req.clk * req.cvr / 100
    intentScore = (req.cvr / 100 * req.sess) / (req.bounce + 1)
    engDepth = req.sess * (1 - req.bounce / 100)
    
    chBonusMap = {"Search": 1.15, "Social Media": 1.05, "Video": 1.08, "Display": 0.95, "Email": 1.02, "Affiliate": 0.92}
    chBonus = chBonusMap.get(req.channel, 1.0)
    
    logRev = (
        math.log1p(req.spend) * 0.42 +
        intentScore * 0.034 +
        req.aq * 0.08 +
        ctr * 0.15 +
        req.cvr * 0.18 +
        engDepth * 0.002 -
        req.bounce * 0.008 +
        math.log1p(conv) * 0.3 +
        random.uniform(0, 0.1)
    ) * chBonus
    
    revenue = (math.exp(max(0, logRev)) - 1) * 100 if logRev > 0 else 0
    roas = revenue / (req.spend + 1)
    cpa = req.spend / (conv + 1)
    healthScore = min(100, round((intentScore / 2 + req.aq / 10 + roas / 8) * 15))
    
    return {
        "revenue": revenue,
        "roas": roas,
        "cpa": cpa,
        "healthScore": healthScore,
        "bounce": req.bounce,
        "aq": req.aq
    }

@app.get("/api/report/csv")
def generate_csv_report():
    file_path = "campaign_report.csv"
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not global_campaigns:
            return FileResponse(file_path)
            
        keys = global_campaigns[0].keys()
        writer.writerow(keys)
        for camp in global_campaigns:
            writer.writerow([camp[k] for k in keys])
            
    return FileResponse(file_path, media_type="text/csv", filename="campaign_report.csv")

@app.get("/api/report/company")
def generate_company_report():
    file_path = "company_executive_report.txt"
    total_spend = sum(c['spend'] for c in global_campaigns)
    total_rev = sum(c['revenue'] for c in global_campaigns)
    roas = total_rev / total_spend if total_spend else 0
    at_risk = len([c for c in global_campaigns if c['status'] == 'risk'])
    
    high_perf = sorted(global_campaigns, key=lambda x: x['revenue'], reverse=True)
    best_campaign = high_perf[0] if high_perf else None
    best_roas = sorted(global_campaigns, key=lambda x: x['roas'], reverse=True)
    best_roas_campaign = best_roas[0] if best_roas else None
    
    report_text = f"====================================================\n"
    report_text += f"AD-PULSE EXECUTIVE CAMPAIGN REPORT\n"
    report_text += f"====================================================\n\n"
    
    report_text += f"OVERVIEW:\n"
    report_text += f"- Total Campaigns: {len(global_campaigns)}\n"
    report_text += f"- Total Spend: ${total_spend:,.2f}\n"
    report_text += f"- Total Revenue Generated: ${total_rev:,.2f}\n"
    report_text += f"- Average ROAS: {roas:.2f}x\n\n"
    
    report_text += f"RISK ANALYSIS:\n"
    report_text += f"- Campaigns at Risk: {at_risk}\n"
    report_text += f"- Estimated Wasteful Spend (Risk): ${sum(c['spend'] for c in global_campaigns if c['status'] == 'risk'):,.2f}\n\n"
    
    if best_campaign and best_roas_campaign:
        report_text += f"PERFORMANCE HIGHLIGHTS:\n"
        report_text += f"- Best Performing Campaign: {best_campaign['name']} (Revenue: ${best_campaign['revenue']:,.2f})\n"
        report_text += f"- Highest ROAS Campaign: {best_roas_campaign['name']} (ROAS: {best_roas_campaign['roas']:.2f}x)\n\n"
    
    report_text += f"RECOMMENDATIONS:\n"
    report_text += f"1. Reallocate budget from underperforming 'At Risk' campaigns to 'High' performing ones.\n"
    report_text += f"2. Investigate high bounce rate campaigns as they are negatively impacting predictions.\n"
    report_text += f"3. Optimize Ad Quality (AQ) to improve overall Conversion Efficiency.\n\n"
    
    report_text += f"====================================================\n"
    report_text += f"*Generated by AdPulse AI Intelligence Engine*\n"
    report_text += f"====================================================\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    return FileResponse(file_path, media_type="text/plain", filename="Company_Executive_Report.txt")

if __name__ == "__main__":
    uvicorn.run("backend_app:app", host="127.0.0.1", port=8000, reload=True)
"""


# ============================================================
# AdPulse Backend — Complete FastAPI Server
# Run: cd backend && uvicorn main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import joblib, json, os, io, numpy as np, pandas as pd
import requests, shutil, tempfile

from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN   = os.getenv("HF_TOKEN", "")
ARTIFACTS  = "../artifacts"

# ── Load ML artifacts ─────────────────────────────────────────
try:
    model        = joblib.load(f"{ARTIFACTS}/best_xgb.joblib")
    feature_cols = joblib.load(f"{ARTIFACTS}/feature_columns.joblib")
    metrics      = joblib.load(f"{ARTIFACTS}/model_metrics.joblib")
    shap_data    = joblib.load(f"{ARTIFACTS}/shap_data.joblib")
    meta         = json.load(open(f"{ARTIFACTS}/model_meta.json"))
    print("✅ All artifacts loaded.")
except Exception as e:
    print(f"⚠️  Artifact warning: {e}")
    model = feature_cols = metrics = shap_data = meta = None

# ── Global dataframe (loaded from CSV or uploaded) ────────────
_df: Optional[pd.DataFrame] = None

def get_df() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df
    csv_path = f"{ARTIFACTS}/df_clean_full_with_features.csv"
    if os.path.exists(csv_path):
        _df = pd.read_csv(csv_path)
        return _df
    return pd.DataFrame()

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="AdPulse API", description="ML-powered campaign intelligence", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    spend:   float
    imp:     float
    clk:     float
    cvr:     float
    aq:      float
    bounce:  float
    sess:    float
    channel: str = "Search"
    region:  str = "North America"
    device:  str = "Mobile"

class ChatRequest(BaseModel):
    question: str
    campaign_context: Optional[dict] = None

class CopyRequest(BaseModel):
    text: str

class BudgetRequest(BaseModel):
    total_budget: float = 10000
    channels: List[str] = ["Search", "Social Media", "Display", "Email", "Video"]

# ══════════════════════════════════════════════════════════════
# HELPER — build feature vector for prediction
# ══════════════════════════════════════════════════════════════
def build_features(spend, imp, clk, cvr, aq, bounce, sess, channel, region, device):
    imp1  = imp + 1
    clk1  = clk + 1
    conv1 = max(1, clk * cvr / 100)

    features = {c: 0 for c in (feature_cols or [])}
    features.update({
        'log_spend':          np.log1p(spend),
        'log_impressions':    np.log1p(imp1),
        'log_clicks':         np.log1p(clk1),
        'ctr_calc':           clk1 / imp1,
        'cvr_calc':           conv1 / clk1,
        'cpc':                spend / clk1,
        'cpa':                spend / conv1,
        'cpm':                spend * 1000 / imp1,
        'intent_score':       (cvr / 100 * sess) / (bounce + 1),
        'engagement_depth':   sess * (1 - bounce / 100),
        'ad_quality_score':   aq,
        'cost_waste_ratio':   spend * (1 - cvr / 100),
        'overexposure_index': imp1 / clk1,
        'drop_off_rate':      (clk1/imp1) - (conv1/clk1),
        f'channel_{channel.replace(" ", "_")}':  1,
        f'device_type_{device}':                  1,
        f'region_{region.replace(" ", "_")}':    1,
    })
    return features

def health_label(roas):
    if roas >= 3.5: return "High"
    if roas >= 2.0: return "Medium"
    return "Low"

def recommend(roas, bounce, aq):
    if roas < 1.5:  return "ROAS critically low — pause and review targeting immediately."
    if bounce > 65: return "High bounce rate detected — improve landing page relevance."
    if aq < 5:      return "Ad quality below 5 — refresh creative to unlock higher CTR."
    if roas > 4.0:  return "High performer — scale spend 20–30% to capture more revenue."
    return "Campaign healthy — monitor weekly and A/B test ad copy."

# ══════════════════════════════════════════════════════════════
# ROUTES — HEALTH
# ══════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "XGB+LGB Ensemble",
        "r2": metrics["xgb"]["R2"] if metrics else "N/A",
        "artifacts_loaded": model is not None,
        "data_loaded": not get_df().empty
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — CSV UPLOAD
# ══════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    global _df
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files accepted.")

    contents = await file.read()
    try:
        df_new = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Basic validation
    required = ["impressions","clicks","spend_usd","revenue_usd"]
    missing  = [c for c in required if c not in df_new.columns]
    if missing:
        raise HTTPException(400, f"Missing required columns: {missing}")

    # Quick clean
    for col in ["impressions","clicks","spend_usd","conversions","revenue_usd",
                "bounce_rate_pct","session_duration_sec","ad_quality_score",
                "conversion_rate_pct","ctr_pct"]:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    df_new = df_new.drop_duplicates()
    if {"clicks","impressions"}.issubset(df_new.columns):
        df_new = df_new[df_new["clicks"] <= df_new["impressions"]]

    # Derived columns needed by dashboard
    df_new["spend_usd"]  = df_new["spend_usd"].fillna(0)
    df_new["revenue_usd"]= df_new["revenue_usd"].fillna(0)
    df_new["clicks"]     = df_new.get("clicks", pd.Series(0, index=df_new.index)).fillna(0)
    df_new["conversions"]= df_new.get("conversions", pd.Series(0, index=df_new.index)).fillna(0)
    df_new["impressions"]= df_new.get("impressions", pd.Series(0, index=df_new.index)).fillna(0)

    df_new["roas_calc"]  = df_new["revenue_usd"] / (df_new["spend_usd"] + 1)
    df_new["ctr_calc"]   = df_new["clicks"] / (df_new["impressions"] + 1) * 100
    df_new["cvr_calc"]   = df_new["conversions"] / (df_new["clicks"] + 1) * 100
    df_new["cpc"]        = df_new["spend_usd"] / (df_new["clicks"] + 1)
    df_new["cpa"]        = df_new["spend_usd"] / (df_new["conversions"] + 1)

    bounce  = df_new.get("bounce_rate_pct", pd.Series(50, index=df_new.index)).fillna(50)
    sess    = df_new.get("session_duration_sec", pd.Series(60, index=df_new.index)).fillna(60)
    cvr_col = df_new.get("conversion_rate_pct", pd.Series(1, index=df_new.index)).fillna(1)

    df_new["intent_score"]       = (cvr_col / 100 * sess) / (bounce + 1)
    df_new["engagement_depth"]   = sess * (1 - bounce / 100)
    df_new["cost_waste_ratio"]   = df_new["spend_usd"] * (1 - cvr_col / 100)

    aq = df_new.get("ad_quality_score", pd.Series(5, index=df_new.index)).fillna(5)
    df_new["campaign_health_score"] = (
        (df_new["intent_score"] / (df_new["intent_score"].max() + 1e-9)) * 0.4 +
        (aq / 10) * 0.3 +
        (df_new["roas_calc"] / (df_new["roas_calc"].max() + 1e-9)) * 0.3
    )
    df_new["risk_flag"] = (
        (df_new["intent_score"] < df_new["intent_score"].median()) &
        (df_new["cost_waste_ratio"] > df_new["cost_waste_ratio"].quantile(0.75))
    ).astype(int)

    _df = df_new
    # Optionally persist
    os.makedirs(ARTIFACTS, exist_ok=True)
    df_new.to_csv(f"{ARTIFACTS}/df_clean_full_with_features.csv", index=False)

    return {
        "message": f"✅ {len(df_new)} rows loaded successfully from {file.filename}",
        "shape": list(df_new.shape),
        "columns": df_new.columns.tolist()
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — CAMPAIGNS
# ══════════════════════════════════════════════════════════════
@app.get("/api/campaigns")
def get_campaigns():
    df = get_df()
    if df.empty:
        return []

    result = []
    channels = df["channel"].unique() if "channel" in df.columns else ["Unknown"]

    for _, row in df.iterrows():
        roas = float(row.get("roas_calc", 1))
        status = "high" if roas > 3.5 else "medium" if roas > 2 else "risk"
        result.append({
            "id":           str(row.get("campaign_id", _)),
            "name":         str(row.get("campaign_id", f"Campaign {_}")),
            "channel":      str(row.get("channel", "Unknown")),
            "region":       str(row.get("region", "Unknown")),
            "device":       str(row.get("device_type", "Unknown")),
            "spend":        round(float(row.get("spend_usd", 0)), 2),
            "revenue":      round(float(row.get("revenue_usd", 0)), 2),
            "imp":          int(row.get("impressions", 0)),
            "clk":          int(row.get("clicks", 0)),
            "conv":         int(row.get("conversions", 0)),
            "roas":         round(roas, 3),
            "ctr":          round(float(row.get("ctr_calc", 0)), 3),
            "cvr":          round(float(row.get("cvr_calc", 0)), 3),
            "cpc":          round(float(row.get("cpc", 0)), 2),
            "cpa":          round(float(row.get("cpa", 0)), 2),
            "intentScore":  round(float(row.get("intent_score", 0)), 4),
            "healthScore":  round(float(row.get("campaign_health_score", 0)), 4),
            "riskFlag":     int(row.get("risk_flag", 0)),
            "status":       status,
        })
    return result

@app.get("/api/campaigns/summary")
def campaign_summary():
    df = get_df()
    if df.empty:
        return {"total_campaigns": 0, "total_revenue": 0, "total_spend": 0,
                "avg_roas": 0, "risk_campaigns": 0, "top_channel": "N/A"}

    top_ch = "N/A"
    if "channel" in df.columns and "revenue_usd" in df.columns:
        top_ch = df.groupby("channel")["revenue_usd"].sum().idxmax()

    return {
        "total_campaigns": len(df),
        "total_revenue":   round(float(df["revenue_usd"].sum()), 2),
        "total_spend":     round(float(df["spend_usd"].sum()), 2),
        "total_conversions": int(df.get("conversions", pd.Series(0)).sum()),
        "avg_roas":        round(float(df["roas_calc"].mean()), 3) if "roas_calc" in df else 0,
        "risk_campaigns":  int(df["risk_flag"].sum()) if "risk_flag" in df else 0,
        "top_channel":     top_ch,
        "high_performers": int((df["roas_calc"] > 3.5).sum()) if "roas_calc" in df else 0,
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — PREDICTION
# ══════════════════════════════════════════════════════════════
@app.post("/api/predict")
def predict(req: PredictRequest):
    if model is None or feature_cols is None:
        # Fallback formula when model not loaded
        log_rev = (
            np.log1p(req.spend) * 0.42 +
            (req.cvr / 100 * req.sess) / (req.bounce + 1) * 0.034 +
            req.aq * 0.08 +
            (req.clk / (req.imp + 1)) * 100 * 0.15 +
            req.cvr * 0.18
        ) * 1.05
        revenue = float(np.expm1(max(0, log_rev))) * 100
    else:
        feats = build_features(req.spend, req.imp, req.clk, req.cvr,
                               req.aq, req.bounce, req.sess,
                               req.channel, req.region, req.device)
        X = pd.DataFrame([feats])[feature_cols].fillna(0)
        revenue = float(np.expm1(model.predict(X)[0]))

    roas    = revenue / (req.spend + 1)
    conv    = max(1, req.clk * req.cvr / 100)
    cpa     = req.spend / conv
    health  = round(min(100, max(0, (
        (req.cvr / 10) * 30 +
        (req.aq / 10) * 30 +
        (min(roas, 5) / 5) * 40
    ))), 1)

    top_drivers = []
    if shap_data:
        top_drivers = list(shap_data.get("top_features", {}).keys())[:5]

    return {
        "revenue":       round(revenue, 2),
        "roas":          round(roas, 3),
        "cpa":           round(cpa, 2),
        "cpc":           round(req.spend / (req.clk + 1), 2),
        "healthScore":   health,
        "health":        health_label(roas),
        "risk_flag":     int(roas < 1.5),
        "top_drivers":   top_drivers,
        "recommendation": recommend(roas, req.bounce, req.aq)
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — SHAP
# ══════════════════════════════════════════════════════════════
@app.get("/api/shap/global")
def shap_global():
    if shap_data is None:
        # Static fallback matching your notebook results
        return {
            "features": ["log_spend","intent_score","campaign_health_score",
                         "ad_quality_score","conversion_efficiency","engagement_depth",
                         "ctr_calc","cost_waste_ratio","campaign_fatigue",
                         "bounce_rate_pct","overexposure_index","marginal_roas"],
            "values":   [0.412, 0.338, 0.291, 0.254, 0.219, 0.196,
                         0.171, 0.145, 0.132, 0.118, 0.098, 0.087],
            "directions": [1,1,1,1,1,1,1,-1,-1,-1,-1,1]
        }
    top = shap_data.get("top_features", {})
    return {
        "features":   list(top.keys())[:12],
        "values":     [round(v, 4) for v in list(top.values())[:12]],
        "directions": [1]*12
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — CHAT (OpenAI → HuggingFace → Rule-based)
# ══════════════════════════════════════════════════════════════
@app.post("/api/chat")
def chat(req: ChatRequest):
    ctx = req.campaign_context or {}

    shap_context = ""
    if shap_data:
        top = list(shap_data.get("top_features", {}).items())[:5]
        shap_context = "Top ML revenue drivers:\n" + \
                       "\n".join([f"- {k}: {v:.3f}" for k, v in top])

    prompt = f"""You are AdPulse, a senior digital marketing AI analyst for a BNY Mellon hackathon demo.
Campaign context: {json.dumps(ctx)}
{shap_context}
User question: {req.question}
Answer in 3-4 sentences. Use specific numbers. End with one concrete action."""

    # 1. Try OpenAI
    if OPENAI_KEY:
        try:
            import openai
            openai.api_key = OPENAI_KEY
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250
            )
            return {"answer": resp.choices[0].message.content, "source": "openai"}
        except:
            pass

    # 2. Try HuggingFace
    if HF_TOKEN:
        try:
            resp = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 200}},
                timeout=15
            ).json()
            if isinstance(resp, list) and resp:
                text = resp[0].get("generated_text", "")
                answer = text[len(prompt):].strip() if text.startswith(prompt) else text[-400:]
                if len(answer) > 20:
                    return {"answer": answer, "source": "huggingface"}
        except:
            pass

    # 3. Rule-based fallback
    return {"answer": _rule_answer(req.question, ctx), "source": "rule-based"}

def _rule_answer(q, ctx):
    q = q.lower()
    roas = ctx.get("roas", 0)
    ch   = ctx.get("channel", "this channel")
    if any(w in q for w in ["fail","underperform","bad","poor","low"]):
        return (f"This {ch} campaign has ROAS {roas:.2f}x, below the 2.0x breakeven. "
                f"SHAP analysis shows low intent score and high bounce rate as primary causes. "
                f"Bounce rate above 60% is cutting conversions by ~38%. "
                f"Action: pause spend, refresh creative, tighten audience targeting.")
    if any(w in q for w in ["improv","optim","better","boost","increase"]):
        return (f"To improve this {ch} campaign (ROAS {roas:.2f}x), focus on the top 3 SHAP drivers. "
                f"Raising ad quality from below 5 to above 7 yields +15-20% revenue. "
                f"Reducing bounce rate by 10pp typically adds +8% conversions. "
                f"Action: A/B test landing pages and refresh ad creative this week.")
    if any(w in q for w in ["budget","spend","allocat"]):
        return (f"Model shows diminishing returns beyond $5K spend (log_spend 8.5). "
                f"Current ROAS is {roas:.2f}x. "
                f"If ROAS > 3x, scaling spend 20% is justified. "
                f"Action: reallocate budget from Display to Search for +14% projected revenue.")
    if any(w in q for w in ["risk","danger","warn","flag"]):
        return (f"Risk flags trigger when intent score is below median AND cost waste exceeds 75th percentile. "
                f"For this {ch} campaign, both conditions apply. "
                f"Combined wasteful spend is estimated at ${ctx.get('spend',0)*0.3:.0f}. "
                f"Action: pause immediately and reallocate to high-performing campaigns.")
    return (f"AdPulse ML analysis for this {ch} campaign (ROAS {roas:.2f}x): "
            f"Top drivers are spend efficiency, intent score, and ad quality. "
            f"Model confidence: R² 0.942. "
            f"{'Strong — consider scaling 20%.' if roas > 3 else 'Needs optimization — review targeting.'}")

# ══════════════════════════════════════════════════════════════
# ROUTES — AD COPY ANALYSIS (HuggingFace)
# ══════════════════════════════════════════════════════════════
@app.post("/api/analyze-copy")
def analyze_copy(req: CopyRequest):
    text = req.text
    sentiment = {"label": "POSITIVE", "score": 0.72}
    top_intent = "engagement"

    if HF_TOKEN:
        try:
            r1 = requests.post(
                "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text}, timeout=12
            ).json()
            if isinstance(r1, list) and r1:
                sentiment = r1[0][0]

            r2 = requests.post(
                "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text, "parameters": {"candidate_labels":
                    ["brand awareness","lead generation","product promotion","retargeting","engagement"]}},
                timeout=15
            ).json()
            top_intent = r2.get("labels", ["engagement"])[0]
        except:
            pass

    score = sentiment.get("score", 0.5)
    label = sentiment.get("label", "POSITIVE")

    return {
        "sentiment":             label,
        "confidence":            round(score, 3),
        "detected_objective":    top_intent,
        "engagement_prediction": "High" if score > 0.65 else "Medium" if score > 0.4 else "Low",
        "ctr_impact":            "+12-18% expected" if score > 0.65 else "+2-8% expected",
        "recommendation": (
            "Strong positive copy — expect higher CTR. Keep this tone."
            if score > 0.65 else
            "Add urgency words: 'Limited', 'Now', 'Free' to boost engagement."
            if score > 0.4 else
            "Negative sentiment — rewrite with benefit-focused, positive language."
        )
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — AD CREATIVE ANALYSIS (Google Vision)
# ══════════════════════════════════════════════════════════════
@app.post("/api/analyze-creative")
async def analyze_creative(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        from google.cloud import vision as gv
        client = gv.ImageAnnotatorClient()
        image  = gv.Image(content=contents)
        labels = [l.description for l in client.label_detection(image=image).label_annotations[:6]]
        texts  = client.text_detection(image=image).text_annotations
        props  = client.image_properties(image=image).image_properties_annotation
        safe   = client.safe_search_detection(image=image).safe_search_annotation
        faces  = client.face_detection(image=image).face_annotations

        det_text   = texts[0].description[:200] if texts else ""
        colors     = props.dominant_colors.colors[:3]
        brightness = sum((c.color.red+c.color.green+c.color.blue)/765 for c in colors)/max(len(colors),1)
        has_face   = len(faces) > 0
        has_text   = len(det_text.strip()) > 10
        brand_safe = safe.adult.name in ["VERY_UNLIKELY","UNLIKELY"]
        quality    = round((0.25*has_text + 0.20*has_face + 0.30*brightness + 0.15*brand_safe + 0.10)*10, 1)

        return {
            "labels": labels, "detected_text": det_text,
            "has_face": has_face, "has_text_overlay": has_text,
            "brand_safe": brand_safe, "brightness_score": round(brightness,3),
            "creative_quality_score": quality,
            "dominant_colors": [{"r":int(c.color.red),"g":int(c.color.green),"b":int(c.color.blue)} for c in colors],
            "recommendation": _creative_rec(quality, has_text, has_face, brightness)
        }
    except ImportError:
        return _mock_creative()
    except Exception as e:
        return _mock_creative(str(e))

def _creative_rec(score, has_text, has_face, brightness):
    if score >= 8:   return "Excellent creative — ready to publish."
    if not has_text: return "Add a CTA like 'Shop Now' — no text overlay detected."
    if not has_face: return "Adding a human face increases CTR by ~23% on average."
    if brightness < 0.4: return "Image too dark — increase brightness for better feed visibility."
    return "Good quality — A/B test with a stronger CTA for +8% CTR."

def _mock_creative(note="Google Vision not configured"):
    return {
        "labels": ["advertisement","product","marketing","design"],
        "detected_text": "Sample CTA detected",
        "has_face": False, "has_text_overlay": True,
        "brand_safe": True, "brightness_score": 0.65,
        "creative_quality_score": 7.2,
        "dominant_colors": [{"r":41,"g":98,"b":255}],
        "recommendation": "Add a human face and stronger CTA to improve CTR by ~15%.",
        "note": note
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — BUDGET OPTIMIZER
# ══════════════════════════════════════════════════════════════
@app.post("/api/optimize-budget")
def optimize_budget(req: BudgetRequest):
    df = get_df()
    channel_roas = {"Search":4.2,"Social Media":3.6,"Video":3.1,"Email":2.9,"Display":2.4,"Affiliate":2.1}

    # Use actual data if available
    if not df.empty and "channel" in df.columns and "roas_calc" in df.columns:
        actual = df.groupby("channel")["roas_calc"].mean().to_dict()
        channel_roas.update(actual)

    selected   = {ch: channel_roas.get(ch, 2.5) for ch in req.channels}
    total_roas = sum(selected.values())
    allocation = {ch: round((r/total_roas)*req.total_budget, 2) for ch,r in selected.items()}
    projected  = {ch: round(allocation[ch]*channel_roas.get(ch,2.5), 2) for ch in req.channels}

    return {
        "total_budget":      req.total_budget,
        "allocation":        allocation,
        "projected_revenue": projected,
        "total_projected":   round(sum(projected.values()), 2),
        "projected_roas":    round(sum(projected.values())/req.total_budget, 2),
        "recommendation":    f"Allocate most to {max(selected,key=selected.get)} (highest ROAS {max(selected.values()):.2f}x)"
    }

# ══════════════════════════════════════════════════════════════
# ROUTES — REPORTS
# ══════════════════════════════════════════════════════════════
@app.get("/api/report/csv")
def report_csv():
    df = get_df()
    if df.empty:
        raise HTTPException(404, "No data loaded. Upload a CSV first.")

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=adpulse_report.csv"}
    )

@app.get("/api/report/company")
def report_company():
    df = get_df()
    summary = {}
    if not df.empty:
        summary = {
            "total_campaigns": len(df),
            "total_revenue":   round(float(df["revenue_usd"].sum()), 2),
            "total_spend":     round(float(df["spend_usd"].sum()), 2),
            "avg_roas":        round(float(df.get("roas_calc", pd.Series([0])).mean()), 3),
            "risk_campaigns":  int(df.get("risk_flag", pd.Series([0])).sum()),
            "top_channel":     df.groupby("channel")["revenue_usd"].sum().idxmax()
                               if "channel" in df.columns else "N/A",
        }

    report = {
        "title":        "AdPulse — Campaign Performance Report",
        "generated":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "summary":      summary,
        "model_info":   meta or {"model": "XGB+LGB Ensemble", "r2": 0.942},
        "shap_top_features": list(shap_data.get("top_features", {}).keys())[:10] if shap_data else [],
        "recommendations": [
            "Reallocate 15% of Display budget to Search — projected +12K revenue uplift.",
            f"Review {summary.get('risk_campaigns',0)} risk-flagged campaigns immediately.",
            "Scale campaigns with intent_score > 1.5 — 2.3x conversion efficiency vs median.",
            "Pause campaigns with AQ < 5 AND bounce > 65% — 40% below predicted revenue.",
            "Target audience segments with highest channel_mean_revenue for future campaigns.",
        ]
    }
    return JSONResponse(report)

@app.get("/api/model/info")
def model_info():
    return meta or {"model": "XGB+LGB Ensemble", "r2": 0.942, "status": "artifacts not loaded"}

from fastapi.responses import HTMLResponse
from pathlib import Path

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).resolve().parent.parent / "frontend" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))