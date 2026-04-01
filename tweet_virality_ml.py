"""
============================================================
  TWEET VIRALITY PREDICTOR — FINAL CLEAN VERSION
  + MODEL SAVING (run this FIRST before the dashboard)
============================================================
"""

# ─── 1. IMPORTS ───────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import warnings
import joblib
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

import matplotlib.pyplot as plt
import seaborn as sns

# ─── 2. LOAD DATA ─────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Loading Datasets")
print("=" * 60)

dfs = []

def load_file(file):
    try:
        if file.endswith(".csv"):
            df = pd.read_csv(file, low_memory=False)
        elif file.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return None
        print(f"  ✅ Loaded {file}: {df.shape[0]:,} rows")
        return df
    except:
        print(f"  ❌ Failed to load {file}")
        return None

files = ["data_analysis.csv", "data_visualization.csv", "data_science.csv"]

for f in files:
    df_temp = load_file(f)
    if df_temp is not None:
        df_temp["source_dataset"] = f
        dfs.append(df_temp)

if not dfs:
    raise RuntimeError("No datasets loaded!")

common_cols = set(dfs[0].columns)
for d in dfs[1:]:
    common_cols = common_cols & set(d.columns)
dfs = [d[list(common_cols)].copy() for d in dfs]

df = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Combined dataset: {df.shape}")

# ─── 3. HANDLE MISSING COLUMNS ────────────────────────────
for col in ["hashtags","mentions","urls","photos","video","retweet","replies_count","likes_count","retweets_count"]:
    if col not in df.columns:
        df[col] = 0

# ─── 4. CLEAN DATA ────────────────────────────────────────
df = df[df["language"] == "en"].copy()
df = df.dropna(subset=["tweet"])

for col in ["likes_count", "retweets_count", "replies_count", "video"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["hour"]       = df["created_at"].dt.hour
df["day_of_week"]= df["created_at"].dt.dayofweek
df["month"]      = df["created_at"].dt.month
df["year"]       = df["created_at"].dt.year

# ─── 5. FEATURE ENGINEERING ───────────────────────────────
def clean_tweet(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip().lower()

df["tweet_clean"]  = df["tweet"].apply(clean_tweet)
df["tweet_length"] = df["tweet"].str.len()
df["word_count"]   = df["tweet_clean"].str.split().str.len()

def count_items(val):
    if pd.isna(val) or val in ("[]", "", "nan"):
        return 0
    try:
        return len(eval(val)) if isinstance(val, str) else 0
    except:
        return 0

df["hashtag_count"] = df["hashtags"].apply(count_items)
df["mention_count"] = df["mentions"].apply(count_items)
df["has_hashtag"]   = (df["hashtag_count"] > 0).astype(int)
df["has_mention"]   = (df["mention_count"] > 0).astype(int)
df["has_url"]       = df["urls"].apply(lambda x: 0 if pd.isna(x) or x in ("[]","") else 1)
df["has_photo"]     = df["photos"].apply(lambda x: 0 if pd.isna(x) or x in ("[]","") else 1)
df["has_video"]     = (df["video"] > 0).astype(int)
df["is_retweet"]    = df["retweet"].astype(int)

# ─── 6. TARGET VARIABLE ───────────────────────────────────
df["engagement"] = df["likes_count"] + df["retweets_count"] + df["replies_count"]
threshold        = df["engagement"].quantile(0.75)
df["viral"]      = (df["engagement"] > threshold).astype(int)
print(f"  Viral threshold (75th percentile): {threshold:.0f}")

# ─── 7. FEATURES ──────────────────────────────────────────
META_FEATURES = [
    "tweet_length","word_count","hashtag_count","mention_count",
    "has_hashtag","has_mention","has_url","has_photo","has_video",
    "is_retweet","hour","day_of_week","month","year","replies_count"
]

X_meta = df[META_FEATURES].fillna(0)
X_text = df["tweet_clean"]
y      = df["viral"]

# ─── 8. SPLIT ─────────────────────────────────────────────
X_meta_train, X_meta_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_meta, X_text, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 9. TF-IDF + SCALER ───────────────────────────────────
tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test  = tfidf.transform(X_text_test)

scaler = StandardScaler()
X_meta_train_s = scaler.fit_transform(X_meta_train)
X_meta_test_s  = scaler.transform(X_meta_test)

X_train = sp.hstack([X_tfidf_train, sp.csr_matrix(X_meta_train_s)])
X_test  = sp.hstack([X_tfidf_test,  sp.csr_matrix(X_meta_test_s)])

# ─── 10. TRAIN MODELS ─────────────────────────────────────
results = {}

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
results["Logistic"] = {"model": lr, "pred": lr.predict(X_test), "prob": lr.predict_proba(X_test)[:,1]}

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
results["RandomForest"] = {"model": rf, "pred": rf.predict(X_test), "prob": rf.predict_proba(X_test)[:,1]}

print("Training Gradient Boosting...")
X_train_dense = X_train.toarray()
X_test_dense  = X_test.toarray()
gb = GradientBoostingClassifier()
gb.fit(X_train_dense, y_train)
results["GradientBoost"] = {"model": gb, "pred": gb.predict(X_test_dense), "prob": gb.predict_proba(X_test_dense)[:,1]}

# ─── 11. EVALUATION ───────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL PERFORMANCE")
print("=" * 60)

best_model_name = None
best_auc = 0
model_metrics = {}

for name, res in results.items():
    acc = accuracy_score(y_test, res["pred"])
    auc = roc_auc_score(y_test, res["prob"])
    cm  = confusion_matrix(y_test, res["pred"])
    model_metrics[name] = {"accuracy": round(acc,4), "auc": round(auc,4), "cm": cm.tolist()}
    print(f"  {name:<16} Accuracy={acc:.4f}  AUC={auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model_name = name

print(f"\n  🏆 BEST MODEL: {best_model_name} (AUC={best_auc:.4f})")
print("\nClassification Report:\n")
print(classification_report(y_test, results[best_model_name]["pred"]))

# ─── 12. SAVE ALL ARTIFACTS ───────────────────────────────
print("=" * 60)
print("  SAVING MODEL ARTIFACTS")
print("=" * 60)

os.makedirs("model_artifacts", exist_ok=True)

# Save individual models
for name, res in results.items():
    path = f"model_artifacts/{name}_model.pkl"
    joblib.dump(res["model"], path)
    print(f"  ✅ {name} model        → {path}")

# Save preprocessing objects
joblib.dump(tfidf,  "model_artifacts/tfidf_vectorizer.pkl")
joblib.dump(scaler, "model_artifacts/scaler.pkl")
print("  ✅ TF-IDF vectorizer   → model_artifacts/tfidf_vectorizer.pkl")
print("  ✅ StandardScaler      → model_artifacts/scaler.pkl")

# Save metrics for dashboard (no re-training needed)
joblib.dump(model_metrics, "model_artifacts/model_metrics.pkl")
print("  ✅ Model metrics       → model_artifacts/model_metrics.pkl")

# Save config (features list, thresholds, test results)
config = {
    "META_FEATURES":  META_FEATURES,
    "threshold":      float(threshold),
    "best_model":     best_model_name,
    "total_tweets":   int(len(df)),
    "viral_pct":      float(df["viral"].mean() * 100),
    "y_test":         y_test.tolist(),
    "probs":          {k: v["prob"].tolist() for k, v in results.items()},
    "preds":          {k: v["pred"].tolist() for k, v in results.items()},
}
joblib.dump(config, "model_artifacts/config.pkl")
print("  ✅ Config              → model_artifacts/config.pkl")

# Save processed dataframe for EDA tab in dashboard
df[[
    "tweet","tweet_clean","tweet_length","word_count",
    "hashtag_count","mention_count","has_hashtag","has_mention",
    "has_url","has_photo","has_video","is_retweet",
    "hour","day_of_week","month","year",
    "likes_count","retweets_count","replies_count",
    "engagement","viral","source_dataset"
]].to_csv("model_artifacts/processed_data.csv", index=False)
print("  ✅ Processed dataset   → model_artifacts/processed_data.csv")

print(f"\n🎉 All artifacts saved in ./model_artifacts/")
print("   ▶  Now run:  streamlit run dashboard.py")

# ─── 13. PREDICTION FUNCTION ──────────────────────────────
def predict_tweet(text):
    clean = clean_tweet(text)
    tfidf_feat = tfidf.transform([clean])
    meta = pd.DataFrame([{
        "tweet_length": len(text), "word_count": len(clean.split()),
        "hashtag_count": text.count("#"), "mention_count": text.count("@"),
        "has_hashtag": int("#" in text), "has_mention": int("@" in text),
        "has_url": int("http" in text), "has_photo": 0, "has_video": 0,
        "is_retweet": 0, "hour": 12, "day_of_week": 2,
        "month": 6, "year": 2021, "replies_count": 0
    }])
    meta_scaled = scaler.transform(meta[META_FEATURES])
    X_new = sp.hstack([tfidf_feat, sp.csr_matrix(meta_scaled)])
    prob  = rf.predict_proba(X_new)[0][1]
    return ("🔥 Viral" if prob > 0.5 else "📉 Not Viral"), prob

tweet = "Excited to share new AI research! #AI #MachineLearning"
label, prob = predict_tweet(tweet)
print(f"\nDemo:\n  \"{tweet}\"\n  → {label}  ({prob:.2%} confidence)")