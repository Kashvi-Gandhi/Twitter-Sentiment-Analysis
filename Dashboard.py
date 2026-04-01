"""
============================================================
  TWEET VIRALITY PREDICTOR — STREAMLIT DASHBOARD
  Run: streamlit run dashboard.py
  ⚠️  Run tweet_virality_ml.py FIRST to generate model_artifacts/
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Absolute path to model_artifacts/ (works no matter where you run streamlit from)
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

def artifact(filename):
    """Return the absolute path to a file inside model_artifacts/."""
    return os.path.join(ARTIFACTS_DIR, filename)

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Tweet Virality Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  /* Dark sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d0d1a 0%, #111827 100%);
    border-right: 1px solid #1e293b;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
  section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div { color: #38bdf8 !important; }

  /* Main background */
  .main { background: #f8fafc; }

  /* Metric cards */
  .metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.07), 0 4px 16px rgba(0,0,0,.04);
    border-left: 4px solid;
    margin-bottom: 8px;
  }
  .metric-card h3 { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin: 0 0 4px; color: #64748b; }
  .metric-card p  { font-size: 32px; font-weight: 700; margin: 0; line-height: 1; font-family: 'JetBrains Mono', monospace; }

  /* Section header */
  .section-header {
    background: white;
    border-radius: 12px;
    padding: 14px 20px;
    margin: 12px 0 16px;
    border-bottom: 3px solid #38bdf8;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .section-header h2 { font-size: 18px; font-weight: 700; margin: 0; color: #0f172a; }

  /* Prediction box */
  .pred-viral   { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 2px solid #10b981; border-radius: 16px; padding: 24px; text-align: center; }
  .pred-notviral{ background: linear-gradient(135deg, #fff1f2, #ffe4e6); border: 2px solid #f43f5e; border-radius: 16px; padding: 24px; text-align: center; }
  .pred-viral   h1, .pred-notviral h1 { font-size: 42px; margin: 0 0 4px; }
  .pred-viral   p,  .pred-notviral p  { font-size: 15px; color: #334155; margin: 4px 0 0; }

  /* Confidence bar */
  .conf-bar-wrap { background: #e2e8f0; border-radius: 99px; height: 12px; overflow: hidden; margin-top: 10px; }
  .conf-bar      { height: 100%; border-radius: 99px; }

  /* Tweet badge */
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 12px; font-weight: 600; letter-spacing: .04em;
  }
  .badge-green { background: #d1fae5; color: #065f46; }
  .badge-red   { background: #ffe4e6; color: #9f1239; }

  /* Tip box */
  .tip-box {
    background: linear-gradient(135deg, #eff6ff, #e0f2fe);
    border: 1px solid #bae6fd; border-radius: 12px;
    padding: 16px 20px; margin-top: 16px;
  }
  .tip-box h4 { margin: 0 0 8px; color: #0369a1; font-size: 14px; }
  .tip-box ul { margin: 0; padding-left: 18px; color: #334155; font-size: 13px; }
  .tip-box li { margin-bottom: 4px; }

  /* Hide streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    try:
        rf      = joblib.load(artifact("RandomForest_model.pkl"))
        lr      = joblib.load(artifact("Logistic_model.pkl"))
        gb      = joblib.load(artifact("GradientBoost_model.pkl"))
        tfidf   = joblib.load(artifact("tfidf_vectorizer.pkl"))
        scaler  = joblib.load(artifact("scaler.pkl"))
        metrics = joblib.load(artifact("model_metrics.pkl"))
        config  = joblib.load(artifact("config.pkl"))
        df      = pd.read_csv(artifact("processed_data.csv"))
        return rf, lr, gb, tfidf, scaler, metrics, config, df, None
    except Exception as e:
        return None, None, None, None, None, None, None, None, str(e)

rf, lr, gb, tfidf, scaler, metrics, config, df, load_err = load_artifacts()

META_FEATURES = [
    "tweet_length","word_count","hashtag_count","mention_count",
    "has_hashtag","has_mention","has_url","has_photo","has_video",
    "is_retweet","hour","day_of_week","month","year","replies_count"
]

def clean_tweet(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip().lower()

def predict(text, hour=12, has_photo=0, has_video=0):
    clean = clean_tweet(text)
    tfidf_feat = tfidf.transform([clean])
    meta = pd.DataFrame([{
        "tweet_length": len(text), "word_count": len(clean.split()),
        "hashtag_count": text.count("#"), "mention_count": text.count("@"),
        "has_hashtag": int("#" in text), "has_mention": int("@" in text),
        "has_url": int("http" in text), "has_photo": has_photo,
        "has_video": has_video, "is_retweet": 0,
        "hour": hour, "day_of_week": 2, "month": 6, "year": 2021, "replies_count": 0
    }])
    meta_s = scaler.transform(meta[META_FEATURES])
    X_new  = sp.hstack([tfidf_feat, sp.csr_matrix(meta_s)])
    prob   = rf.predict_proba(X_new)[0][1]
    return prob


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔥 Tweet Virality")
    st.markdown("**Data Science Tweets 2010–2021**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Predict a Tweet", "📊 Model Performance", "📈 Data Explorer"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if load_err is None and config:
        st.markdown(f"**Dataset:** {config['total_tweets']:,} tweets")
        st.markdown(f"**Viral rate:** {config['viral_pct']:.1f}%")
        st.markdown(f"**Best model:** {config['best_model']}")
        st.markdown(f"**Threshold:** top 25% engagement")
    st.markdown("---")
    st.caption("College ML Project · 2025")


# ══════════════════════════════════════════════════════════
#  ERROR STATE
# ══════════════════════════════════════════════════════════
if load_err:
    st.error("⚠️ Could not load model artifacts")
    st.markdown(f"**Error details:** `{load_err}`")
    st.markdown(f"**Looking for artifacts in:** `{ARTIFACTS_DIR}`")
    st.info("""
    **Fix:** Run the training script first, from the same folder as `dashboard.py`:
    ```bash
    cd /path/to/your/project
    python tweet_virality_ml.py
    streamlit run dashboard.py
    ```
    This will create the `model_artifacts/` folder next to your scripts.
    """)
    st.stop()


# ══════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🔥 Tweet Virality Predictor")
    st.markdown("Predicting whether Data Science tweets go viral using **NLP + Machine Learning**")
    st.markdown("---")

    # ── Top KPI cards ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Tweets", f"{config['total_tweets']:,}", "#38bdf8"),
        (c2, "Viral Rate",   f"{config['viral_pct']:.1f}%",           "#10b981"),
        (c3, "Best Model",   config["best_model"],                      "#f59e0b"),
        (c4, "Best AUC",     f"{metrics[config['best_model']]['auc']:.4f}", "#8b5cf6"),
    ]
    for col, title, val, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
              <h3>{title}</h3>
              <p style="color:{color}">{val}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-header"><h2>📅 Tweets Posted by Hour</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        hour_eng = df.groupby("hour")["engagement"].mean()
        colors = ["#38bdf8" if v < hour_eng.median() else "#0ea5e9" for v in hour_eng]
        ax.bar(hour_eng.index, hour_eng.values, color=colors, width=0.7, edgecolor="white")
        ax.set_xlabel("Hour of Day (UTC+5:30)", fontsize=10)
        ax.set_ylabel("Avg Engagement", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header"><h2>📊 Viral vs Non-Viral Split</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["viral"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=["Not Viral", "Viral"],
            autopct="%1.1f%%",
            colors=["#f43f5e", "#10b981"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11, "fontweight": "600"}
        )
        for a in autotexts: a.set_color("white"); a.set_fontweight("700")
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    # ── Bottom row ─────────────────────────────────────────
    left2, right2 = st.columns([1, 1])

    with left2:
        st.markdown('<div class="section-header"><h2>📆 Tweets by Year</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        year_counts = df.groupby("year").size()
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.2, color="#38bdf8")
        ax.plot(year_counts.index, year_counts.values, color="#38bdf8", linewidth=2.5, marker="o", markersize=6)
        ax.set_xlabel("Year"); ax.set_ylabel("Number of Tweets")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    with right2:
        st.markdown('<div class="section-header"><h2>🔗 Feature Impact on Virality</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        feat_cols = ["has_hashtag","has_mention","has_url","has_photo","has_video"]
        viral_rates = [df[df[f]==1]["viral"].mean()*100 for f in feat_cols]
        labels = ["Has Hashtag","Has Mention","Has URL","Has Photo","Has Video"]
        colors = ["#38bdf8","#10b981","#f59e0b","#8b5cf6","#f43f5e"]
        bars = ax.barh(labels, viral_rates, color=colors, height=0.55, edgecolor="white")
        for b, v in zip(bars, viral_rates):
            ax.text(v+0.5, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=10, fontweight="600")
        ax.set_xlabel("Viral Rate (%)")
        ax.set_xlim(0, max(viral_rates)+10)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE 2: PREDICT A TWEET
# ══════════════════════════════════════════════════════════
elif page == "🔍 Predict a Tweet":
    st.markdown("# 🔍 Predict a Tweet")
    st.markdown("Type any tweet below and the ML model will predict if it will go viral.")
    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        tweet_text = st.text_area(
            "✍️ Your Tweet",
            placeholder="e.g., Thrilled to share our new #MachineLearning paper! #DataScience #AI 🚀",
            height=130
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            post_hour = st.slider("⏰ Post Hour", 0, 23, 10)
        with col_b:
            has_photo = st.selectbox("📷 Photo?", ["No", "Yes"])
        with col_c:
            has_video = st.selectbox("🎬 Video?", ["No", "Yes"])

        predict_btn = st.button("🚀 Predict Virality", use_container_width=True, type="primary")

    with right:
        st.markdown("**💡 Example Tweets**")
        examples = [
            "Excited to release our #Python library for NLP! 10k downloads in a week. #MachineLearning #OpenSource",
            "Just had coffee.",
            "New #DataScience tutorial: How to build a tweet classifier from scratch 🔥 #AI #Python #Tutorial",
        ]
        for ex in examples:
            if st.button(ex[:65] + "...", key=ex):
                tweet_text = ex
                predict_btn = True

    if predict_btn and tweet_text.strip():
        prob = predict(
            tweet_text,
            hour=post_hour,
            has_photo=int(has_photo == "Yes"),
            has_video=int(has_video == "Yes")
        )
        is_viral = prob > 0.5
        pct      = int(prob * 100)

        st.markdown("<br>", unsafe_allow_html=True)

        if is_viral:
            st.markdown(f"""
            <div class="pred-viral">
              <h1>🔥 VIRAL!</h1>
              <p>This tweet is predicted to go <strong>viral</strong></p>
              <div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct}%;background:linear-gradient(90deg,#10b981,#059669)"></div></div>
              <p style="margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#065f46">{pct}% confidence</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-notviral">
              <h1>📉 Not Viral</h1>
              <p>This tweet is predicted to <strong>not go viral</strong></p>
              <div class="conf-bar-wrap"><div class="conf-bar" style="width:{100-pct}%;background:linear-gradient(90deg,#f43f5e,#e11d48)"></div></div>
              <p style="margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#9f1239">{100-pct}% likely not viral</p>
            </div>""", unsafe_allow_html=True)

        # Tweet anatomy breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>🔬 Tweet Anatomy</h2></div>', unsafe_allow_html=True)
        a1, a2, a3, a4, a5 = st.columns(5)
        checks = [
            (a1, "Characters",   len(tweet_text), "📝", "#38bdf8"),
            (a2, "Words",        len(tweet_text.split()), "💬", "#10b981"),
            (a3, "Hashtags",     tweet_text.count("#"), "🏷️", "#f59e0b"),
            (a4, "Mentions",     tweet_text.count("@"), "👤", "#8b5cf6"),
            (a5, "Has URL",      "Yes" if "http" in tweet_text else "No", "🔗", "#f43f5e"),
        ]
        for col, label, val, icon, color in checks:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color};padding:14px 16px">
                  <h3>{icon} {label}</h3>
                  <p style="font-size:24px;color:{color}">{val}</p>
                </div>""", unsafe_allow_html=True)

        # Tips
        tips = []
        if tweet_text.count("#") == 0: tips.append("Add 2–3 relevant hashtags like <strong>#DataScience #AI</strong>")
        if len(tweet_text) < 80:       tips.append("Longer tweets (100–200 chars) tend to perform better")
        if "http" not in tweet_text:   tips.append("Include a link to an article, paper, or tool")
        if has_photo == "No":          tips.append("Attach an image — visual tweets get ~3× more engagement")
        if post_hour < 7 or post_hour > 21: tips.append("Post between 8 AM–9 PM for higher visibility")

        if tips:
            tips_html = "".join(f"<li>{t}</li>" for t in tips)
            st.markdown(f"""
            <div class="tip-box">
              <h4>🚀 Tips to Boost Virality</h4>
              <ul>{tips_html}</ul>
            </div>""", unsafe_allow_html=True)

    elif predict_btn:
        st.warning("Please enter a tweet first!")


# ══════════════════════════════════════════════════════════
#  PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance")
    st.markdown("Comparing all three trained models on the 20% test set.")
    st.markdown("---")

    # ── Metric cards ───────────────────────────────────────
    cols = st.columns(3)
    m_colors = {"Logistic": "#38bdf8", "RandomForest": "#10b981", "GradientBoost": "#f59e0b"}
    for col, (name, m) in zip(cols, metrics.items()):
        color = m_colors.get(name, "#8b5cf6")
        with col:
            is_best = name == config["best_model"]
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}">
              <h3>{'🏆 ' if is_best else ''}{name}</h3>
              <p style="color:{color};font-size:26px">{m['accuracy']:.4f}</p>
              <p style="font-size:13px;color:#64748b;margin-top:4px">Accuracy</p>
              <p style="font-size:20px;font-weight:700;color:#334155;margin:0">{m['auc']:.4f} <span style="font-size:12px;color:#94a3b8">AUC-ROC</span></p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curve + Confusion Matrix ───────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-header"><h2>📈 ROC Curves</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        y_test = config["y_test"]
        for name, color in m_colors.items():
            if name in config["probs"]:
                fpr, tpr, _ = roc_curve(y_test, config["probs"][name])
                auc = metrics[name]["auc"]
                lw  = 2.5 if name == config["best_model"] else 1.5
                ax.plot(fpr, tpr, color=color, linewidth=lw, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown(f'<div class="section-header"><h2>🎯 Confusion Matrix — {config["best_model"]}</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        cm = np.array(metrics[config["best_model"]]["cm"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Not Viral","Viral"],
                    yticklabels=["Not Viral","Viral"],
                    ax=ax, cbar=False, linewidths=1,
                    annot_kws={"fontsize":14,"fontweight":"bold"})
        ax.set_ylabel("Actual", fontsize=11); ax.set_xlabel("Predicted", fontsize=11)
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    # ── Accuracy bar comparison ─────────────────────────────
    st.markdown('<div class="section-header"><h2>📊 Model Accuracy Comparison</h2></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    names = list(metrics.keys())
    accs  = [metrics[n]["accuracy"] for n in names]
    aucs  = [metrics[n]["auc"] for n in names]
    x     = np.arange(len(names))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#38bdf8", edgecolor="white")
    bars2 = ax.bar(x + 0.2, aucs, 0.35, label="AUC-ROC",  color="#10b981", edgecolor="white")
    for b in bars1 + bars2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="600")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.5, 1.0); ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE 4: DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.markdown("# 📈 Data Explorer")
    st.markdown("Explore the processed dataset used for training.")
    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-header"><h2>💬 Tweet Length Distribution</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        viral_tweets    = df[df["viral"]==1]["tweet_length"]
        nonviral_tweets = df[df["viral"]==0]["tweet_length"]
        ax.hist(nonviral_tweets, bins=40, alpha=0.6, color="#f43f5e", label="Not Viral", density=True)
        ax.hist(viral_tweets,    bins=40, alpha=0.6, color="#10b981", label="Viral",     density=True)
        ax.set_xlabel("Tweet Length (chars)"); ax.set_ylabel("Density")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header"><h2>📅 Day-of-Week Engagement</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow_eng = df.groupby("day_of_week")["engagement"].mean()
        ax.bar(days[:len(dow_eng)], dow_eng.values, color="#8b5cf6", edgecolor="white", width=0.6)
        ax.set_ylabel("Avg Engagement"); ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    left2, right2 = st.columns([1, 1])

    with left2:
        st.markdown('<div class="section-header"><h2>#️⃣ Hashtag Count vs Engagement</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        hc = df[df["hashtag_count"] <= 10].groupby("hashtag_count")["engagement"].median()
        ax.plot(hc.index, hc.values, color="#f59e0b", linewidth=2.5, marker="o", markersize=7)
        ax.fill_between(hc.index, hc.values, alpha=0.15, color="#f59e0b")
        ax.set_xlabel("Number of Hashtags"); ax.set_ylabel("Median Engagement")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    with right2:
        st.markdown('<div class="section-header"><h2>📁 Dataset Source Breakdown</h2></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        src = df["source_dataset"].value_counts()
        colors_src = ["#38bdf8","#10b981","#f59e0b"]
        wedges, texts, autotexts = ax.pie(
            src.values, labels=src.index,
            autopct="%1.1f%%", colors=colors_src[:len(src)],
            startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2},
            textprops={"fontsize":10}
        )
        for a in autotexts: a.set_color("white"); a.set_fontweight("700")
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)

    # ── Raw data table ─────────────────────────────────────
    st.markdown('<div class="section-header"><h2>📋 Sample Data</h2></div>', unsafe_allow_html=True)
    show_cols = ["tweet","tweet_length","hashtag_count","mention_count",
                 "has_url","has_photo","likes_count","retweets_count","engagement","viral"]
    filter_viral = st.radio("Filter:", ["All", "Viral Only", "Not Viral Only"], horizontal=True)
    df_show = df.copy()
    if filter_viral == "Viral Only":    df_show = df_show[df_show["viral"]==1]
    elif filter_viral == "Not Viral Only": df_show = df_show[df_show["viral"]==0]

    st.dataframe(
        df_show[show_cols].head(50).reset_index(drop=True),
        use_container_width=True,
        height=300
    )
    st.caption(f"Showing 50 of {len(df_show):,} rows")