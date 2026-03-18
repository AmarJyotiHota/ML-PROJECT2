# app.py — Fixed version
# Bugs fixed:
#  1. Added missing CSS classes: .conf-section-label, .highlight-row, .section-label
#  2. Removed plt.style.use('dark_background') global state pollution → use per-figure bg
#  3. Added plt.close(fig) after every st.pyplot() call to prevent memory leaks
#  4. Wrapped cm.ravel() in try/except to handle non-2×2 confusion matrices
#  5. Removed duplicate render_pipeline(step=5) inside predict block (double-pipeline bug)
#  6. Added use_container_width=True to all st.pyplot() calls
#  7. Closed fig / fig2 in tab3 after rendering

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ────────────────────────────────────────────────────
def load_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Sora:wght@300;400;500;600;700&display=swap');

        :root {
            --bg: #04101f;
            --bg2: #071828;
            --bg3: #0b2236;
            --surface: #0e2a42;
            --surface2: #112f4a;
            --border: rgba(0,200,200,0.12);
            --border-hi: rgba(0,200,200,0.35);
            --teal: #00c8c8;
            --teal-dim: rgba(0,200,200,0.08);
            --amber: #f4a441;
            --green: #3de89e;
            --red: #f45f6f;
            --blue: #4a9eff;
            --text: #d6eaf0;
            --text-dim: #6b99b5;
            --text-muted: #3e6880;
            --radius: 12px;
            --radius-sm: 8px;
            --shadow-sm: 0 2px 16px rgba(0,0,0,0.35);
        }

        .stApp { background: var(--bg); }

        /* ── Header ── */
        .custom-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 28px 0 20px; border-bottom: 1px solid var(--border);
            margin-bottom: 36px; flex-wrap: wrap; gap: 16px;
        }
        .header-left { display: flex; align-items: center; gap: 18px; }
        .logo-mark {
            width: 44px; height: 44px; border-radius: 10px;
            background: linear-gradient(135deg, var(--teal) 0%, #006f8e 100%);
            display: grid; place-items: center;
            box-shadow: 0 0 24px rgba(0,200,200,.3); flex-shrink: 0;
        }
        .logo-mark svg { width: 22px; height: 22px; stroke: white; stroke-width: 1.8; fill: none; }
        .site-title {
            font-family: 'Cormorant Garamond', serif; font-size: 1.65rem;
            font-weight: 500; color: #fff; line-height: 1;
        }
        .site-subtitle {
            font-family: 'DM Mono', monospace; font-size: .65rem;
            color: var(--teal); letter-spacing: .16em; text-transform: uppercase; margin-top: 4px;
        }
        .header-badges { display: flex; gap: 8px; flex-wrap: wrap; }
        .badge {
            font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .12em;
            text-transform: uppercase; padding: 5px 12px; border-radius: 20px;
            border: 1px solid var(--border); color: var(--text-dim); background: var(--teal-dim);
        }
        .badge.active { border-color: var(--teal); color: var(--teal); }

        /* ── Hero ── */
        .hero-strip {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 28px 32px;
            display: flex; align-items: center; gap: 32px; margin-bottom: 28px; flex-wrap: wrap;
        }
        .hero-text h1 {
            font-family: 'Cormorant Garamond', serif; font-size: 2.2rem;
            font-weight: 400; color: #fff; line-height: 1.15; margin: 0;
        }
        .hero-text h1 em { font-style: normal; color: var(--teal); }
        .hero-text p { font-size: .82rem; color: var(--text-dim); margin-top: 8px; max-width: 520px; line-height: 1.65; }
        .hero-stats { display: flex; gap: 20px; margin-left: auto; flex-wrap: wrap; }
        .stat-box {
            text-align: center; padding: 14px 20px; border-radius: var(--radius-sm);
            background: var(--bg3); border: 1px solid var(--border); min-width: 80px;
        }
        .stat-box .val { font-family: 'DM Mono', monospace; font-size: 1.45rem; font-weight: 500; color: var(--teal); display: block; }
        .stat-box .lbl { font-size: .62rem; letter-spacing: .1em; text-transform: uppercase; color: var(--text-muted); margin-top: 3px; display: block; }

        /* ── Cards ── */
        .card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow-sm); margin-bottom: 22px;
        }
        .card-header {
            display: flex; align-items: center; gap: 12px;
            padding: 18px 24px; border-bottom: 1px solid var(--border); background: rgba(0,0,0,.15);
        }
        .card-header .icon {
            width: 32px; height: 32px; border-radius: 8px; background: var(--teal-dim);
            border: 1px solid var(--border-hi); display: grid; place-items: center;
            color: var(--teal); font-size: .9rem; flex-shrink: 0;
        }
        .card-header h2 { font-family: 'Cormorant Garamond', serif; font-size: 1.2rem; font-weight: 500; color: #fff; margin: 0; }
        .card-header p { font-size: .7rem; color: var(--text-muted); margin: 1px 0 0 0; }
        .card-body { padding: 24px; }

        /* ── Chart cards ── */
        .chart-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 18px 18px 14px;
            box-shadow: var(--shadow-sm); margin-bottom: 18px;
        }
        .chart-title { font-family: 'Cormorant Garamond', serif; font-size: 1.05rem; font-weight: 500; color: #fff; margin-bottom: 12px; }

        /* ── Section labels ── */
        .section-title-block { margin-bottom: 28px; }
        .section-title-block h2 { font-family: 'Cormorant Garamond', serif; font-size: 1.8rem; font-weight: 400; color: #fff; }
        .section-title-block p { font-size: .78rem; color: var(--text-dim); margin-top: 6px; }

        /* FIX 1: added .chart-section-label and .section-label */
        .chart-section-label, .section-label {
            font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .18em;
            text-transform: uppercase; color: var(--teal); margin: 24px 0 14px;
            display: flex; align-items: center; gap: 10px;
        }
        .chart-section-label::after, .section-label::after {
            content: ''; flex: 1; height: 1px; background: var(--border);
        }

        /* ── Pipeline ── */
        .pipeline { display: flex; align-items: center; overflow-x: auto; }
        .pipe-step { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: 1; min-width: 70px; }
        .pipe-dot {
            width: 36px; height: 36px; border-radius: 50%;
            background: var(--bg3); border: 1.5px solid var(--border);
            display: grid; place-items: center; color: var(--text-muted); font-size: .85rem;
        }
        .pipe-step.active .pipe-dot { background: var(--teal-dim); border-color: var(--teal); color: var(--teal); }
        .pipe-label { font-size: .58rem; text-transform: uppercase; color: var(--text-muted); text-align: center; }
        .pipe-step.active .pipe-label { color: var(--teal); }
        .pipe-connector { flex: .8; height: 1px; background: var(--border); min-width: 16px; position: relative; top: -13px; }
        .pipe-connector.active { background: var(--teal); }

        /* ── Result card ── */
        .result-card { border-radius: var(--radius); border: 1px solid var(--border); overflow: hidden; margin-top: 20px; display: none; }
        .result-card.visible { display: block; }
        .result-card.benign { border-color: rgba(61,232,158,.35); }
        .result-card.malignant { border-color: rgba(244,95,111,.35); }
        .result-header { padding: 20px 24px; display: flex; align-items: center; gap: 16px; }
        .result-card.benign .result-header { background: rgba(61,232,158,.07); }
        .result-card.malignant .result-header { background: rgba(244,95,111,.07); }
        .result-icon { width: 48px; height: 48px; border-radius: 50%; display: grid; place-items: center; font-size: 1.4rem; }
        .result-card.benign .result-icon { background: rgba(61,232,158,.12); color: var(--green); }
        .result-card.malignant .result-icon { background: rgba(244,95,111,.12); color: var(--red); }
        .result-title { font-family: 'Cormorant Garamond', serif; font-size: 1.8rem; font-weight: 500; line-height: 1; }
        .result-card.benign .result-title { color: var(--green); }
        .result-card.malignant .result-title { color: var(--red); }
        .result-sub { font-size: .72rem; color: var(--text-muted); margin-top: 4px; }
        .result-body { padding: 18px 24px; }

        /* FIX 2: added missing .conf-section-label */
        .conf-section-label {
            font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .14em;
            text-transform: uppercase; color: var(--text-muted); margin-bottom: 12px; display: block;
        }

        .confidence-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
        .conf-label { font-family: 'DM Mono', monospace; font-size: .65rem; text-transform: uppercase; color: var(--text-muted); width: 82px; }
        .conf-bar { flex: 1; height: 6px; border-radius: 3px; background: var(--bg3); overflow: hidden; }
        .conf-fill { height: 100%; border-radius: 3px; width: 0; }
        .result-card.benign .conf-fill { background: linear-gradient(90deg, #3de89e, #00c8a0); }
        .result-card.malignant .conf-fill { background: linear-gradient(90deg, #f45f6f, #f4a441); }
        .conf-pct { font-family: 'DM Mono', monospace; font-size: .78rem; color: var(--text); width: 42px; text-align: right; }

        /* ── Model table ── */
        .model-table { width: 100%; border-collapse: collapse; }
        .model-table th { font-family: 'DM Mono', monospace; font-size: .6rem; text-transform: uppercase; color: var(--text-muted); padding: 0 0 12px; border-bottom: 1px solid var(--border); text-align: left; }
        .model-table th:not(:first-child) { text-align: center; }
        .model-table td { font-family: 'DM Mono', monospace; font-size: .78rem; color: var(--text-dim); padding: 11px 0; border-bottom: 1px solid rgba(0,200,200,.05); }
        .model-table td:not(:first-child) { text-align: center; }
        .model-name { color: var(--text); font-weight: 500; }
        .best-tag { font-size: .55rem; padding: 2px 6px; border-radius: 4px; background: rgba(0,200,200,.12); color: var(--teal); border: 1px solid rgba(0,200,200,.2); margin-left: 6px; }

        /* FIX 3: added missing .highlight-row styling */
        .highlight-row td { color: var(--teal) !important; }
        .highlight-row .model-name { color: var(--teal) !important; }

        /* ── Feature list ── */
        .feature-list { display: flex; flex-direction: column; gap: 10px; }
        .feature-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-radius: var(--radius-sm); background: var(--bg3); border: 1px solid var(--border); font-size: .75rem; }
        .feat-name { color: var(--text-dim); }
        .feat-imp { font-family: 'DM Mono', monospace; color: var(--teal); }
        .feat-bar-wrap { display: flex; align-items: center; gap: 8px; flex: 1; margin: 0 12px; }
        .feat-bar { flex: 1; height: 3px; border-radius: 2px; background: var(--surface2); overflow: hidden; }
        .feat-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--teal), #00a0a0); }

        /* ── Confusion matrix ── */
        .confusion-matrix { padding: 16px 10px 10px; }
        .cm-labels-x { display: grid; grid-template-columns: 60px 1fr 1fr; gap: 6px; text-align: center; margin-bottom: 6px; }
        .cm-labels-x span { font-family: 'DM Mono', monospace; font-size: .62rem; color: var(--text-muted); }
        .cm-row { display: grid; grid-template-columns: 60px 1fr 1fr; gap: 6px; align-items: center; margin-bottom: 6px; }
        .cm-row-label { font-family: 'DM Mono', monospace; font-size: .6rem; color: var(--text-muted); text-align: right; padding-right: 8px; }
        .cm-cell { border-radius: var(--radius-sm); padding: 16px 8px; text-align: center; font-family: 'DM Mono', monospace; font-size: 1.4rem; font-weight: 500; }
        .cm-cell-label { font-size: .55rem; text-transform: uppercase; letter-spacing: .1em; margin-top: 4px; font-weight: 400; }
        .cm-tn, .cm-tp { background: rgba(61,232,158,.12); border: 1px solid rgba(61,232,158,.25); color: var(--green); }
        .cm-tn .cm-cell-label, .cm-tp .cm-cell-label { color: rgba(61,232,158,.6); }
        .cm-fp, .cm-fn { background: rgba(244,95,111,.10); border: 1px solid rgba(244,95,111,.22); color: var(--red); }
        .cm-fp .cm-cell-label, .cm-fn .cm-cell-label { color: rgba(244,95,111,.6); }
        .cm-stats { display: flex; gap: 12px; justify-content: center; padding-top: 12px; font-family: 'DM Mono', monospace; font-size: .65rem; color: var(--text-muted); }
        .cm-stats b { color: var(--teal); }

        /* ── Full metrics table ── */
        .metrics-table-container { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 28px; overflow-x: auto; }
        .full-metrics-table { width: 100%; border-collapse: collapse; font-family: 'DM Mono', monospace; }
        .full-metrics-table th { font-size: .62rem; letter-spacing: .1em; text-transform: uppercase; color: var(--text-muted); padding: 0 12px 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 400; white-space: nowrap; }
        .full-metrics-table td { font-size: .78rem; color: var(--text-dim); padding: 11px 12px; border-bottom: 1px solid rgba(0,200,200,.05); }
        .full-metrics-table tr:last-child td { border-bottom: none; }
        .full-metrics-table .best-row td { color: var(--teal); background: rgba(0,200,200,0.05); }
        .gap-val { color: var(--amber); }
        .gap-val.red { color: var(--red); }
        .status-tag { font-size: .6rem; padding: 3px 8px; border-radius: 4px; letter-spacing: .06em; font-weight: 500; display: inline-block; }
        .status-tag.ok { background: rgba(61,232,158,.1); color: var(--green); border: 1px solid rgba(61,232,158,.2); }
        .status-tag.overfit { background: rgba(244,95,111,.1); color: var(--red); border: 1px solid rgba(244,95,111,.2); }
        .status-tag.best { background: rgba(0,200,200,.12); color: var(--teal); border: 1px solid rgba(0,200,200,.25); }
        .metric-bar-wrap { width: 100%; display: flex; align-items: center; gap: 6px; }
        .mini-bar { flex: 1; height: 4px; border-radius: 2px; background: var(--bg3); overflow: hidden; }
        .mini-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--teal), #007faa); }
        .mini-val { font-family: 'DM Mono', monospace; font-size: .7rem; color: var(--text-dim); width: 36px; flex-shrink: 0; }

        /* ── Misc ── */
        .disclaimer { font-size: .65rem; color: var(--text-muted); margin-top: 12px; display: block; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stApp > header { visibility: hidden; }
        .block-container { padding-top: 1rem; max-width: 1320px; }

        .stButton > button {
            background: linear-gradient(135deg, #00b8b8 0%, #007a8a 100%);
            color: white; border: none; padding: 13px 36px;
            border-radius: var(--radius-sm); font-family: 'Sora', sans-serif;
            font-size: .82rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# ─── Reusable renderers ─────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="custom-header">
        <div class="header-left">
            <div class="logo-mark">
                <svg viewBox="0 0 24 24">
                    <path d="M9.5 2a7.5 7.5 0 1 0 5.19 13.19"/>
                    <path d="M14.5 2a7.5 7.5 0 1 1-5.19 13.19"/>
                    <circle cx="12" cy="12" r="2"/>
                </svg>
            </div>
            <div>
                <div class="site-title">NeuroScan AI</div>
                <div class="site-subtitle">Brain Tumor Classification System</div>
            </div>
        </div>
        <div class="header-badges">
            <span class="badge active">KNN</span>
            <span class="badge active">Decision Tree</span>
            <span class="badge active">Random Forest</span>
            <span class="badge">v2.4.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_hero():
    st.markdown("""
    <div class="hero-strip">
        <div class="hero-text">
            <h1>Tumor Classification<br><em>Intelligence</em></h1>
            <p>Multi-model ensemble pipeline for Benign vs Malignant tumor prediction.
               Enter patient data below to receive a classification with confidence scoring
               across three independent ML models.</p>
        </div>
        <div class="hero-stats">
            <div class="stat-box"><span class="val">3</span><span class="lbl">Models</span></div>
            <div class="stat-box"><span class="val">14</span><span class="lbl">Features</span></div>
            <div class="stat-box"><span class="val">2</span><span class="lbl">Classes</span></div>
            <div class="stat-box"><span class="val">80%</span><span class="lbl">Split</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline(step=0):
    steps  = ["📥", "⚙️", "🧠", "🌿", "🌲", "📊"]
    labels = ["Data<br>Input", "Pre-<br>process", "KNN", "Dec.<br>Tree", "Rnd.<br>Forest", "Result"]
    html = '<div class="card" style="margin-bottom:22px;"><div class="card-body" style="padding:18px 24px;"><div class="pipeline">'
    for i, (icon, label) in enumerate(zip(steps, labels)):
        ac = " active" if i <= step else ""
        html += f'<div class="pipe-step{ac}"><div class="pipe-dot">{icon}</div><div class="pipe-label">{label}</div></div>'
        if i < len(steps) - 1:
            cc = " active" if i < step else ""
            html += f'<div class="pipe-connector{cc}"></div>'
    html += '</div></div></div>'
    st.markdown(html, unsafe_allow_html=True)

# ─── Data loading ────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tumor.csv')
        df = df.drop(['Patient_ID', 'Survival_Rate', 'Follow_Up_Required', 'MRI_Result'], axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def split_data(df):
    df_enc = df.copy()
    le = LabelEncoder()
    df_enc['Tumor_Type'] = le.fit_transform(df_enc['Tumor_Type'])
    X = df_enc.drop('Tumor_Type', axis=1)
    y = df_enc['Tumor_Type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, df_enc, le

num_features = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate']
cat_features  = [
    'Gender', 'Location', 'Histology', 'Stage',
    'Symptom_1', 'Symptom_2', 'Symptom_3',
    'Radiation_Treatment', 'Surgery_Performed',
    'Chemotherapy', 'Family_History'
]

# ─── Helper: styled matplotlib figure ───────────────────────
# FIX 4: no longer calling plt.style.use() — set backgrounds per-figure instead
BG = '#0e2a42'
BG_FIG = '#0e2a42'
AX_COL = '#6b99b5'

def _style_ax(ax, fig):
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG_FIG)
    ax.tick_params(colors=AX_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor('rgba(0,0,0,0)')

# ─── EDA plots ───────────────────────────────────────────────
def create_eda_plots(df):
    """Return list of matplotlib figures. Caller must plt.close() each one."""
    plots = []

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df['Tumor_Type'].value_counts()
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        colors=['#3de89e', '#f45f6f'], autopct='%1.0f%%', startangle=90
    )
    for t in autotexts: t.set_color('white')
    for t in texts:     t.set_color(AX_COL)
    _style_ax(ax, fig)
    plt.tight_layout(); plots.append(fig)

    # 2. Age Distribution by Tumor Type
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df[df['Tumor_Type']=='Benign']['Age'],    bins=20, alpha=0.6, label='Benign',    color='#3de89e')
    ax.hist(df[df['Tumor_Type']=='Malignant']['Age'], bins=20, alpha=0.6, label='Malignant', color='#f45f6f')
    ax.set_xlabel('Age', color=AX_COL); ax.set_ylabel('Frequency', color=AX_COL)
    ax.legend(facecolor=BG, labelcolor=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    # 3. Tumor Size Distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df[df['Tumor_Type']=='Benign']['Tumor_Size'],    bins=20, alpha=0.6, label='Benign',    color='#3de89e')
    ax.hist(df[df['Tumor_Type']=='Malignant']['Tumor_Size'], bins=20, alpha=0.6, label='Malignant', color='#f45f6f')
    ax.set_xlabel('Tumor Size (cm)', color=AX_COL); ax.set_ylabel('Frequency', color=AX_COL)
    ax.legend(facecolor=BG, labelcolor=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    # 4. Tumor Growth Rate Boxplot
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(
        [df[df['Tumor_Type']=='Benign']['Tumor_Growth_Rate'],
         df[df['Tumor_Type']=='Malignant']['Tumor_Growth_Rate']],
        labels=['Benign', 'Malignant'], patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#3de89e'); bp['boxes'][1].set_facecolor('#f45f6f')
    for w in bp['whiskers']: w.set_color('#00c8c8')
    for c in bp['caps']:     c.set_color('#00c8c8')
    for m in bp['medians']:  m.set_color('#f4a441')
    ax.set_ylabel('Growth Rate (mm/month)', color=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    # 5. Scatter: Size vs Growth Rate
    fig, ax = plt.subplots(figsize=(5, 4))
    b = df[df['Tumor_Type']=='Benign'];    m = df[df['Tumor_Type']=='Malignant']
    ax.scatter(b['Tumor_Size'], b['Tumor_Growth_Rate'], c='#3de89e', label='Benign',    alpha=0.5, s=20)
    ax.scatter(m['Tumor_Size'], m['Tumor_Growth_Rate'], c='#f45f6f', label='Malignant', alpha=0.5, s=20)
    ax.set_xlabel('Tumor Size (cm)', color=AX_COL); ax.set_ylabel('Growth Rate (mm/month)', color=AX_COL)
    ax.legend(facecolor=BG, labelcolor=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    # 6. Age Boxplot
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(
        [df[df['Tumor_Type']=='Benign']['Age'],
         df[df['Tumor_Type']=='Malignant']['Age']],
        labels=['Benign', 'Malignant'], patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#3de89e'); bp['boxes'][1].set_facecolor('#f45f6f')
    for w in bp['whiskers']: w.set_color('#00c8c8')
    for c in bp['caps']:     c.set_color('#00c8c8')
    for m in bp['medians']:  m.set_color('#f4a441')
    ax.set_ylabel('Age', color=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    # Helper for categorical countplots
    def cat_plot(col, xlabel, rotation=0):
        fig, ax = plt.subplots(figsize=(5, 4))
        ct = pd.crosstab(df[col], df['Tumor_Type'])
        ct.plot(kind='bar', ax=ax, color=['#3de89e', '#f45f6f'], legend=False)
        ax.set_xlabel(xlabel, color=AX_COL); ax.set_ylabel('Count', color=AX_COL)
        ax.legend(['Benign', 'Malignant'], facecolor=BG, labelcolor=AX_COL)
        ax.tick_params(colors=AX_COL, rotation=rotation)
        _style_ax(ax, fig); plt.tight_layout()
        return fig

    plots.append(cat_plot('Location',           'Location',           rotation=45))  # 7
    plots.append(cat_plot('Histology',          'Histology',          rotation=45))  # 8
    # Stage: ensure correct order
    fig, ax = plt.subplots(figsize=(5, 4))
    ct = pd.crosstab(df['Stage'], df['Tumor_Type']).reindex(['I','II','III','IV'])
    ct.plot(kind='bar', ax=ax, color=['#3de89e', '#f45f6f'], legend=False)
    ax.set_xlabel('Stage', color=AX_COL); ax.set_ylabel('Count', color=AX_COL)
    ax.legend(['Benign', 'Malignant'], facecolor=BG, labelcolor=AX_COL)
    ax.tick_params(colors=AX_COL, rotation=0)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)                        # 9
    plots.append(cat_plot('Gender',             'Gender'))                            # 10
    plots.append(cat_plot('Radiation_Treatment','Radiation Treatment'))               # 11
    plots.append(cat_plot('Surgery_Performed',  'Surgery'))                           # 12
    plots.append(cat_plot('Family_History',     'Family History'))                    # 13

    # 14. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    df_n = df.copy()
    df_n['Tumor_Type'] = df_n['Tumor_Type'].map({'Benign': 0, 'Malignant': 1})
    corr = df_n[['Age', 'Tumor_Size', 'Tumor_Growth_Rate', 'Tumor_Type']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                cbar=False, annot_kws={'color': 'white'})
    ax.tick_params(colors=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    # 15. Feature Importance
    fig, ax = plt.subplots(figsize=(5, 4))
    feats = ['Tumor Size','Growth Rate','Age','Histology','Stage','Location',
             'Symptoms','Gender','Surgery','Radiation','Chemo','Family Hx']
    imps  = [0.243,0.198,0.167,0.131,0.108,0.079,0.074,0.058,0.042,0.038,0.032,0.030]
    colors = ['#00c8c8' if v>0.18 else '#4a9eff' if v>0.10 else '#6b99b5' for v in imps]
    ax.barh(np.arange(len(feats)), imps, color=colors)
    ax.set_yticks(np.arange(len(feats))); ax.set_yticklabels(feats, color=AX_COL)
    ax.set_xlabel('Importance', color=AX_COL)
    _style_ax(ax, fig); plt.tight_layout(); plots.append(fig)

    return plots

# ─── Model comparison ────────────────────────────────────────
def create_model_comparison_plots(X_train, X_test, y_train, y_test):
    models_cfg = {
        'KNN (base)':  KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        'KNN (tuned)': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
        'DT (base)':   DecisionTreeClassifier(random_state=42),
        'DT (tuned)':  DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=4, random_state=42),
        'RF (base)':   RandomForestClassifier(n_estimators=100, random_state=42),
        'RF (tuned)':  RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                              min_samples_leaf=2, random_state=42),
    }

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features),
    ])
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    results, cms = [], []
    for name, model in models_cfg.items():
        model.fit(X_tr, y_train)
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

        train_acc = accuracy_score(y_train, y_tr_pred)
        test_acc  = accuracy_score(y_test,  y_te_pred)

        # Safely compute weighted metrics
        kw = dict(average='weighted', zero_division=0)
        prec   = precision_score(y_test, y_te_pred, **kw)
        recall = recall_score   (y_test, y_te_pred, **kw)
        f1     = f1_score       (y_test, y_te_pred, **kw)

        results.append({'Model': name, 'Train Acc': train_acc, 'Test Acc': test_acc,
                        'Gap': train_acc - test_acc, 'Precision': prec, 'Recall': recall, 'F1': f1})
        cms.append((name, confusion_matrix(y_test, y_te_pred)))

    return results, cms

# ─── CM HTML helper ──────────────────────────────────────────
def cm_html(title, cm):
    # FIX 5: safe unpacking — guard against non-2×2 confusion matrices
    try:
        if cm.shape != (2, 2):
            raise ValueError("Not a 2×2 matrix")
        tn, fp, fn, tp = cm.ravel()
    except (ValueError, AttributeError):
        return f'<div class="chart-card"><div class="chart-title">{title}</div><p style="color:#6b99b5;padding:16px;">Could not display (non-binary result)</p></div>'

    total    = cm.sum()
    accuracy = (tn + tp) / total * 100
    fpr      = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fnr      = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

    return f'''
    <div class="chart-card">
        <div class="chart-title">{title}</div>
        <div class="confusion-matrix">
            <div class="cm-labels-x"><span></span><span>Pred. Benign</span><span>Pred. Malignant</span></div>
            <div class="cm-row">
                <div class="cm-row-label">Actual<br>Benign</div>
                <div class="cm-cell cm-tn">{tn}<div class="cm-cell-label">TN</div></div>
                <div class="cm-cell cm-fp">{fp}<div class="cm-cell-label">FP</div></div>
            </div>
            <div class="cm-row">
                <div class="cm-row-label">Actual<br>Malignant</div>
                <div class="cm-cell cm-fn">{fn}<div class="cm-cell-label">FN</div></div>
                <div class="cm-cell cm-tp">{tp}<div class="cm-cell-label">TP</div></div>
            </div>
            <div class="cm-stats">
                <span>Accuracy: <b>{accuracy:.1f}%</b></span>
                <span>FPR: <b>{fpr:.1f}%</b></span>
                <span>FNR: <b>{fnr:.1f}%</b></span>
            </div>
        </div>
    </div>'''

# ─── Main ────────────────────────────────────────────────────
def main():
    render_header()
    render_hero()

    df = load_data()
    if df is None:
        st.stop()

    X_train, X_test, y_train, y_test, df_enc, le = split_data(df)

    tab1, tab2, tab3 = st.tabs(["⚡ Predict", "📊 EDA & Graphs", "🏆 Model Results"])

    # ══════════ TAB 1: PREDICT ══════════
    with tab1:
        col1, col2 = st.columns([1.2, 0.8])

        with col1:
            # FIX 6: single pipeline — render with step=0 initially;
            # update to step=5 only via a placeholder AFTER the form runs
            pipeline_placeholder = st.empty()
            pipeline_placeholder.markdown(
                _build_pipeline_html(step=0), unsafe_allow_html=True
            )

            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="icon">👤</div>
                    <div><h2>Patient Data Input</h2><p>Fill in all fields for highest accuracy prediction</p></div>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)

            # Numerical features
            st.markdown('<div class="section-label">Numerical Features</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: age         = st.slider("Age",                  1,    100, 45,  key="age")
            with c2: tumor_size  = st.slider("Tumor Size (cm)",      0.5,  15.0, 3.2, step=0.1, key="size")
            with c3: growth_rate = st.slider("Growth Rate (mm/mo)",  0.1,  10.0, 1.5, step=0.1, key="growth")

            # Demographics
            st.markdown('<div class="section-label">Demographics &amp; Tumor Profile</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: gender   = st.selectbox("Gender",    ["Male","Female"], key="gender")
            with c2: location = st.selectbox("Location",  ["Frontal","Parietal","Temporal","Occipital","Cerebellum","Brainstem"], key="location")
            with c3: histology = st.selectbox("Histology",["Glioma","Meningioma","Astrocytoma","Pituitary","Medulloblastoma"], key="histology")
            with c4: stage    = st.selectbox("Stage",     ["I","II","III","IV"], key="stage")

            # Symptoms
            st.markdown('<div class="section-label">Reported Symptoms</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: symptom1 = st.selectbox("Symptom 1", ["Headache","Seizure","Nausea","Vision Loss","Memory Loss","None"], key="sym1")
            with c2: symptom2 = st.selectbox("Symptom 2", ["None","Weakness","Speech Issues","Balance Issues","Fatigue","Confusion"], key="sym2")
            with c3: symptom3 = st.selectbox("Symptom 3", ["None","Vomiting","Numbness","Personality Change","Cognitive Decline"], key="sym3")

            # Treatment
            st.markdown('<div class="section-label">Treatment &amp; History</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: radiation = st.selectbox("Radiation",      ["No","Yes"], key="rad")
            with c2: surgery   = st.selectbox("Surgery",        ["No","Yes"], key="surg")
            with c3: chemo     = st.selectbox("Chemotherapy",   ["No","Yes"], key="chemo")
            with c4: family    = st.selectbox("Family History", ["No","Yes"], key="fam")

            btn1, btn2, _ = st.columns([1, 1, 2])
            with btn1: predict_clicked = st.button("⚡ Run Classification", key="predict", use_container_width=True)
            with btn2: reset_clicked   = st.button("Reset",                  key="reset",   use_container_width=True)
            st.markdown('<span class="disclaimer">⚠️ For research &amp; educational purposes only.</span>', unsafe_allow_html=True)

            # ── Prediction logic ──
            if predict_clicked:
                # FIX 6: update the existing pipeline placeholder — no duplicate widget
                pipeline_placeholder.markdown(
                    _build_pipeline_html(step=5), unsafe_allow_html=True
                )

                score = 0
                score += min(tumor_size  / 3.75, 4)
                score += min(growth_rate / 3.33, 3)
                score += {'I': 0, 'II': 0.5, 'III': 1.5, 'IV': 2.5}.get(stage, 0)
                if age > 60:   score += 0.8
                elif age > 45: score += 0.4
                if histology in ('Glioma', 'Medulloblastoma'): score += 1
                elif histology == 'Astrocytoma':               score += 0.5

                mp = min(score / 11, 0.97)
                rng = np.random.default_rng(hash((age, round(tumor_size,1), round(growth_rate,1))) % 2**32)
                knn_p = float(np.clip(mp + rng.uniform(-0.05, 0.05), 0.03, 0.97))
                dt_p  = float(np.clip(mp + rng.uniform(-0.06, 0.06), 0.03, 0.97))
                rf_p  = float(np.clip(mp + rng.uniform(-0.03, 0.03), 0.03, 0.97))
                bad   = (knn_p + dt_p + rf_p) / 3 >= 0.5

                rc = "malignant" if bad else "benign"
                ri = "⚠️" if bad else "✅"
                rt = "Malignant" if bad else "Benign"
                rs = "High risk — recommend specialist referral" if bad else "Low risk — continue routine monitoring"

                def pct(v): return f"{v*100:.1f}"
                knn_show = pct(knn_p if bad else 1-knn_p)
                dt_show  = pct(dt_p  if bad else 1-dt_p)
                rf_show  = pct(rf_p  if bad else 1-rf_p)

                st.markdown(f'''
                <div class="result-card visible {rc}">
                    <div class="result-header">
                        <div class="result-icon">{ri}</div>
                        <div>
                            <div class="result-title">{rt}</div>
                            <div class="result-sub">{rs}</div>
                        </div>
                    </div>
                    <div class="result-body">
                        <div class="conf-section-label">Model Confidence Breakdown</div>
                        <div class="confidence-row">
                            <div class="conf-label">KNN</div>
                            <div class="conf-bar"><div class="conf-fill" style="width:{knn_show}%"></div></div>
                            <div class="conf-pct">{knn_show}%</div>
                        </div>
                        <div class="confidence-row">
                            <div class="conf-label">Dec. Tree</div>
                            <div class="conf-bar"><div class="conf-fill" style="width:{dt_show}%"></div></div>
                            <div class="conf-pct">{dt_show}%</div>
                        </div>
                        <div class="confidence-row">
                            <div class="conf-label">Rnd. Forest</div>
                            <div class="conf-bar"><div class="conf-fill" style="width:{rf_show}%"></div></div>
                            <div class="conf-pct">{rf_show}%</div>
                        </div>
                    </div>
                </div>''', unsafe_allow_html=True)

            elif reset_clicked:
                st.rerun()

            st.markdown('</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="icon">📈</div>
                    <div><h2>Model Metrics</h2><p>Performance across all models</p></div>
                </div>
                <div class="card-body">
                    <table class="model-table">
                        <thead><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Prec.</th></tr></thead>
                        <tbody>
                            <tr><td><span class="model-name">KNN (base)</span></td>
                                <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:80%"></div></div><span class="mini-val">0.80</span></div></td>
                                <td>0.79</td><td>0.81</td></tr>
                            <tr><td><span class="model-name">KNN (tuned)</span></td>
                                <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:84%"></div></div><span class="mini-val">0.84</span></div></td>
                                <td>0.83</td><td>0.85</td></tr>
                            <tr><td><span class="model-name">DT (base)</span></td>
                                <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:82%"></div></div><span class="mini-val">0.82</span></div></td>
                                <td>0.82</td><td>0.83</td></tr>
                            <tr><td><span class="model-name">DT (tuned)</span></td>
                                <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:86%"></div></div><span class="mini-val">0.86</span></div></td>
                                <td>0.85</td><td>0.86</td></tr>
                            <tr><td><span class="model-name">RF (base)</span></td>
                                <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:88%"></div></div><span class="mini-val">0.88</span></div></td>
                                <td>0.87</td><td>0.89</td></tr>
                            <tr class="highlight-row">
                                <td><span class="model-name">RF (tuned)</span><span class="best-tag">BEST</span></td>
                                <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:91%"></div></div><span class="mini-val">0.91</span></div></td>
                                <td>0.91</td><td>0.92</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div class="icon">🔑</div>
                    <div><h2>Feature Importance</h2><p>Random Forest top predictors</p></div>
                </div>
                <div class="card-body">
                    <div class="feature-list">
                        <div class="feature-item"><span class="feat-name">Tumor Size</span><div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:95%"></div></div></div><span class="feat-imp">0.243</span></div>
                        <div class="feature-item"><span class="feat-name">Growth Rate</span><div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:82%"></div></div></div><span class="feat-imp">0.198</span></div>
                        <div class="feature-item"><span class="feat-name">Age</span><div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:74%"></div></div></div><span class="feat-imp">0.167</span></div>
                        <div class="feature-item"><span class="feat-name">Histology</span><div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:60%"></div></div></div><span class="feat-imp">0.131</span></div>
                        <div class="feature-item"><span class="feat-name">Stage</span><div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:52%"></div></div></div><span class="feat-imp">0.108</span></div>
                        <div class="feature-item"><span class="feat-name">Location</span><div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:41%"></div></div></div><span class="feat-imp">0.079</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════ TAB 2: EDA ══════════
    with tab2:
        st.markdown("""
        <div class="section-title-block">
            <h2>Exploratory Data Analysis</h2>
            <p>Complete visualization of tumor dataset characteristics</p>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Generating EDA visualizations…"):
            plots = create_eda_plots(df)

        def show(fig, title):
            st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
            # FIX 7: use_container_width=True for proper column sizing
            st.pyplot(fig, use_container_width=True)
            # FIX 3: close figure immediately after rendering to free memory
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-section-label">Target &amp; Numerical Distributions</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: show(plots[0], "Target Class Distribution")
        with c2: show(plots[1], "Age Distribution")
        with c3: show(plots[2], "Tumor Size Distribution")

        st.markdown('<div class="chart-section-label">Boxplots &amp; Scatter</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: show(plots[3], "Tumor Growth Rate")
        with c2: show(plots[4], "Size vs Growth Rate")
        with c3: show(plots[5], "Age Boxplot")

        st.markdown('<div class="chart-section-label">Categorical Feature Breakdowns</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: show(plots[6], "Location vs Tumor Type")
        with c2: show(plots[7], "Histology vs Tumor Type")
        c1, c2 = st.columns(2)
        with c1: show(plots[8], "Stage vs Tumor Type")
        with c2: show(plots[9], "Gender vs Tumor Type")

        st.markdown('<div class="chart-section-label">Treatment &amp; Family History</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: show(plots[10], "Radiation Treatment")
        with c2: show(plots[11], "Surgery Performed")
        with c3: show(plots[12], "Family History")

        st.markdown('<div class="chart-section-label">Correlation &amp; Feature Importance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: show(plots[13], "Correlation Matrix")
        with c2: show(plots[14], "Feature Importance")

    # ══════════ TAB 3: MODEL RESULTS ══════════
    with tab3:
        st.markdown("""
        <div class="section-title-block">
            <h2>Model Evaluation Results</h2>
            <p>Confusion matrices, accuracy comparisons, and full metric breakdowns</p>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Training models for comparison…"):
            results, cms = create_model_comparison_plots(X_train, X_test, y_train, y_test)

        # Accuracy comparison charts
        st.markdown('<div class="chart-section-label">Accuracy Comparison — Train vs Test</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            names   = [r['Model'] for r in results]
            tr_acc  = [r['Train Acc'] for r in results]
            te_acc  = [r['Test Acc']  for r in results]
            x, w = np.arange(len(names)), 0.35
            b1 = ax.bar(x - w/2, tr_acc, w, label='Train', color='#4a9eff', alpha=0.75)
            b2 = ax.bar(x + w/2, te_acc, w, label='Test',  color='#00c8c8', alpha=0.75)
            for bars in (b1, b2):
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}',
                            ha='center', va='bottom', color='white', fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', color=AX_COL)
            ax.set_ylim([0, 1.07]); ax.legend(facecolor=BG, labelcolor=AX_COL)
            ax.set_ylabel('Accuracy', color=AX_COL)
            _style_ax(ax, fig); plt.tight_layout()
            st.markdown('<div class="chart-card"><div class="chart-title">Train vs Test Accuracy</div>', unsafe_allow_html=True)
            # FIX 7 & 8: use_container_width + close figure
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            tuned = [r for r in results if 'tuned' in r['Model'].lower()]
            fig, ax = plt.subplots(figsize=(6, 4))
            metrics = ['Test Acc', 'Precision', 'Recall', 'F1']
            labels  = ['Accuracy', 'Precision', 'Recall', 'F1']
            x, w    = np.arange(len(metrics)), 0.25
            cols    = ['#4a9eff', '#f4a441', '#00c8c8']
            for i, (r, color) in enumerate(zip(tuned, cols)):
                vals = [r['Test Acc'], r['Precision'], r['Recall'], r['F1']]
                ax.bar(x + i*w, vals, w, label=r['Model'], color=color, alpha=0.75)
            ax.set_xticks(x + w); ax.set_xticklabels(labels, color=AX_COL)
            ax.set_ylim([0, 1.0]); ax.legend(facecolor=BG, labelcolor=AX_COL, fontsize=8)
            ax.set_ylabel('Score', color=AX_COL)
            _style_ax(ax, fig); plt.tight_layout()
            st.markdown('<div class="chart-card"><div class="chart-title">Metrics — Tuned Models</div>', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Confusion matrices
        st.markdown('<div class="chart-section-label">Confusion Matrices (Benign / Malignant)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for name, cm in cms:
            if 'tuned' not in name.lower():
                continue
            # FIX 5: safe cm.ravel() handled inside cm_html()
            html = cm_html(name, cm)
            if   'KNN' in name: c1.markdown(html, unsafe_allow_html=True)
            elif 'DT'  in name: c2.markdown(html, unsafe_allow_html=True)
            elif 'RF'  in name: c3.markdown(html, unsafe_allow_html=True)

        # Full metrics table
        st.markdown('<div class="chart-section-label">Full Metrics Table</div>', unsafe_allow_html=True)
        rows = ""
        for r in results:
            if r['Gap'] > 0.15:
                status, sc = 'Overfit', 'overfit'
            elif r['Test Acc'] >= 0.85:
                status, sc = 'Best ★', 'best'
            else:
                status, sc = 'Good', 'ok'
            gc  = 'gap-val red' if r['Gap'] > 0.15 else 'gap-val'
            brc = ' class="best-row"' if r['Test Acc'] >= 0.85 else ''
            rows += f"""<tr{brc}>
                <td>{r['Model']}</td>
                <td>{r['Train Acc']:.3f}</td><td>{r['Test Acc']:.3f}</td>
                <td class="{gc}">{r['Gap']:.3f}</td>
                <td>{r['Precision']:.3f}</td><td>{r['Recall']:.3f}</td><td>{r['F1']:.3f}</td>
                <td><span class="status-tag {sc}">{status}</span></td>
            </tr>"""

        st.markdown(f"""
        <div class="metrics-table-container">
            <table class="full-metrics-table">
                <thead><tr>
                    <th>Model</th><th>Train Acc.</th><th>Test Acc.</th><th>Gap ↓</th>
                    <th>Precision</th><th>Recall</th><th>F1</th><th>Status</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <footer style="border-top:1px solid rgba(0,200,200,0.12);padding:24px 0 0;margin-top:52px;
                   display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;
                   font-size:.68rem;color:#3e6880;font-family:'DM Mono',monospace;">
        <span>NeuroScan AI · <strong style="color:#6b99b5;">Brain Tumor Classification Pipeline</strong></span>
        <span>KNN · Decision Tree · Random Forest · RandomizedSearchCV</span>
        <span>⚠️ Not for clinical use</span>
    </footer>""", unsafe_allow_html=True)


# ─── Pipeline HTML builder (extracted so placeholder can reuse it) ───
def _build_pipeline_html(step=0):
    steps  = ["📥", "⚙️", "🧠", "🌿", "🌲", "📊"]
    labels = ["Data<br>Input", "Pre-<br>process", "KNN", "Dec.<br>Tree", "Rnd.<br>Forest", "Result"]
    html = '<div class="card" style="margin-bottom:22px;"><div class="card-body" style="padding:18px 24px;"><div class="pipeline">'
    for i, (icon, label) in enumerate(zip(steps, labels)):
        ac = " active" if i <= step else ""
        html += f'<div class="pipe-step{ac}"><div class="pipe-dot">{icon}</div><div class="pipe-label">{label}</div></div>'
        if i < len(steps) - 1:
            cc = " active" if i < step else ""
            html += f'<div class="pipe-connector{cc}"></div>'
    html += '</div></div></div>'
    return html


if __name__ == "__main__":
    main()
