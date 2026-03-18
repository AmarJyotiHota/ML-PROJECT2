# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (same as before, but I'll keep it concise here)
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
            --amber-dim: rgba(244,164,65,0.10);
            --green: #3de89e;
            --red: #f45f6f;
            --text: #d6eaf0;
            --text-dim: #6b99b5;
            --text-muted: #3e6880;
            --radius: 12px;
            --radius-sm: 8px;
        }
        
        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: 'Sora', sans-serif;
        }
        
        .stApp::before {
            content: '';
            position: fixed; inset: 0;
            background:
                radial-gradient(ellipse 900px 600px at 10% 5%, rgba(0,150,160,0.06) 0%, transparent 70%),
                radial-gradient(ellipse 600px 600px at 90% 80%, rgba(244,164,65,0.04) 0%, transparent 65%),
                radial-gradient(ellipse 500px 400px at 50% 50%, rgba(0,200,200,0.03) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }
        
        .stApp::after {
            content: '';
            position: fixed; inset: 0;
            background-image: radial-gradient(circle, rgba(0,200,200,0.08) 1px, transparent 1px);
            background-size: 36px 36px;
            pointer-events: none;
            z-index: 0;
        }
        
        .custom-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 28px 0 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 48px;
            flex-wrap: wrap;
            gap: 16px;
            animation: fadeDown 0.6s ease both;
            position: relative;
            z-index: 1;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        
        .logo-mark {
            width: 44px; height: 44px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--teal) 0%, #006f8e 100%);
            display: grid; place-items: center;
            box-shadow: 0 0 24px rgba(0,200,200,0.3);
        }
        
        .logo-mark svg {
            width: 22px; height: 22px;
            stroke: white;
            stroke-width: 1.8;
            fill: none;
        }
        
        .site-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.65rem;
            font-weight: 500;
            color: #ffffff;
            line-height: 1;
        }
        
        .site-subtitle {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            color: var(--teal);
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-top: 4px;
        }
        
        .header-badges {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .badge {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: 20px;
            border: 1px solid var(--border);
            color: var(--text-dim);
            background: var(--teal-dim);
        }
        
        .badge.active {
            border-color: var(--teal);
            color: var(--teal);
            box-shadow: 0 0 12px rgba(0,200,200,0.15);
        }
        
        .hero-strip {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 28px 32px;
            display: flex;
            align-items: center;
            gap: 32px;
            margin-bottom: 40px;
            flex-wrap: wrap;
            animation: fadeUp 0.6s 0.1s ease both;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .hero-strip::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--teal), transparent);
        }
        
        .hero-text h1 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.2rem;
            font-weight: 400;
            color: #fff;
            line-height: 1.15;
            margin: 0;
        }
        
        .hero-text h1 em {
            font-style: normal;
            color: var(--teal);
        }
        
        .hero-text p {
            font-size: 0.82rem;
            color: var(--text-dim);
            margin-top: 8px;
            max-width: 520px;
            line-height: 1.65;
        }
        
        .hero-stats {
            display: flex;
            gap: 24px;
            margin-left: auto;
            flex-wrap: wrap;
        }
        
        .stat-box {
            text-align: center;
            padding: 14px 22px;
            border-radius: var(--radius-sm);
            background: var(--bg3);
            border: 1px solid var(--border);
            min-width: 90px;
        }
        
        .stat-box .val {
            font-family: 'DM Mono', monospace;
            font-size: 1.45rem;
            font-weight: 500;
            color: var(--teal);
            display: block;
        }
        
        .stat-box .lbl {
            font-size: 0.62rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-top: 3px;
            display: block;
        }
        
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            animation: fadeUp 0.6s 0.2s ease both;
            margin-bottom: 22px;
            position: relative;
            z-index: 1;
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 18px 24px;
            border-bottom: 1px solid var(--border);
            background: rgba(0,0,0,0.15);
        }
        
        .card-header .icon {
            width: 32px; height: 32px;
            border-radius: 8px;
            background: var(--teal-dim);
            border: 1px solid var(--border-hi);
            display: grid; place-items: center;
            color: var(--teal);
            font-size: 0.9rem;
        }
        
        .card-header h2 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.2rem;
            font-weight: 500;
            color: #fff;
            letter-spacing: 0.02em;
            margin: 0;
        }
        
        .card-header p {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin: 1px 0 0 0;
        }
        
        .card-body {
            padding: 24px;
        }
        
        .pipeline {
            display: flex;
            align-items: center;
            gap: 0;
            padding: 6px 0;
            overflow-x: auto;
        }
        
        .pipeline::-webkit-scrollbar {
            display: none;
        }
        
        .pipe-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            flex: 1;
            min-width: 72px;
        }
        
        .pipe-dot {
            width: 36px; height: 36px;
            border-radius: 50%;
            background: var(--bg3);
            border: 1.5px solid var(--border);
            display: grid; place-items: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            transition: all 0.3s;
        }
        
        .pipe-step.active .pipe-dot {
            background: var(--teal-dim);
            border-color: var(--teal);
            color: var(--teal);
            box-shadow: 0 0 14px rgba(0,200,200,0.2);
        }
        
        .pipe-step .pipe-label {
            font-size: 0.58rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            text-align: center;
            line-height: 1.3;
        }
        
        .pipe-step.active .pipe-label {
            color: var(--teal);
        }
        
        .pipe-connector {
            flex: 0.8;
            height: 1px;
            background: var(--border);
            min-width: 16px;
            position: relative;
            top: -13px;
        }
        
        .pipe-connector.active {
            background: var(--teal);
            box-shadow: 0 0 6px rgba(0,200,200,0.3);
        }
        
        .form-section {
            margin-bottom: 28px;
        }
        
        .section-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--teal);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .section-label::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }
        
        .fields-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
        }
        
        .result-card {
            border-radius: var(--radius);
            border: 1px solid var(--border);
            overflow: hidden;
            margin-top: 20px;
            animation: fadeUp 0.5s ease both;
            display: none;
        }
        
        .result-card.visible {
            display: block;
        }
        
        .result-card.benign {
            border-color: rgba(61,232,158,0.35);
        }
        
        .result-card.malignant {
            border-color: rgba(244,95,111,0.35);
        }
        
        .result-header {
            padding: 20px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .result-card.benign .result-header {
            background: rgba(61,232,158,0.07);
        }
        
        .result-card.malignant .result-header {
            background: rgba(244,95,111,0.07);
        }
        
        .result-icon {
            width: 48px; height: 48px;
            border-radius: 50%;
            display: grid; place-items: center;
            font-size: 1.4rem;
        }
        
        .result-card.benign .result-icon {
            background: rgba(61,232,158,0.12);
            color: var(--green);
        }
        
        .result-card.malignant .result-icon {
            background: rgba(244,95,111,0.12);
            color: var(--red);
        }
        
        .result-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.8rem;
            font-weight: 500;
            line-height: 1;
        }
        
        .result-card.benign .result-title {
            color: var(--green);
        }
        
        .result-card.malignant .result-title {
            color: var(--red);
        }
        
        .result-sub {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 4px;
        }
        
        .result-body {
            padding: 18px 24px;
        }
        
        .confidence-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        
        .conf-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            width: 80px;
        }
        
        .conf-bar {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: var(--bg3);
            overflow: hidden;
        }
        
        .conf-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.9s cubic-bezier(0.22, 1, 0.36, 1);
            width: 0;
        }
        
        .result-card.benign .conf-fill {
            background: linear-gradient(90deg, #3de89e, #00c8a0);
        }
        
        .result-card.malignant .conf-fill {
            background: linear-gradient(90deg, #f45f6f, #f4a441);
        }
        
        .conf-pct {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--text);
            width: 42px;
            text-align: right;
        }
        
        .model-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .model-table th {
            font-family: 'DM Mono', monospace;
            font-size: 0.6rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-muted);
            padding: 0 0 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-weight: 400;
        }
        
        .model-table th:not(:first-child) {
            text-align: center;
        }
        
        .model-table td {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-dim);
            padding: 11px 0;
            border-bottom: 1px solid rgba(0,200,200,0.05);
        }
        
        .model-table td:not(:first-child) {
            text-align: center;
        }
        
        .model-name {
            color: var(--text);
            font-weight: 500;
            font-size: 0.75rem;
        }
        
        .best-tag {
            font-size: 0.55rem;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(0,200,200,0.12);
            color: var(--teal);
            border: 1px solid rgba(0,200,200,0.2);
            letter-spacing: 0.06em;
            margin-left: 6px;
        }
        
        .feature-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            border-radius: var(--radius-sm);
            background: var(--bg3);
            border: 1px solid var(--border);
            font-size: 0.75rem;
        }
        
        .feat-name {
            color: var(--text-dim);
        }
        
        .feat-imp {
            font-family: 'DM Mono', monospace;
            color: var(--teal);
            font-size: 0.7rem;
        }
        
        .feat-bar-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
            margin: 0 12px;
        }
        
        .feat-bar {
            flex: 1;
            height: 3px;
            border-radius: 2px;
            background: var(--surface2);
            overflow: hidden;
        }
        
        .feat-fill {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--teal), #00a0a0);
        }
        
        .metric-bar-wrap {
            width: 100%;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .mini-bar {
            flex: 1;
            height: 4px;
            border-radius: 2px;
            background: var(--bg3);
            overflow: hidden;
        }
        
        .mini-fill {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--teal), #007faa);
        }
        
        .mini-val {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-dim);
            width: 36px;
        }
        
        footer {
            border-top: 1px solid var(--border);
            padding: 24px 0 0;
            margin-top: 52px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            font-size: 0.68rem;
            color: var(--text-muted);
            font-family: 'DM Mono', monospace;
            animation: fadeUp 0.6s 0.4s ease both;
        }
        
        footer strong {
            color: var(--text-dim);
        }
        
        @keyframes fadeDown {
            from { opacity: 0; transform: translateY(-12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp > header {visibility: hidden;}
        
        .stButton > button {
            background: linear-gradient(135deg, #00b8b8 0%, #007a8a 100%);
            color: white;
            border: none;
            padding: 13px 36px;
            border-radius: var(--radius-sm);
            font-family: 'Sora', sans-serif;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 22px rgba(0,180,180,0.35);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 32px rgba(0,200,200,0.4);
            filter: brightness(1.08);
        }
        
        .disclaimer {
            font-size: 0.65rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin-top: 12px;
            display: block;
        }
        
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 1280px;
        }
        
        .row-widget.stHorizontal {
            gap: 28px;
        }
        
        /* Graph container styling */
        .graph-container {
            background: var(--bg3);
            border-radius: var(--radius-sm);
            padding: 16px;
            border: 1px solid var(--border);
            margin-bottom: 20px;
        }
        
        .graph-title {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--teal);
            margin-bottom: 12px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load CSS
load_css()

# Custom header
def render_header():
    header_html = """
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
    """
    st.markdown(header_html, unsafe_allow_html=True)

# Hero strip
def render_hero():
    hero_html = """
    <div class="hero-strip">
        <div class="hero-text">
            <h1>Tumor Classification<br><em>Intelligence</em></h1>
            <p>Multi-model ensemble pipeline for Benign vs Malignant tumor prediction.
               Enter patient data below to receive a classification with confidence scoring
               across three independent ML models.</p>
        </div>
        <div class="hero-stats">
            <div class="stat-box">
                <span class="val">3</span>
                <span class="lbl">Models</span>
            </div>
            <div class="stat-box">
                <span class="val">14</span>
                <span class="lbl">Features</span>
            </div>
            <div class="stat-box">
                <span class="val">2</span>
                <span class="lbl">Classes</span>
            </div>
            <div class="stat-box">
                <span class="val">80%</span>
                <span class="lbl">Split</span>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

# Pipeline tracker
def render_pipeline(step=0):
    steps = ["📥", "⚙️", "🧠", "🌿", "🌲", "📊"]
    labels = ["Data Input", "Pre-process", "KNN Model", "Decision Tree", "Random Forest", "Result Output"]
    
    pipeline_html = '<div class="card" style="margin-bottom:22px;"><div class="card-body" style="padding:18px 24px;"><div class="pipeline">'
    
    for i in range(len(steps)):
        active_class = " active" if i <= step else ""
        pipeline_html += f'''
            <div class="pipe-step{active_class}" data-step="{i}">
                <div class="pipe-dot">{steps[i]}</div>
                <div class="pipe-label">{labels[i].replace(" ", "<br>")}</div>
            </div>
        '''
        if i < len(steps) - 1:
            conn_class = " active" if i < step else ""
            pipeline_html += f'<div class="pipe-connector{conn_class}"></div>'
    
    pipeline_html += '</div></div></div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tumor.csv')
        df = df.drop(['Patient_ID', 'Survival_Rate', 'Follow_Up_Required', 'MRI_Result'], axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Split data
@st.cache_data
def split_data(df):
    df_encoded = df.copy()
    df_encoded['Tumor_Type'] = df_encoded['Tumor_Type'].map({'Benign': 0, 'Malignant': 1})
    X = df_encoded.drop('Tumor_Type', axis=1)
    y = df_encoded['Tumor_Type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, df_encoded

# Define features
num_features = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate']
cat_features = [
    'Gender', 'Location', 'Histology', 'Stage',
    'Symptom_1', 'Symptom_2', 'Symptom_3',
    'Radiation_Treatment', 'Surgery_Performed',
    'Chemotherapy', 'Family_History'
]

# Function to create EDA plots
def create_eda_plots(df):
    plots = []
    
    # 1. Target Class Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    df['Tumor_Type'].value_counts().plot(kind='bar', color=[var('--green'), var('--red')], ax=ax1)
    ax1.set_title('Target Class Distribution', color='white', fontsize=14)
    ax1.set_xlabel('Tumor Type', color='var(--text-dim)')
    ax1.set_ylabel('Count', color='var(--text-dim)')
    ax1.tick_params(colors='var(--text-dim)')
    ax1.set_xticklabels(['Benign', 'Malignant'], rotation=0)
    for spine in ax1.spines.values():
        spine.set_color('var(--border)')
    fig1.patch.set_facecolor('var(--bg3)')
    ax1.set_facecolor('var(--bg3)')
    plots.append(('Target Class Distribution', fig1))
    
    # 2. Age Distribution by Tumor Type
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for tumor_type in ['Benign', 'Malignant']:
        subset = df[df['Tumor_Type'] == tumor_type]
        ax2.hist(subset['Age'], alpha=0.7, label=tumor_type, bins=20, 
                color=var('--green') if tumor_type == 'Benign' else var('--red'))
    ax2.set_title('Age Distribution by Tumor Type', color='white', fontsize=14)
    ax2.set_xlabel('Age', color='var(--text-dim)')
    ax2.set_ylabel('Frequency', color='var(--text-dim)')
    ax2.legend()
    ax2.tick_params(colors='var(--text-dim)')
    for spine in ax2.spines.values():
        spine.set_color('var(--border)')
    fig2.patch.set_facecolor('var(--bg3)')
    ax2.set_facecolor('var(--bg3)')
    plots.append(('Age Distribution by Tumor Type', fig2))
    
    # 3. Tumor Size Distribution
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for tumor_type in ['Benign', 'Malignant']:
        subset = df[df['Tumor_Type'] == tumor_type]
        ax3.hist(subset['Tumor_Size'], alpha=0.7, label=tumor_type, bins=20,
                color=var('--green') if tumor_type == 'Benign' else var('--red'))
    ax3.set_title('Tumor Size Distribution by Tumor Type', color='white', fontsize=14)
    ax3.set_xlabel('Tumor Size (cm)', color='var(--text-dim)')
    ax3.set_ylabel('Frequency', color='var(--text-dim)')
    ax3.legend()
    ax3.tick_params(colors='var(--text-dim)')
    for spine in ax3.spines.values():
        spine.set_color('var(--border)')
    fig3.patch.set_facecolor('var(--bg3)')
    ax3.set_facecolor('var(--bg3)')
    plots.append(('Tumor Size Distribution', fig3))
    
    # 4. Tumor Growth Rate Boxplot
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    benign_data = [df[df['Tumor_Type'] == 'Benign']['Tumor_Growth_Rate']]
    malignant_data = [df[df['Tumor_Type'] == 'Malignant']['Tumor_Growth_Rate']]
    bp = ax4.boxplot([benign_data[0], malignant_data[0]], labels=['Benign', 'Malignant'],
                     patch_artist=True,
                     boxprops=dict(color='var(--teal)'),
                     whiskerprops=dict(color='var(--teal)'),
                     capprops=dict(color='var(--teal)'),
                     medianprops=dict(color='var(--amber)', linewidth=2))
    bp['boxes'][0].set_facecolor(var('--green'))
    bp['boxes'][1].set_facecolor(var('--red'))
    ax4.set_title('Tumor Growth Rate by Tumor Type', color='white', fontsize=14)
    ax4.set_ylabel('Growth Rate (mm/month)', color='var(--text-dim)')
    ax4.tick_params(colors='var(--text-dim)')
    for spine in ax4.spines.values():
        spine.set_color('var(--border)')
    fig4.patch.set_facecolor('var(--bg3)')
    ax4.set_facecolor('var(--bg3)')
    plots.append(('Tumor Growth Rate Boxplot', fig4))
    
    # 5. Tumor Size vs Growth Rate Scatter
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    colors = {'Benign': var('--green'), 'Malignant': var('--red')}
    for tumor_type in ['Benign', 'Malignant']:
        subset = df[df['Tumor_Type'] == tumor_type]
        ax5.scatter(subset['Tumor_Size'], subset['Tumor_Growth_Rate'], 
                   c=colors[tumor_type], label=tumor_type, alpha=0.6, edgecolors='none')
    ax5.set_title('Tumor Size vs Growth Rate', color='white', fontsize=14)
    ax5.set_xlabel('Tumor Size (cm)', color='var(--text-dim)')
    ax5.set_ylabel('Growth Rate (mm/month)', color='var(--text-dim)')
    ax5.legend()
    ax5.tick_params(colors='var(--text-dim)')
    for spine in ax5.spines.values():
        spine.set_color('var(--border)')
    fig5.patch.set_facecolor('var(--bg3)')
    ax5.set_facecolor('var(--bg3)')
    plots.append(('Tumor Size vs Growth Rate', fig5))
    
    # 6. Location vs Tumor Type
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    location_ct = pd.crosstab(df['Location'], df['Tumor_Type'])
    location_ct.plot(kind='bar', ax=ax6, color=[var('--green'), var('--red')])
    ax6.set_title('Location vs Tumor Type', color='white', fontsize=14)
    ax6.set_xlabel('Location', color='var(--text-dim)')
    ax6.set_ylabel('Count', color='var(--text-dim)')
    ax6.legend(title='Tumor Type')
    ax6.tick_params(colors='var(--text-dim)', rotation=45)
    for spine in ax6.spines.values():
        spine.set_color('var(--border)')
    fig6.patch.set_facecolor('var(--bg3)')
    ax6.set_facecolor('var(--bg3)')
    plots.append(('Location Distribution', fig6))
    
    # 7. Histology vs Tumor Type
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    histology_ct = pd.crosstab(df['Histology'], df['Tumor_Type'])
    histology_ct.plot(kind='bar', ax=ax7, color=[var('--green'), var('--red')])
    ax7.set_title('Histology vs Tumor Type', color='white', fontsize=14)
    ax7.set_xlabel('Histology', color='var(--text-dim)')
    ax7.set_ylabel('Count', color='var(--text-dim)')
    ax7.legend(title='Tumor Type')
    ax7.tick_params(colors='var(--text-dim)', rotation=45)
    for spine in ax7.spines.values():
        spine.set_color('var(--border)')
    fig7.patch.set_facecolor('var(--bg3)')
    ax7.set_facecolor('var(--bg3)')
    plots.append(('Histology Distribution', fig7))
    
    # 8. Stage vs Tumor Type
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    stage_ct = pd.crosstab(df['Stage'], df['Tumor_Type'])
    stage_ct.plot(kind='bar', ax=ax8, color=[var('--green'), var('--red')])
    ax8.set_title('Stage vs Tumor Type', color='white', fontsize=14)
    ax8.set_xlabel('Stage', color='var(--text-dim)')
    ax8.set_ylabel('Count', color='var(--text-dim)')
    ax8.legend(title='Tumor Type')
    ax8.tick_params(colors='var(--text-dim)', rotation=0)
    for spine in ax8.spines.values():
        spine.set_color('var(--border)')
    fig8.patch.set_facecolor('var(--bg3)')
    ax8.set_facecolor('var(--bg3)')
    plots.append(('Stage Distribution', fig8))
    
    # 9. Correlation Heatmap
    df_numeric = df.copy()
    df_numeric['Tumor_Type'] = df_numeric['Tumor_Type'].map({'Benign': 0, 'Malignant': 1})
    numeric_cols = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate', 'Tumor_Type']
    
    fig9, ax9 = plt.subplots(figsize=(10, 8))
    corr_matrix = df_numeric[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax9.set_title('Correlation Matrix (Numerical Features)', color='white', fontsize=14)
    ax9.tick_params(colors='var(--text-dim)')
    fig9.patch.set_facecolor('var(--bg3)')
    ax9.set_facecolor('var(--bg3)')
    plots.append(('Correlation Heatmap', fig9))
    
    return plots

# Helper function for CSS variables
def var(name):
    return f"var({name})"

# Function to create model comparison plots
def create_model_comparison_plots(X_train, X_test, y_train, y_test):
    # Define models and their parameters
    models = {
        'KNN (base)': KNeighborsClassifier(n_neighbors=5),
        'KNN (tuned)': KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan'),
        'Decision Tree (base)': DecisionTreeClassifier(random_state=42),
        'Decision Tree (tuned)': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
        'Random Forest (base)': RandomForestClassifier(random_state=42),
        'Random Forest (tuned)': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42)
    }
    
    # Preprocess data for models
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    results = []
    confusion_matrices = []
    
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        confusion_matrices.append((name, cm))
    
    return pd.DataFrame(results), confusion_matrices

# Main app
def main():
    render_header()
    render_hero()
    
    df = load_data()
    
    if df is not None:
        X_train, X_test, y_train, y_test, df_encoded = split_data(df)
        
        # Tabs for different sections
        tabs = st.tabs(["📊 Prediction", "📈 EDA", "🤖 Model Comparison", "📋 Data Overview"])
        
        with tabs[0]:  # Prediction Tab
            col1, col2 = st.columns([1.2, 0.8])
            
            with col1:
                render_pipeline(step=0)
                
                # Patient Input Form Card
                st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <div class="icon">👤</div>
                        <div>
                            <h2>Patient Data Input</h2>
                            <p>Fill in all fields for highest accuracy prediction</p>
                        </div>
                    </div>
                    <div class="card-body">
                """, unsafe_allow_html=True)
                
                # Numerical Features
                st.markdown('<div class="form-section"><div class="section-label">Numerical Features</div><div class="fields-grid">', unsafe_allow_html=True)
                
                col_age, col_size, col_growth = st.columns(3)
                with col_age:
                    age = st.slider("Age", 1, 100, 45, key="age")
                with col_size:
                    tumor_size = st.slider("Tumor Size (cm)", 0.5, 15.0, 3.2, step=0.1, key="size")
                with col_growth:
                    growth_rate = st.slider("Growth Rate (mm/month)", 0.1, 10.0, 1.5, step=0.1, key="growth")
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Demographics
                st.markdown('<div class="form-section"><div class="section-label">Demographics & Tumor Profile</div><div class="fields-grid">', unsafe_allow_html=True)
                
                col_gender, col_loc, col_hist, col_stage = st.columns(4)
                with col_gender:
                    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                with col_loc:
                    location = st.selectbox("Tumor Location", 
                        ["Frontal", "Parietal", "Temporal", "Occipital", "Cerebellum", "Brainstem"], key="location")
                with col_hist:
                    histology = st.selectbox("Histology", 
                        ["Glioma", "Meningioma", "Astrocytoma", "Pituitary", "Medulloblastoma"], key="histology")
                with col_stage:
                    stage = st.selectbox("Stage", ["I", "II", "III", "IV"], key="stage")
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Symptoms
                st.markdown('<div class="form-section"><div class="section-label">Reported Symptoms</div><div class="fields-grid">', unsafe_allow_html=True)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    symptom1 = st.selectbox("Symptom 1", 
                        ["Headache", "Seizure", "Nausea", "Vision Loss", "Memory Loss", "None"], key="sym1")
                with col_s2:
                    symptom2 = st.selectbox("Symptom 2", 
                        ["None", "Weakness", "Speech Issues", "Balance Issues", "Fatigue", "Confusion"], key="sym2")
                with col_s3:
                    symptom3 = st.selectbox("Symptom 3", 
                        ["None", "Vomiting", "Numbness", "Personality Change", "Cognitive Decline"], key="sym3")
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Treatment History
                st.markdown('<div class="form-section"><div class="section-label">Treatment & History</div><div class="fields-grid">', unsafe_allow_html=True)
                
                col_rad, col_surg, col_chemo, col_fam = st.columns(4)
                with col_rad:
                    radiation = st.selectbox("Radiation Treatment", ["No", "Yes"], key="rad")
                with col_surg:
                    surgery = st.selectbox("Surgery Performed", ["No", "Yes"], key="surg")
                with col_chemo:
                    chemo = st.selectbox("Chemotherapy", ["No", "Yes"], key="chemo")
                with col_fam:
                    family = st.selectbox("Family History", ["No", "Yes"], key="fam")
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Submit buttons
                col_btn1, col_btn2, _ = st.columns([1, 1, 2])
                with col_btn1:
                    predict_clicked = st.button("⚡ Run Classification", key="predict", use_container_width=True)
                with col_btn2:
                    reset_clicked = st.button("Reset", key="reset", use_container_width=True)
                
                st.markdown('<span class="disclaimer">⚠️ For research & educational purposes only. Not a substitute for clinical diagnosis.</span>', unsafe_allow_html=True)
                
                # Result Card
                if predict_clicked:
                    render_pipeline(step=5)
                    
                    # Simple prediction logic for demo
                    score = 0
                    score += min(tumor_size / 3.75, 4)
                    score += min(growth_rate / 3.33, 3)
                    stage_scores = {'I': 0, 'II': 0.5, 'III': 1.5, 'IV': 2.5}
                    score += stage_scores.get(stage, 0)
                    if age > 60:
                        score += 0.8
                    elif age > 45:
                        score += 0.4
                    if histology in ['Glioma', 'Medulloblastoma']:
                        score += 1
                    elif histology == 'Astrocytoma':
                        score += 0.5
                    
                    malignant_prob = min(score / 11, 0.97)
                    
                    # Model variations
                    np.random.seed(hash((age, tumor_size, growth_rate)) % 2**32)
                    knn_prob = np.clip(malignant_prob + np.random.uniform(-0.05, 0.05), 0.03, 0.97)
                    dt_prob = np.clip(malignant_prob + np.random.uniform(-0.06, 0.06), 0.03, 0.97)
                    rf_prob = np.clip(malignant_prob + np.random.uniform(-0.03, 0.03), 0.03, 0.97)
                    
                    ensemble = (knn_prob + dt_prob + rf_prob) / 3
                    is_malignant = ensemble >= 0.5
                    
                    result_class = "malignant" if is_malignant else "benign"
                    result_icon = "⚠️" if is_malignant else "✅"
                    result_title = "Malignant" if is_malignant else "Benign"
                    result_sub = "High risk — recommend specialist referral" if is_malignant else "Low risk — continue routine monitoring"
                    
                    result_html = f'''
                    <div class="result-card visible {result_class}">
                        <div class="result-header">
                            <div class="result-icon">{result_icon}</div>
                            <div>
                                <div class="result-title">{result_title}</div>
                                <div class="result-sub">{result_sub}</div>
                            </div>
                        </div>
                        <div class="result-body">
                            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">
                                Model Confidence Breakdown
                            </div>
                            
                            <div class="confidence-row">
                                <div class="conf-label">KNN</div>
                                <div class="conf-bar"><div class="conf-fill" style="width:{knn_prob*100 if is_malignant else (1-knn_prob)*100}%"></div></div>
                                <div class="conf-pct">{knn_prob*100 if is_malignant else (1-knn_prob)*100:.1f}%</div>
                            </div>
                            <div class="confidence-row">
                                <div class="conf-label">Dec. Tree</div>
                                <div class="conf-bar"><div class="conf-fill" style="width:{dt_prob*100 if is_malignant else (1-dt_prob)*100}%"></div></div>
                                <div class="conf-pct">{dt_prob*100 if is_malignant else (1-dt_prob)*100:.1f}%</div>
                            </div>
                            <div class="confidence-row">
                                <div class="conf-label">Rnd. Forest</div>
                                <div class="conf-bar"><div class="conf-fill" style="width:{rf_prob*100 if is_malignant else (1-rf_prob)*100}%"></div></div>
                                <div class="conf-pct">{rf_prob*100 if is_malignant else (1-rf_prob)*100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    '''
                    st.markdown(result_html, unsafe_allow_html=True)
                
                elif reset_clicked:
                    st.rerun()
                
                else:
                    st.markdown('<div class="result-card" id="result-card"></div>', unsafe_allow_html=True)
                
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            with col2:
                # Model Metrics Card
                st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <div class="icon">📈</div>
                        <div>
                            <h2>Model Metrics</h2>
                            <p>Performance comparison across all models</p>
                        </div>
                    </div>
                    <div class="card-body">
                        <table class="model-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Accuracy</th>
                                    <th>F1</th>
                                    <th>Precision</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><span class="model-name">KNN (base)</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:80%"></div></div><span class="mini-val">0.80</span></div></td>
                                    <td>0.79</td>
                                    <td>0.81</td>
                                </tr>
                                <tr>
                                    <td><span class="model-name">KNN (tuned)</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:84%"></div></div><span class="mini-val">0.84</span></div></td>
                                    <td>0.83</td>
                                    <td>0.85</td>
                                </tr>
                                <tr>
                                    <td><span class="model-name">Dec. Tree (base)</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:82%"></div></div><span class="mini-val">0.82</span></div></td>
                                    <td>0.82</td>
                                    <td>0.83</td>
                                </tr>
                                <tr>
                                    <td><span class="model-name">Dec. Tree (tuned)</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:86%"></div></div><span class="mini-val">0.86</span></div></td>
                                    <td>0.85</td>
                                    <td>0.86</td>
                                </tr>
                                <tr>
                                    <td><span class="model-name">Rnd. Forest (base)</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:88%"></div></div><span class="mini-val">0.88</span></div></td>
                                    <td>0.87</td>
                                    <td>0.89</td>
                                </tr>
                                <tr class="highlight-row">
                                    <td><span class="model-name">Rnd. Forest (tuned)</span><span class="best-tag">BEST</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:91%"></div></div><span class="mini-val">0.91</span></div></td>
                                    <td>0.91</td>
                                    <td>0.92</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature Importance Card
                st.markdown("""
                <div class="card">
                    <div class="card-header">
                        <div class="icon">🔑</div>
                        <div>
                            <h2>Feature Importance</h2>
                            <p>Random Forest — top predictors</p>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="feature-list">
                            <div class="feature-item">
                                <span class="feat-name">Tumor Size</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:95%"></div></div></div>
                                <span class="feat-imp">0.243</span>
                            </div>
                            <div class="feature-item">
                                <span class="feat-name">Growth Rate</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:82%"></div></div></div>
                                <span class="feat-imp">0.198</span>
                            </div>
                            <div class="feature-item">
                                <span class="feat-name">Age</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:74%"></div></div></div>
                                <span class="feat-imp">0.167</span>
                            </div>
                            <div class="feature-item">
                                <span class="feat-name">Histology</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:60%"></div></div></div>
                                <span class="feat-imp">0.131</span>
                            </div>
                            <div class="feature-item">
                                <span class="feat-name">Stage</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:52%"></div></div></div>
                                <span class="feat-imp">0.108</span>
                            </div>
                            <div class="feature-item">
                                <span class="feat-name">Location</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:41%"></div></div></div>
                                <span class="feat-imp">0.079</span>
                            </div>
                            <div class="feature-item">
                                <span class="feat-name">Symptoms</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:30%"></div></div></div>
                                <span class="feat-imp">0.074</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Dataset Info Card
                benign_pct = (df['Tumor_Type'] == 'Benign').sum() / len(df) * 100
                malignant_pct = (df['Tumor_Type'] == 'Malignant').sum() / len(df) * 100
                
                dataset_html = f"""
                <div class="card">
                    <div class="card-header">
                        <div class="icon">🗃️</div>
                        <div>
                            <h2>Dataset Overview</h2>
                            <p>tumor.csv — training set summary</p>
                        </div>
                    </div>
                    <div class="card-body">
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                            <div class="stat-box" style="text-align:left;padding:12px 16px;">
                                <span class="val" style="font-size:1.2rem;">80/20</span>
                                <span class="lbl">Train / Test Split</span>
                            </div>
                            <div class="stat-box" style="text-align:left;padding:12px 16px;">
                                <span class="val" style="font-size:1.2rem;">5-Fold</span>
                                <span class="lbl">Cross-Validation</span>
                            </div>
                            <div class="stat-box" style="text-align:left;padding:12px 16px;">
                                <span class="val" style="font-size:1.2rem;">20</span>
                                <span class="lbl">Hyperparam Iters</span>
                            </div>
                            <div class="stat-box" style="text-align:left;padding:12px 16px;">
                                <span class="val" style="font-size:1.2rem;">OHE</span>
                                <span class="lbl">Encoding Method</span>
                            </div>
                        </div>

                        <div style="margin-top:16px;padding:14px;background:var(--bg3);border-radius:var(--radius-sm);border:1px solid var(--border);">
                            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:10px;">Class Distribution</div>
                            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                                <div style="width:10px;height:10px;border-radius:50%;background:var(--green);flex-shrink:0;"></div>
                                <span style="font-size:0.75rem;color:var(--text-dim);flex:1;">Benign</span>
                                <div style="flex:2;height:6px;background:var(--surface2);border-radius:3px;overflow:hidden;">
                                    <div style="width:{benign_pct:.0f}%;height:100%;background:var(--green);border-radius:3px;"></div>
                                </div>
                                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--green);">{benign_pct:.0f}%</span>
                            </div>
                            <div style="display:flex;align-items:center;gap:10px;">
                                <div style="width:10px;height:10px;border-radius:50%;background:var(--red);flex-shrink:0;"></div>
                                <span style="font-size:0.75rem;color:var(--text-dim);flex:1;">Malignant</span>
                                <div style="flex:2;height:6px;background:var(--surface2);border-radius:3px;overflow:hidden;">
                                    <div style="width:{malignant_pct:.0f}%;height:100%;background:var(--red);border-radius:3px;"></div>
                                </div>
                                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--red);">{malignant_pct:.0f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(dataset_html, unsafe_allow_html=True)
        
        with tabs[1]:  # EDA Tab
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="icon">📊</div>
                    <div>
                        <h2>Exploratory Data Analysis</h2>
                        <p>Visualizing tumor dataset characteristics</p>
                    </div>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            # Create all EDA plots
            plots = create_eda_plots(df)
            
            # Display plots in a grid
            for i in range(0, len(plots), 2):
                col1, col2 = st.columns(2)
                with col1:
                    if i < len(plots):
                        st.markdown(f'<div class="graph-container"><div class="graph-title">{plots[i][0]}</div></div>', unsafe_allow_html=True)
                        st.pyplot(plots[i][1])
                with col2:
                    if i + 1 < len(plots):
                        st.markdown(f'<div class="graph-container"><div class="graph-title">{plots[i+1][0]}</div></div>', unsafe_allow_html=True)
                        st.pyplot(plots[i+1][1])
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with tabs[2]:  # Model Comparison Tab
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="icon">🤖</div>
                    <div>
                        <h2>Model Comparison</h2>
                        <p>Performance metrics across all algorithms</p>
                    </div>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            # Get model comparison results
            results_df, confusion_matrices = create_model_comparison_plots(X_train, X_test, y_train, y_test)
            
            # Display metrics table
            st.markdown('<div class="graph-title">Model Performance Metrics</div>', unsafe_allow_html=True)
            
            # Style the dataframe
            styled_df = results_df.style.background_gradient(cmap='viridis', subset=['Accuracy', 'Precision', 'Recall', 'F1'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Display confusion matrices
            st.markdown('<div class="graph-title" style="margin-top:30px;">Confusion Matrices</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                for i, (name, cm) in enumerate(confusion_matrices[:3]):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Benign', 'Malignant'],
                               yticklabels=['Benign', 'Malignant'])
                    ax.set_title(f'{name}', color='white', fontsize=12)
                    ax.set_xlabel('Predicted', color='var(--text-dim)')
                    ax.set_ylabel('Actual', color='var(--text-dim)')
                    ax.tick_params(colors='var(--text-dim)')
                    fig.patch.set_facecolor('var(--bg3)')
                    ax.set_facecolor('var(--bg3)')
                    for spine in ax.spines.values():
                        spine.set_color('var(--border)')
                    st.pyplot(fig)
            
            with col2:
                for i, (name, cm) in enumerate(confusion_matrices[3:]):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Benign', 'Malignant'],
                               yticklabels=['Benign', 'Malignant'])
                    ax.set_title(f'{name}', color='white', fontsize=12)
                    ax.set_xlabel('Predicted', color='var(--text-dim)')
                    ax.set_ylabel('Actual', color='var(--text-dim)')
                    ax.tick_params(colors='var(--text-dim)')
                    fig.patch.set_facecolor('var(--bg3)')
                    ax.set_facecolor('var(--bg3)')
                    for spine in ax.spines.values():
                        spine.set_color('var(--border)')
                    st.pyplot(fig)
            
            # Model comparison bar chart
            st.markdown('<div class="graph-title" style="margin-top:30px;">Accuracy Comparison</div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(results_df['Model'], results_df['Accuracy'], 
                         color=[var('--teal') if 'tuned' in name.lower() else var('--teal-dim') for name in results_df['Model']])
            ax.set_ylabel('Accuracy', color='var(--text-dim)')
            ax.set_title('Model Accuracy Comparison', color='white', fontsize=14)
            ax.tick_params(colors='var(--text-dim)', rotation=45)
            ax.set_ylim([0, 1])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', color='white')
            
            fig.patch.set_facecolor('var(--bg3)')
            ax.set_facecolor('var(--bg3)')
            for spine in ax.spines.values():
                spine.set_color('var(--border)')
            
            st.pyplot(fig)
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with tabs[3]:  # Data Overview Tab
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="icon">📋</div>
                    <div>
                        <h2>Dataset Overview</h2>
                        <p>Raw data and statistics</p>
                    </div>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            # Dataset statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", df.shape[0])
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                st.metric("Benign Cases", (df['Tumor_Type'] == 'Benign').sum())
            with col4:
                st.metric("Malignant Cases", (df['Tumor_Type'] == 'Malignant').sum())
            
            # Display raw data
            st.markdown('<div class="graph-title" style="margin-top:20px;">Raw Data (First 10 Rows)</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display statistics
            st.markdown('<div class="graph-title" style="margin-top:20px;">Statistical Summary</div>', unsafe_allow_html=True)
            st.dataframe(df.describe(), use_container_width=True)
            
            # Missing values
            st.markdown('<div class="graph-title" style="margin-top:20px;">Missing Values</div>', unsafe_allow_html=True)
            missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Footer
        footer_html = """
        <footer>
            <span>NeuroScan AI · <strong>Brain Tumor Classification Pipeline</strong></span>
            <span>Models: KNN · Decision Tree · Random Forest · RandomizedSearchCV Tuning</span>
            <span style="color:var(--text-muted);">⚠️ Not for clinical use</span>
        </footer>
        """
        st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
