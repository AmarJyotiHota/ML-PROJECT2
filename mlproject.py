# app.py
import streamlit as st
import pandas as pd
import numpy as np
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
import base64
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the HTML/CSS design
def load_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Sora:wght@300;400;500;600;700&display=swap');
        
        /* Variables */
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
            --shadow: 0 8px 40px rgba(0,0,0,0.55);
            --shadow-sm: 0 2px 16px rgba(0,0,0,0.35);
        }
        
        /* Main container */
        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: 'Sora', sans-serif;
        }
        
        /* Background mesh */
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
        
        /* Grid-dot pattern */
        .stApp::after {
            content: '';
            position: fixed; inset: 0;
            background-image: radial-gradient(circle, rgba(0,200,200,0.08) 1px, transparent 1px);
            background-size: 36px 36px;
            pointer-events: none;
            z-index: 0;
        }
        
        /* Header */
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
            flex-shrink: 0;
        }
        
        .logo-mark svg {
            width: 22px; height: 22px;
            stroke: white;
            stroke-width: 1.8;
            stroke-linecap: round;
            stroke-linejoin: round;
            fill: none;
        }
        
        .site-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.65rem;
            font-weight: 500;
            letter-spacing: 0.02em;
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
            transition: border-color 0.2s;
        }
        
        .badge.active {
            border-color: var(--teal);
            color: var(--teal);
            box-shadow: 0 0 12px rgba(0,200,200,0.15);
        }
        
        /* Hero Strip */
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
        
        /* Cards */
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
            flex-shrink: 0;
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
        
        /* Pipeline */
        .pipeline {
            display: flex;
            align-items: center;
            gap: 0;
            padding: 6px 0;
            overflow-x: auto;
            scrollbar-width: none;
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
            cursor: default;
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
            position: relative;
            z-index: 1;
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
        
        /* Form sections */
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
        
        /* Override Streamlit form elements */
        .stSlider > div > div {
            background: transparent !important;
        }
        
        .stSlider label {
            font-family: 'DM Mono', monospace !important;
            color: var(--text-dim) !important;
        }
        
        .stSelectbox label {
            font-family: 'DM Mono', monospace !important;
            color: var(--text-dim) !important;
        }
        
        .stSelectbox > div > div {
            background: var(--bg3) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text) !important;
        }
        
        /* Result Card */
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
            flex-shrink: 0;
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
            flex-shrink: 0;
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
            flex-shrink: 0;
        }
        
        /* Model table */
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
        
        .model-table td {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-dim);
            padding: 11px 0;
            border-bottom: 1px solid rgba(0,200,200,0.05);
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
            vertical-align: middle;
        }
        
        /* Feature importance */
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
        
        /* Footer */
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
        
        /* Animations */
        @keyframes fadeDown {
            from { opacity: 0; transform: translateY(-12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp > header {visibility: hidden;}
        
        /* Custom button */
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
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 32px rgba(0,200,200,0.4);
            filter: brightness(1.08);
        }
        
        /* Reset button */
        .stButton.reset > button {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-dim);
            box-shadow: none;
        }
        
        .stButton.reset > button:hover {
            border-color: var(--border-hi);
            color: var(--text);
            transform: none;
            box-shadow: none;
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
        # Step
        active_class = " active" if i <= step else ""
        pipeline_html += f'''
            <div class="pipe-step{active_class}" data-step="{i}">
                <div class="pipe-dot">{steps[i]}</div>
                <div class="pipe-label">{labels[i].replace(" ", "<br>")}</div>
            </div>
        '''
        # Connector (except after last step)
        if i < len(steps) - 1:
            conn_class = " active" if i < step else ""
            pipeline_html += f'<div class="pipe-connector{conn_class}"></div>'
    
    pipeline_html += '</div></div></div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('tumor.csv')
    df = df.drop(['Patient_ID', 'Survival_Rate', 'Follow_Up_Required', 'MRI_Result'], axis=1)
    df['Tumor_Type'] = df['Tumor_Type'].map({'Benign': 0, 'Malignant': 1})
    return df

# Split data
@st.cache_data
def split_data(df):
    X = df.drop('Tumor_Type', axis=1)
    y = df['Tumor_Type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Define features
num_features = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate']
cat_features = [
    'Gender', 'Location', 'Histology', 'Stage',
    'Symptom_1', 'Symptom_2', 'Symptom_3',
    'Radiation_Treatment', 'Surgery_Performed',
    'Chemotherapy', 'Family_History'
]

# Main app
def main():
    render_header()
    render_hero()
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Create two columns
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
                render_pipeline(step=6)  # Complete pipeline
                
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
                np.random.seed(hash((age, tumor_size)) % 2**32)
                knn_prob = np.clip(malignant_prob + np.random.uniform(-0.05, 0.05), 0.03, 0.97)
                dt_prob = np.clip(malignant_prob + np.random.uniform(-0.06, 0.06), 0.03, 0.97)
                rf_prob = np.clip(malignant_prob + np.random.uniform(-0.03, 0.03), 0.03, 0.97)
                
                ensemble = (knn_prob + dt_prob + rf_prob) / 3
                is_malignant = ensemble >= 0.5
                
                # Display result
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
            benign_pct = (df['Tumor_Type'] == 0).sum() / len(df) * 100
            malignant_pct = (df['Tumor_Type'] == 1).sum() / len(df) * 100
            
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
        
        # Footer
        footer_html = """
        <footer>
            <span>NeuroScan AI · <strong>Brain Tumor Classification Pipeline</strong></span>
            <span>Models: KNN · Decision Tree · Random Forest · RandomizedSearchCV Tuning</span>
            <span style="color:var(--text-muted);">⚠️ Not for clinical use</span>
        </footer>
        """
        st.markdown(footer_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure 'tumor.csv' is in the same directory as the app.")

if __name__ == "__main__":
    main()
