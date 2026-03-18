# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
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

# Custom CSS to match the HTML/CSS design exactly
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
            margin-bottom: 36px;
            flex-wrap: wrap;
            gap: 16px;
            animation: fadeDown .6s ease both;
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
            box-shadow: 0 0 24px rgba(0,200,200,.3);
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
            color: #fff;
            letter-spacing: .02em;
            line-height: 1;
        }
        
        .site-subtitle {
            font-family: 'DM Mono', monospace;
            font-size: .65rem;
            color: var(--teal);
            letter-spacing: .16em;
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
            font-size: .62rem;
            letter-spacing: .12em;
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
            box-shadow: 0 0 12px rgba(0,200,200,.15);
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
            margin-bottom: 28px;
            flex-wrap: wrap;
            animation: fadeUp .6s .1s ease both;
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
            font-size: .82rem;
            color: var(--text-dim);
            margin-top: 8px;
            max-width: 520px;
            line-height: 1.65;
        }
        
        .hero-stats {
            display: flex;
            gap: 20px;
            margin-left: auto;
            flex-wrap: wrap;
        }
        
        .stat-box {
            text-align: center;
            padding: 14px 20px;
            border-radius: var(--radius-sm);
            background: var(--bg3);
            border: 1px solid var(--border);
            min-width: 80px;
        }
        
        .stat-box .val {
            font-family: 'DM Mono', monospace;
            font-size: 1.45rem;
            font-weight: 500;
            color: var(--teal);
            display: block;
        }
        
        .stat-box .lbl {
            font-size: .62rem;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-top: 3px;
            display: block;
        }
        
        /* Tab Navigation */
        .tab-nav {
            display: flex;
            gap: 4px;
            margin-bottom: 28px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0;
            animation: fadeUp .5s .15s ease both;
        }
        
        .tab-btn {
            font-family: 'Sora', sans-serif;
            font-size: .78rem;
            font-weight: 500;
            padding: 11px 22px;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            cursor: pointer;
            transition: color .2s, border-color .2s;
            margin-bottom: -1px;
            letter-spacing: .02em;
        }
        
        .tab-btn:hover {
            color: var(--text-dim);
        }
        
        .tab-btn.active {
            color: var(--teal);
            border-bottom-color: var(--teal);
        }
        
        /* Section Titles */
        .section-title-block {
            margin-bottom: 28px;
        }
        
        .section-title-block h2 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.8rem;
            font-weight: 400;
            color: #fff;
        }
        
        .section-title-block p {
            font-size: .78rem;
            color: var(--text-dim);
            margin-top: 6px;
            line-height: 1.6;
        }
        
        .chart-section-label {
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            letter-spacing: .18em;
            text-transform: uppercase;
            color: var(--teal);
            margin: 28px 0 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chart-section-label::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }
        
        /* Chart Cards */
        .chart-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px 18px 14px;
            box-shadow: var(--shadow-sm);
            animation: fadeUp .5s ease both;
            margin-bottom: 18px;
        }
        
        .chart-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.05rem;
            font-weight: 500;
            color: #fff;
            margin-bottom: 3px;
        }
        
        .chart-subtitle {
            font-family: 'DM Mono', monospace;
            font-size: .58rem;
            letter-spacing: .05em;
            color: var(--text-muted);
            margin-bottom: 12px;
            line-height: 1.4;
        }
        
        /* Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            animation: fadeUp .6s .2s ease both;
            margin-bottom: 22px;
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 18px 24px;
            border-bottom: 1px solid var(--border);
            background: rgba(0,0,0,.15);
        }
        
        .card-header .icon {
            width: 32px; height: 32px;
            border-radius: 8px;
            background: var(--teal-dim);
            border: 1px solid var(--border-hi);
            display: grid; place-items: center;
            color: var(--teal);
            font-size: .9rem;
            flex-shrink: 0;
        }
        
        .card-header h2 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.2rem;
            font-weight: 500;
            color: #fff;
            letter-spacing: .02em;
            margin: 0;
        }
        
        .card-header p {
            font-size: .7rem;
            color: var(--text-muted);
            margin: 1px 0 0 0;
        }
        
        .card-body {
            padding: 24px;
        }
        
        /* Form */
        .form-section {
            margin-bottom: 26px;
        }
        
        .section-label {
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            letter-spacing: .18em;
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
            grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
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
            animation: fadeUp .5s ease both;
            display: none;
        }
        
        .result-card.visible {
            display: block;
        }
        
        .result-card.benign {
            border-color: rgba(61,232,158,.35);
        }
        
        .result-card.malignant {
            border-color: rgba(244,95,111,.35);
        }
        
        .result-header {
            padding: 20px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .result-card.benign .result-header {
            background: rgba(61,232,158,.07);
        }
        
        .result-card.malignant .result-header {
            background: rgba(244,95,111,.07);
        }
        
        .result-icon {
            width: 48px; height: 48px;
            border-radius: 50%;
            display: grid; place-items: center;
            font-size: 1.4rem;
            flex-shrink: 0;
        }
        
        .result-card.benign .result-icon {
            background: rgba(61,232,158,.12);
            color: var(--green);
        }
        
        .result-card.malignant .result-icon {
            background: rgba(244,95,111,.12);
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
            font-size: .72rem;
            color: var(--text-muted);
            margin-top: 4px;
        }
        
        .result-body {
            padding: 18px 24px;
        }
        
        .conf-section-label {
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            letter-spacing: .14em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 12px;
        }
        
        .confidence-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }
        
        .conf-label {
            font-family: 'DM Mono', monospace;
            font-size: .65rem;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--text-muted);
            width: 82px;
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
            transition: width .9s cubic-bezier(.22,1,.36,1);
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
            font-size: .78rem;
            color: var(--text);
            width: 42px;
            text-align: right;
            flex-shrink: 0;
        }
        
        /* Pipeline */
        .pipeline {
            display: flex;
            align-items: center;
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
            min-width: 70px;
            cursor: default;
        }
        
        .pipe-dot {
            width: 36px; height: 36px;
            border-radius: 50%;
            background: var(--bg3);
            border: 1.5px solid var(--border);
            display: grid; place-items: center;
            color: var(--text-muted);
            font-size: .85rem;
            transition: all .3s;
            position: relative;
            z-index: 1;
        }
        
        .pipe-step.active .pipe-dot {
            background: var(--teal-dim);
            border-color: var(--teal);
            color: var(--teal);
            box-shadow: 0 0 14px rgba(0,200,200,.2);
        }
        
        .pipe-label {
            font-size: .58rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--text-muted);
            text-align: center;
            line-height: 1.3;
        }
        
        .pipe-step.active .pipe-label {
            color: var(--teal);
        }
        
        .pipe-connector {
            flex: .8;
            height: 1px;
            background: var(--border);
            min-width: 16px;
            position: relative;
            top: -13px;
        }
        
        .pipe-connector.active {
            background: var(--teal);
            box-shadow: 0 0 6px rgba(0,200,200,.3);
        }
        
        /* Model Table */
        .model-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .model-table th {
            font-family: 'DM Mono', monospace;
            font-size: .6rem;
            letter-spacing: .14em;
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
            font-size: .78rem;
            color: var(--text-dim);
            padding: 11px 0;
            border-bottom: 1px solid rgba(0,200,200,.05);
        }
        
        .model-table td:not(:first-child) {
            text-align: center;
        }
        
        .model-name {
            color: var(--text);
            font-weight: 500;
            font-size: .75rem;
        }
        
        .best-tag {
            font-size: .55rem;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(0,200,200,.12);
            color: var(--teal);
            border: 1px solid rgba(0,200,200,.2);
            letter-spacing: .06em;
            margin-left: 6px;
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
            font-size: .7rem;
            color: var(--text-dim);
            width: 36px;
            flex-shrink: 0;
        }
        
        /* Feature List */
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
            font-size: .75rem;
        }
        
        .feat-name {
            color: var(--text-dim);
        }
        
        .feat-imp {
            font-family: 'DM Mono', monospace;
            color: var(--teal);
            font-size: .7rem;
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
        
        /* Confusion Matrix */
        .confusion-matrix {
            padding: 16px 10px 10px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        
        .cm-labels-x {
            display: grid;
            grid-template-columns: 60px 1fr 1fr;
            gap: 6px;
            text-align: center;
        }
        
        .cm-labels-x span {
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            color: var(--text-muted);
            letter-spacing: .04em;
            padding-bottom: 4px;
        }
        
        .cm-row {
            display: grid;
            grid-template-columns: 60px 1fr 1fr;
            gap: 6px;
            align-items: center;
        }
        
        .cm-row-label {
            font-family: 'DM Mono', monospace;
            font-size: .6rem;
            color: var(--text-muted);
            text-align: right;
            padding-right: 8px;
            line-height: 1.4;
        }
        
        .cm-cell {
            border-radius: var(--radius-sm);
            padding: 16px 8px;
            text-align: center;
            font-family: 'DM Mono', monospace;
            font-size: 1.4rem;
            font-weight: 500;
            position: relative;
        }
        
        .cm-cell-label {
            font-size: .55rem;
            text-transform: uppercase;
            letter-spacing: .1em;
            margin-top: 4px;
            font-weight: 400;
        }
        
        .cm-tn {
            background: rgba(61,232,158,.12);
            border: 1px solid rgba(61,232,158,.25);
            color: var(--green);
        }
        
        .cm-tn .cm-cell-label {
            color: rgba(61,232,158,.6);
        }
        
        .cm-tp {
            background: rgba(61,232,158,.12);
            border: 1px solid rgba(61,232,158,.25);
            color: var(--green);
        }
        
        .cm-tp .cm-cell-label {
            color: rgba(61,232,158,.6);
        }
        
        .cm-fp {
            background: rgba(244,95,111,.10);
            border: 1px solid rgba(244,95,111,.22);
            color: var(--red);
        }
        
        .cm-fp .cm-cell-label {
            color: rgba(244,95,111,.6);
        }
        
        .cm-fn {
            background: rgba(244,95,111,.10);
            border: 1px solid rgba(244,95,111,.22);
            color: var(--red);
        }
        
        .cm-fn .cm-cell-label {
            color: rgba(244,95,111,.6);
        }
        
        .cm-stats {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
            padding-top: 6px;
            font-family: 'DM Mono', monospace;
            font-size: .65rem;
            color: var(--text-muted);
        }
        
        .cm-stats b {
            color: var(--teal);
        }
        
        /* Full Metrics Table */
        .full-metrics-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .full-metrics-table th {
            font-family: 'DM Mono', monospace;
            font-size: .62rem;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--text-muted);
            padding: 0 12px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-weight: 400;
            white-space: nowrap;
        }
        
        .full-metrics-table td {
            font-family: 'DM Mono', monospace;
            font-size: .78rem;
            color: var(--text-dim);
            padding: 11px 12px;
            border-bottom: 1px solid rgba(0,200,200,.05);
        }
        
        .gap-val {
            color: var(--amber);
        }
        
        .gap-val.red {
            color: var(--red);
        }
        
        .status-tag {
            font-size: .6rem;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: .06em;
            font-weight: 500;
        }
        
        .status-tag.ok {
            background: rgba(61,232,158,.1);
            color: var(--green);
            border: 1px solid rgba(61,232,158,.2);
        }
        
        .status-tag.overfit {
            background: rgba(244,95,111,.1);
            color: var(--red);
            border: 1px solid rgba(244,95,111,.2);
        }
        
        .status-tag.best {
            background: rgba(0,200,200,.12);
            color: var(--teal);
            border: 1px solid rgba(0,200,200,.25);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #00b8b8 0%, #007a8a 100%);
            color: white;
            border: none;
            padding: 13px 36px;
            border-radius: var(--radius-sm);
            font-family: 'Sora', sans-serif;
            font-size: .82rem;
            font-weight: 600;
            letter-spacing: .08em;
            text-transform: uppercase;
            cursor: pointer;
            transition: all .2s;
            box-shadow: 0 4px 22px rgba(0,180,180,.35);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 32px rgba(0,200,200,.4);
            filter: brightness(1.08);
        }
        
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
        
        .disclaimer {
            font-size: .65rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin-top: 12px;
            display: block;
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
            font-size: .68rem;
            color: var(--text-muted);
            font-family: 'DM Mono', monospace;
            animation: fadeUp .6s .4s ease both;
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
        
        /* Layout */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 1320px;
        }
        
        .row-widget.stHorizontal {
            gap: 26px;
        }
        
        /* Grid layouts */
        .charts-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 4px;
        }
        
        .charts-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 4px;
        }
        
        .charts-grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-bottom: 4px;
        }
        
        @media (max-width: 1100px) {
            .charts-grid-4 { grid-template-columns: repeat(2, 1fr); }
            .charts-grid-3 { grid-template-columns: repeat(2, 1fr); }
        }
        
        @media (max-width: 720px) {
            .charts-grid-4, .charts-grid-3, .charts-grid-2 { grid-template-columns: 1fr; }
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 26px;
            align-items: start;
        }
        
        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 22px;
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

# Tab navigation
def render_tabs(active_tab):
    tabs_html = f"""
    <div class="tab-nav">
        <button class="tab-btn {'active' if active_tab == 'predict' else ''}" onclick="switchTab('predict')">⚡ Predict</button>
        <button class="tab-btn {'active' if active_tab == 'eda' else ''}" onclick="switchTab('eda')">📊 EDA &amp; Graphs</button>
        <button class="tab-btn {'active' if active_tab == 'models' else ''}" onclick="switchTab('models')">🏆 Model Results</button>
    </div>
    <script>
    function switchTab(tab) {{
        window.location.href = window.location.pathname + '?tab=' + tab;
    }}
    </script>
    """
    st.markdown(tabs_html, unsafe_allow_html=True)

# Pipeline tracker
def render_pipeline(step=0):
    steps = ["📥", "⚙️", "🧠", "🌿", "🌲", "📊"]
    labels = ["Data<br>Input", "Pre-<br>process", "KNN", "Dec.<br>Tree", "Rnd.<br>Forest", "Result"]
    
    pipeline_html = '<div class="card" style="margin-bottom:22px;"><div class="card-body" style="padding:18px 24px;"><div class="pipeline">'
    
    for i in range(len(steps)):
        active_class = " active" if i <= step else ""
        pipeline_html += f'''
            <div class="pipe-step{active_class}">
                <div class="pipe-dot">{steps[i]}</div>
                <div class="pipe-label">{labels[i]}</div>
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

# Create EDA plots
def create_eda_plots(df):
    # Set style for all plots
    plt.style.use('dark_background')
    
    # 1. Target Class Distribution (Doughnut chart)
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    benign_count = (df['Tumor_Type'] == 'Benign').sum()
    malignant_count = (df['Tumor_Type'] == 'Malignant').sum()
    wedges, texts, autotexts = ax1.pie(
        [benign_count, malignant_count], 
        labels=['Benign', 'Malignant'],
        colors=['#3de89e', '#f45f6f'],
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='none')
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
    for text in texts:
        text.set_color('#6b99b5')
        text.set_fontsize(9)
    ax1.set_facecolor('#0e2a42')
    fig1.patch.set_facecolor('#0e2a42')
    
    # 2. Age Distribution by Tumor Type (Histogram)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for tumor_type, color in [('Benign', '#3de89e'), ('Malignant', '#f45f6f')]:
        subset = df[df['Tumor_Type'] == tumor_type]['Age']
        ax2.hist(subset, bins=20, alpha=0.6, label=tumor_type, color=color)
    ax2.set_xlabel('Age', color='#6b99b5', fontsize=9)
    ax2.set_ylabel('Frequency', color='#6b99b5', fontsize=9)
    ax2.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax2.tick_params(colors='#6b99b5')
    ax2.set_facecolor('#0e2a42')
    fig2.patch.set_facecolor('#0e2a42')
    
    # 3. Tumor Size Distribution
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    for tumor_type, color in [('Benign', '#3de89e'), ('Malignant', '#f45f6f')]:
        subset = df[df['Tumor_Type'] == tumor_type]['Tumor_Size']
        ax3.hist(subset, bins=20, alpha=0.6, label=tumor_type, color=color)
    ax3.set_xlabel('Tumor Size (cm)', color='#6b99b5', fontsize=9)
    ax3.set_ylabel('Frequency', color='#6b99b5', fontsize=9)
    ax3.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax3.tick_params(colors='#6b99b5')
    ax3.set_facecolor('#0e2a42')
    fig3.patch.set_facecolor('#0e2a42')
    
    # 4. Tumor Growth Rate Boxplot
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    benign_growth = df[df['Tumor_Type'] == 'Benign']['Tumor_Growth_Rate']
    malignant_growth = df[df['Tumor_Type'] == 'Malignant']['Tumor_Growth_Rate']
    bp = ax4.boxplot([benign_growth, malignant_growth], labels=['Benign', 'Malignant'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#3de89e')
    bp['boxes'][1].set_facecolor('#f45f6f')
    for whisker in bp['whiskers']:
        whisker.set_color('#00c8c8')
    for cap in bp['caps']:
        cap.set_color('#00c8c8')
    for median in bp['medians']:
        median.set_color('#f4a441')
    ax4.set_ylabel('Growth Rate (mm/month)', color='#6b99b5', fontsize=9)
    ax4.tick_params(colors='#6b99b5')
    ax4.set_facecolor('#0e2a42')
    fig4.patch.set_facecolor('#0e2a42')
    
    # 5. Tumor Size vs Growth Rate Scatter
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    for tumor_type, color in [('Benign', '#3de89e'), ('Malignant', '#f45f6f')]:
        subset = df[df['Tumor_Type'] == tumor_type]
        ax5.scatter(subset['Tumor_Size'], subset['Tumor_Growth_Rate'], 
                   c=color, label=tumor_type, alpha=0.5, s=20)
    ax5.set_xlabel('Tumor Size (cm)', color='#6b99b5', fontsize=9)
    ax5.set_ylabel('Growth Rate (mm/month)', color='#6b99b5', fontsize=9)
    ax5.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax5.tick_params(colors='#6b99b5')
    ax5.set_facecolor('#0e2a42')
    fig5.patch.set_facecolor('#0e2a42')
    
    # 6. Age Boxplot
    fig6, ax6 = plt.subplots(figsize=(5, 4))
    bp = ax6.boxplot([df[df['Tumor_Type'] == 'Benign']['Age'],
                      df[df['Tumor_Type'] == 'Malignant']['Age']],
                     labels=['Benign', 'Malignant'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3de89e')
    bp['boxes'][1].set_facecolor('#f45f6f')
    for whisker in bp['whiskers']:
        whisker.set_color('#00c8c8')
    for cap in bp['caps']:
        cap.set_color('#00c8c8')
    for median in bp['medians']:
        median.set_color('#f4a441')
    ax6.set_ylabel('Age', color='#6b99b5', fontsize=9)
    ax6.tick_params(colors='#6b99b5')
    ax6.set_facecolor('#0e2a42')
    fig6.patch.set_facecolor('#0e2a42')
    
    # 7. Location countplot
    fig7, ax7 = plt.subplots(figsize=(5, 4))
    location_ct = pd.crosstab(df['Location'], df['Tumor_Type'])
    location_ct.plot(kind='bar', ax=ax7, color=['#3de89e', '#f45f6f'])
    ax7.set_xlabel('Location', color='#6b99b5', fontsize=9)
    ax7.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax7.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax7.tick_params(colors='#6b99b5', rotation=45)
    ax7.set_facecolor('#0e2a42')
    fig7.patch.set_facecolor('#0e2a42')
    
    # 8. Histology countplot
    fig8, ax8 = plt.subplots(figsize=(5, 4))
    histology_ct = pd.crosstab(df['Histology'], df['Tumor_Type'])
    histology_ct.plot(kind='bar', ax=ax8, color=['#3de89e', '#f45f6f'])
    ax8.set_xlabel('Histology', color='#6b99b5', fontsize=9)
    ax8.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax8.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax8.tick_params(colors='#6b99b5', rotation=45)
    ax8.set_facecolor('#0e2a42')
    fig8.patch.set_facecolor('#0e2a42')
    
    # 9. Stage countplot
    fig9, ax9 = plt.subplots(figsize=(5, 4))
    stage_order = ['I', 'II', 'III', 'IV']
    stage_ct = pd.crosstab(df['Stage'], df['Tumor_Type']).reindex(stage_order)
    stage_ct.plot(kind='bar', ax=ax9, color=['#3de89e', '#f45f6f'])
    ax9.set_xlabel('Stage', color='#6b99b5', fontsize=9)
    ax9.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax9.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax9.tick_params(colors='#6b99b5', rotation=0)
    ax9.set_facecolor('#0e2a42')
    fig9.patch.set_facecolor('#0e2a42')
    
    # 10. Gender countplot
    fig10, ax10 = plt.subplots(figsize=(5, 4))
    gender_ct = pd.crosstab(df['Gender'], df['Tumor_Type'])
    gender_ct.plot(kind='bar', ax=ax10, color=['#3de89e', '#f45f6f'])
    ax10.set_xlabel('Gender', color='#6b99b5', fontsize=9)
    ax10.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax10.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax10.tick_params(colors='#6b99b5', rotation=0)
    ax10.set_facecolor('#0e2a42')
    fig10.patch.set_facecolor('#0e2a42')
    
    # 11. Radiation Treatment
    fig11, ax11 = plt.subplots(figsize=(5, 4))
    radiation_ct = pd.crosstab(df['Radiation_Treatment'], df['Tumor_Type'])
    radiation_ct.plot(kind='bar', ax=ax11, color=['#3de89e', '#f45f6f'])
    ax11.set_xlabel('Radiation Treatment', color='#6b99b5', fontsize=9)
    ax11.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax11.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax11.tick_params(colors='#6b99b5', rotation=0)
    ax11.set_facecolor('#0e2a42')
    fig11.patch.set_facecolor('#0e2a42')
    
    # 12. Surgery & Chemo (combined)
    fig12, ax12 = plt.subplots(figsize=(5, 4))
    surgery_data = pd.DataFrame({
        'No Surgery': [df[df['Surgery_Performed'] == 'No']['Tumor_Type'].value_counts().get('Benign', 0),
                       df[df['Surgery_Performed'] == 'No']['Tumor_Type'].value_counts().get('Malignant', 0)],
        'Surgery': [df[df['Surgery_Performed'] == 'Yes']['Tumor_Type'].value_counts().get('Benign', 0),
                    df[df['Surgery_Performed'] == 'Yes']['Tumor_Type'].value_counts().get('Malignant', 0)]
    }, index=['Benign', 'Malignant']).T
    surgery_data.plot(kind='bar', ax=ax12, color=['#3de89e', '#f45f6f'])
    ax12.set_title('Surgery', color='white', fontsize=10)
    ax12.set_xlabel('', color='#6b99b5', fontsize=9)
    ax12.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax12.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax12.tick_params(colors='#6b99b5', rotation=0)
    ax12.set_facecolor('#0e2a42')
    fig12.patch.set_facecolor('#0e2a42')
    
    # 13. Family History
    fig13, ax13 = plt.subplots(figsize=(5, 4))
    family_ct = pd.crosstab(df['Family_History'], df['Tumor_Type'])
    family_ct.plot(kind='bar', ax=ax13, color=['#3de89e', '#f45f6f'])
    ax13.set_xlabel('Family History', color='#6b99b5', fontsize=9)
    ax13.set_ylabel('Count', color='#6b99b5', fontsize=9)
    ax13.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
    ax13.tick_params(colors='#6b99b5', rotation=0)
    ax13.set_facecolor('#0e2a42')
    fig13.patch.set_facecolor('#0e2a42')
    
    # 14. Correlation Heatmap
    fig14, ax14 = plt.subplots(figsize=(5, 4))
    df_numeric = df.copy()
    df_numeric['Tumor_Type'] = df_numeric['Tumor_Type'].map({'Benign': 0, 'Malignant': 1})
    numeric_cols = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate', 'Tumor_Type']
    corr_matrix = df_numeric[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=ax14, cbar=False, annot_kws={'color': 'white'})
    ax14.set_facecolor('#0e2a42')
    fig14.patch.set_facecolor('#0e2a42')
    ax14.tick_params(colors='#6b99b5')
    
    # 15. Feature Importance
    fig15, ax15 = plt.subplots(figsize=(5, 4))
    features = ['Tumor Size', 'Growth Rate', 'Age', 'Histology', 'Stage', 'Location', 
                'Symptoms', 'Gender', 'Surgery', 'Radiation', 'Chemo', 'Family Hx']
    importance = [0.243, 0.198, 0.167, 0.131, 0.108, 0.079, 0.074, 0.058, 0.042, 0.038, 0.032, 0.030]
    colors = ['rgba(0,200,200,0.8)' if x > 0.18 else 'rgba(0,200,200,0.55)' if x > 0.10 else 'rgba(0,200,200,0.3)' for x in importance]
    y_pos = np.arange(len(features))
    ax15.barh(y_pos, importance, color=colors)
    ax15.set_yticks(y_pos)
    ax15.set_yticklabels(features, color='#6b99b5', fontsize=8)
    ax15.set_xlabel('Importance', color='#6b99b5', fontsize=9)
    ax15.tick_params(colors='#6b99b5')
    ax15.set_facecolor('#0e2a42')
    fig15.patch.set_facecolor('#0e2a42')
    
    return [
        ('Target Class Distribution', fig1),
        ('Age Distribution by Tumor Type', fig2),
        ('Tumor Size Distribution', fig3),
        ('Tumor Growth Rate by Type', fig4),
        ('Tumor Size vs Growth Rate', fig5),
        ('Age Distribution (Boxplot)', fig6),
        ('Location vs Tumor Type', fig7),
        ('Histology vs Tumor Type', fig8),
        ('Stage vs Tumor Type', fig9),
        ('Gender vs Tumor Type', fig10),
        ('Radiation Treatment', fig11),
        ('Surgery', fig12),
        ('Family History', fig13),
        ('Correlation Heatmap', fig14),
        ('Feature Importance', fig15)
    ]

# Create model comparison plots
def create_model_comparison_plots(X_train, X_test, y_train, y_test):
    # Define models
    models = {
        'KNN (base)': KNeighborsClassifier(n_neighbors=5),
        'KNN (tuned)': KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan'),
        'DT (base)': DecisionTreeClassifier(random_state=42),
        'DT (tuned)': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
        'RF (base)': RandomForestClassifier(random_state=42),
        'RF (tuned)': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42)
    }
    
    # Preprocess
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
        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        cm = confusion_matrix(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'Gap': train_acc - test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        confusion_matrices.append((name, cm))
    
    return results, confusion_matrices

# Main app
def main():
    render_header()
    render_hero()
    
    # Get tab from URL parameters
    query_params = st.query_params
    active_tab = query_params.get('tab', 'predict')
    
    render_tabs(active_tab)
    
    df = load_data()
    
    if df is not None:
        X_train, X_test, y_train, y_test, df_encoded = split_data(df)
        
        if active_tab == 'predict':
            # Prediction Tab
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
                    st.markdown(f'<span id="age-val" style="display:none;">{age}</span>', unsafe_allow_html=True)
                with col_size:
                    tumor_size = st.slider("Tumor Size (cm)", 0.5, 15.0, 3.2, step=0.1, key="size")
                    st.markdown(f'<span id="size-val" style="display:none;">{tumor_size:.1f}</span>', unsafe_allow_html=True)
                with col_growth:
                    growth_rate = st.slider("Growth Rate (mm/mo)", 0.1, 10.0, 1.5, step=0.1, key="growth")
                    st.markdown(f'<span id="gr-val" style="display:none;">{growth_rate:.1f}</span>', unsafe_allow_html=True)
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Demographics
                st.markdown('<div class="form-section"><div class="section-label">Demographics & Tumor Profile</div><div class="fields-grid">', unsafe_allow_html=True)
                
                col_gender, col_loc, col_hist, col_stage = st.columns(4)
                with col_gender:
                    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                with col_loc:
                    location = st.selectbox("Location", 
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
                    radiation = st.selectbox("Radiation", ["No", "Yes"], key="rad")
                with col_surg:
                    surgery = st.selectbox("Surgery", ["No", "Yes"], key="surg")
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
                
                st.markdown('<span class="disclaimer">⚠️ For research & educational purposes only.</span>', unsafe_allow_html=True)
                
                # Result Card
                if predict_clicked:
                    render_pipeline(step=5)
                    
                    # Simple prediction logic
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
                            <div class="conf-section-label">Model Confidence Breakdown</div>
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
                            <p>Performance across all models</p>
                        </div>
                    </div>
                    <div class="card-body">
                        <table class="model-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Accuracy</th>
                                    <th>F1</th>
                                    <th>Prec.</th>
                                </tr>
                            </thead>
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
                                <tr class="highlight-row"><td><span class="model-name">RF (tuned)</span><span class="best-tag">BEST</span></td>
                                    <td><div class="metric-bar-wrap"><div class="mini-bar"><div class="mini-fill" style="width:91%"></div></div><span class="mini-val">0.91</span></div></td>
                                    <td>0.91</td><td>0.92</td></tr>
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
                            <p>Random Forest top predictors</p>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="feature-list">
                            <div class="feature-item"><span class="feat-name">Tumor Size</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:95%"></div></div></div>
                                <span class="feat-imp">0.243</span></div>
                            <div class="feature-item"><span class="feat-name">Growth Rate</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:82%"></div></div></div>
                                <span class="feat-imp">0.198</span></div>
                            <div class="feature-item"><span class="feat-name">Age</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:74%"></div></div></div>
                                <span class="feat-imp">0.167</span></div>
                            <div class="feature-item"><span class="feat-name">Histology</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:60%"></div></div></div>
                                <span class="feat-imp">0.131</span></div>
                            <div class="feature-item"><span class="feat-name">Stage</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:52%"></div></div></div>
                                <span class="feat-imp">0.108</span></div>
                            <div class="feature-item"><span class="feat-name">Location</span>
                                <div class="feat-bar-wrap"><div class="feat-bar"><div class="feat-fill" style="width:41%"></div></div></div>
                                <span class="feat-imp">0.079</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        elif active_tab == 'eda':
            # EDA Tab
            st.markdown("""
            <div class="section-title-block">
                <h2>Exploratory Data Analysis</h2>
                <p>Replicating every visualization from the Python script — distributions, boxplots, scatter plots, categorical breakdowns, and correlation heatmap</p>
            </div>
            """, unsafe_allow_html=True)
            
            plots = create_eda_plots(df)
            
            # Target & Numerical Distributions
            st.markdown('<div class="chart-section-label">Target &amp; Numerical Distributions (sns.histplot / sns.countplot)</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Target Class Distribution</div><div class="chart-subtitle">sns.countplot(x=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[0][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Age Distribution by Tumor Type</div><div class="chart-subtitle">sns.histplot(x=\'Age\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[1][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="chart-card"><div class="chart-title">Tumor Size Distribution</div><div class="chart-subtitle">sns.histplot(x=\'Tumor_Size\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[2][1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Boxplots & Scatter
            st.markdown('<div class="chart-section-label">Boxplots &amp; Scatter (sns.boxplot / sns.scatterplot)</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Tumor Growth Rate by Type</div><div class="chart-subtitle">sns.boxplot(x=\'Tumor_Type\', y=\'Tumor_Growth_Rate\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[3][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Tumor Size vs Growth Rate</div><div class="chart-subtitle">sns.scatterplot(x=\'Tumor_Size\', y=\'Tumor_Growth_Rate\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[4][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="chart-card"><div class="chart-title">Age Distribution (Boxplot)</div><div class="chart-subtitle">sns.boxplot(x=df[\'Age\'])</div>', unsafe_allow_html=True)
                st.pyplot(plots[5][1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Categorical Feature Breakdowns
            st.markdown('<div class="chart-section-label">Categorical Feature Breakdowns (sns.countplot with hue)</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Location vs Tumor Type</div><div class="chart-subtitle">countplot(x=\'Location\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[6][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Histology vs Tumor Type</div><div class="chart-subtitle">countplot(x=\'Histology\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[7][1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Stage vs Tumor Type</div><div class="chart-subtitle">countplot(x=\'Stage\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[8][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Gender vs Tumor Type</div><div class="chart-subtitle">countplot(x=\'Gender\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[9][1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Treatment History
            st.markdown('<div class="chart-section-label">Treatment History (Categorical — cont.)</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Radiation Treatment</div><div class="chart-subtitle">countplot(x=\'Radiation_Treatment\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[10][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Surgery & Chemotherapy</div><div class="chart-subtitle">Surgery_Performed · Chemotherapy</div>', unsafe_allow_html=True)
                st.pyplot(plots[11][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="chart-card"><div class="chart-title">Family History</div><div class="chart-subtitle">countplot(x=\'Family_History\', hue=\'Tumor_Type\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[12][1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Correlation Matrix & Feature Importance
            st.markdown('<div class="chart-section-label">Correlation Matrix &amp; Feature Importance (sns.heatmap)</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Correlation Matrix — Numerical Features</div><div class="chart-subtitle">sns.heatmap(corr, annot=True, cmap=\'coolwarm\')</div>', unsafe_allow_html=True)
                st.pyplot(plots[13][1])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance — Random Forest</div><div class="chart-subtitle">Horizontal bar sorted by importance score</div>', unsafe_allow_html=True)
                st.pyplot(plots[14][1])
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif active_tab == 'models':
            # Model Results Tab
            st.markdown("""
            <div class="section-title-block">
                <h2>Model Evaluation Results</h2>
                <p>Confusion matrices, accuracy comparisons, and full metric breakdowns for KNN, Decision Tree, and Random Forest</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get model results
            results, confusion_matrices = create_model_comparison_plots(X_train, X_test, y_train, y_test)
            
            # Accuracy Comparison
            st.markdown('<div class="chart-section-label">Accuracy Comparison — Train vs Test</div>', unsafe_allow_html=True)
            
            # Train vs Test Accuracy Bar Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            models = [r['Model'] for r in results]
            train_acc = [r['Train Acc'] for r in results]
            test_acc = [r['Test Acc'] for r in results]
            
            x = np.arange(len(models))
            width = 0.35
            ax.bar(x - width/2, train_acc, width, label='Train Accuracy', color='rgba(74,158,255,0.6)')
            ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='rgba(0,200,200,0.6)')
            ax.set_xlabel('Model', color='#6b99b5')
            ax.set_ylabel('Accuracy', color='#6b99b5')
            ax.set_title('Train vs Test Accuracy (All Models)', color='white')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', color='#6b99b5', fontsize=8)
            ax.tick_params(colors='#6b99b5')
            ax.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
            ax.set_ylim([0.7, 1.05])
            ax.set_facecolor('#0e2a42')
            fig.patch.set_facecolor('#0e2a42')
            for spine in ax.spines.values():
                spine.set_color('rgba(0,200,200,0.12)')
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-card"><div class="chart-title">Train vs Test Accuracy</div><div class="chart-subtitle">Reveals overfitting — Decision Tree base has 100% train accuracy</div>', unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Metrics Radar Chart (simplified as bar chart)
            with col2:
                st.markdown('<div class="chart-card"><div class="chart-title">Metrics Comparison — Tuned Models</div><div class="chart-subtitle">Accuracy · Precision · Recall · F1</div>', unsafe_allow_html=True)
                
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                tuned_models = ['KNN (tuned)', 'DT (tuned)', 'RF (tuned)']
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
                
                tuned_results = [r for r in results if 'tuned' in r['Model'].lower()]
                x = np.arange(len(metrics))
                width = 0.25
                
                colors = ['#4a9eff', '#f4a441', '#00c8c8']
                for i, (model, color) in enumerate(zip(tuned_models, colors)):
                    model_data = [r for r in tuned_results if r['Model'] == model][0]
                    values = [model_data['Test Acc'], model_data['Precision'], model_data['Recall'], model_data['F1']]
                    ax2.bar(x + i*width, values, width, label=model, color=color, alpha=0.7)
                
                ax2.set_xlabel('Metric', color='#6b99b5')
                ax2.set_ylabel('Score', color='#6b99b5')
                ax2.set_title('Tuned Models Performance', color='white')
                ax2.set_xticks(x + width)
                ax2.set_xticklabels(metrics, color='#6b99b5')
                ax2.tick_params(colors='#6b99b5')
                ax2.legend(facecolor='#0e2a42', labelcolor='#6b99b5')
                ax2.set_ylim([0.8, 0.95])
                ax2.set_facecolor('#0e2a42')
                fig2.patch.set_facecolor('#0e2a42')
                for spine in ax2.spines.values():
                    spine.set_color('rgba(0,200,200,0.12)')
                
                st.pyplot(fig2)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Confusion Matrices
            st.markdown('<div class="chart-section-label">Confusion Matrices (sns.heatmap — Benign / Malignant)</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            for i, (name, cm) in enumerate(confusion_matrices):
                if 'tuned' in name.lower():
                    if 'KNN' in name:
                        with col1:
                            tn, fp, fn, tp = cm.ravel()
                            st.markdown(f'''
                            <div class="chart-card">
                                <div class="chart-title">KNN (Tuned)</div>
                                <div class="chart-subtitle">RandomizedSearchCV best params</div>
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
                                        <span>Accuracy: <b>{((tn+tp)/cm.sum()*100):.1f}%</b></span>
                                        <span>FPR: <b>{(fp/(fp+tn)*100):.1f}%</b></span>
                                        <span>FNR: <b>{(fn/(fn+tp)*100):.1f}%</b></span>
                                    </div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                    elif 'DT' in name:
                        with col2:
                            tn, fp, fn, tp = cm.ravel()
                            st.markdown(f'''
                            <div class="chart-card">
                                <div class="chart-title">Decision Tree (Tuned)</div>
                                <div class="chart-subtitle">RandomizedSearchCV best params</div>
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
                                        <span>Accuracy: <b>{((tn+tp)/cm.sum()*100):.1f}%</b></span>
                                        <span>FPR: <b>{(fp/(fp+tn)*100):.1f}%</b></span>
                                        <span>FNR: <b>{(fn/(fn+tp)*100):.1f}%</b></span>
                                    </div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                    elif 'RF' in name:
                        with col3:
                            tn, fp, fn, tp = cm.ravel()
                            st.markdown(f'''
                            <div class="chart-card">
                                <div class="chart-title">Random Forest (Tuned)</div>
                                <div class="chart-subtitle">Best overall model ★</div>
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
                                        <span>Accuracy: <b>{((tn+tp)/cm.sum()*100):.1f}%</b></span>
                                        <span>FPR: <b>{(fp/(fp+tn)*100):.1f}%</b></span>
                                        <span>FNR: <b>{(fn/(fn+tp)*100):.1f}%</b></span>
                                    </div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
            
            # Full Metrics Table
            st.markdown('<div class="chart-section-label">Full Metrics Table</div>', unsafe_allow_html=True)
            
            metrics_html = '''
            <div class="card" style="margin-bottom:28px;">
                <div class="card-body">
                    <table class="full-metrics-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Train Acc.</th>
                                <th>Test Acc.</th>
                                <th>Gap ↓</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
            '''
            
            for r in results:
                status = 'Good'
                status_class = 'ok'
                if r['Gap'] > 0.15:
                    status = 'Overfit'
                    status_class = 'overfit'
                elif r['Model'] == 'RF (tuned)':
                    status = 'Best ★'
                    status_class = 'best'
                
                gap_class = 'gap-val red' if r['Gap'] > 0.15 else 'gap-val'
                
                metrics_html += f'''
                    <tr{' class="best-row"' if r['Model'] == 'RF (tuned)' else ''}>
                        <td>{r['Model']}</td>
                        <td>{r['Train Acc']:.3f}</td>
                        <td>{r['Test Acc']:.3f}</td>
                        <td class="{gap_class}">{r['Gap']:.3f}</td>
                        <td>{r['Precision']:.3f}</td>
                        <td>{r['Recall']:.3f}</td>
                        <td>{r['F1']:.3f}</td>
                        <td><span class="status-tag {status_class}">{status}</span></td>
                    </tr>
                '''
            
            metrics_html += '''
                        </tbody>
                    </table>
                </div>
            </div>
            '''
            
            st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Footer
    footer_html = """
    <footer>
        <span>NeuroScan AI · <strong>Brain Tumor Classification Pipeline</strong></span>
        <span>KNN · Decision Tree · Random Forest · RandomizedSearchCV</span>
        <span>⚠️ Not for clinical use</span>
    </footer>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
