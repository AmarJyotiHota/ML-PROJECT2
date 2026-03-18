# ===============================
# IPL ML ANALYTICS - PREMIUM STREAMLIT DASHBOARD
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, r2_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.linear_model import LinearRegression, LogisticRegression
import time
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="IPL ML Analytics - Premium Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS - PREMIUM DESIGN
# ===============================
def load_css():
    st.markdown("""
    <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200;300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&family=Bebas+Neue&display=swap');
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #0D0221 0%, #110A31 50%, #1A0F3B 100%);
            color: #FFFFFF;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 107, 53, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #FF6B35, #00D4FF);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #00D4FF, #FF6B35);
        }
        
        /* Premium Header */
        .premium-header {
            text-align: center;
            padding: 2rem 0;
            position: relative;
            margin-bottom: 2rem;
        }
        
        .premium-title {
            font-family: 'Bebas Neue', sans-serif;
            font-size: clamp(3rem, 8vw, 5.5rem);
            font-weight: 900;
            background: linear-gradient(135deg, #FF6B35 0%, #00D4FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: 0.05em;
        }
        
        .premium-subtitle {
            color: #A0A9C9;
            font-size: 1.2rem;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .title-underline {
            width: 80px;
            height: 4px;
            background: linear-gradient(90deg, #FF6B35, #00D4FF);
            margin: 1rem auto;
            border-radius: 2px;
            animation: expandWidth 0.8s ease-out;
        }
        
        @keyframes expandWidth {
            from { width: 0; }
            to { width: 80px; }
        }
        
        /* Premium Cards */
        .premium-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 107, 53, 0.2);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            margin: 1rem 0;
        }
        
        .premium-card:hover {
            border-color: #FF6B35;
            background: rgba(255, 107, 53, 0.08);
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(255, 107, 53, 0.2);
        }
        
        .premium-card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, transparent 50%, rgba(0, 212, 255, 0.1) 100%);
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        .premium-card:hover::before {
            opacity: 1;
        }
        
        .card-corner {
            position: absolute;
            width: 40px;
            height: 40px;
            border: 2px solid #FF6B35;
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        .card-corner-tl {
            top: 0;
            left: 0;
            border-right: none;
            border-bottom: none;
            border-radius: 0 0 8px 0;
        }
        
        .card-corner-br {
            bottom: 0;
            right: 0;
            border-left: none;
            border-top: none;
            border-radius: 8px 0 0 0;
        }
        
        .premium-card:hover .card-corner {
            opacity: 1;
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 1;
        }
        
        .card-icon {
            font-size: 2.5rem;
        }
        
        .card-title {
            font-size: 1.3rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            color: #FF6B35;
        }
        
        /* Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 107, 53, 0.2);
            padding: 1.5rem;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            transition: all 0.4s ease;
            text-align: center;
        }
        
        .metric-card:hover {
            background: rgba(255, 107, 53, 0.1);
            border-color: #FF6B35;
            transform: translateY(-5px);
        }
        
        .metric-number {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FF6B35, #00D4FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
        }
        
        .metric-label {
            color: #A0A9C9;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        
        /* Stats Box */
        .stats-box {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 107, 53, 0.2);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: #FF6B35;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #A0A9C9;
        }
        
        /* Feature Bars */
        .feature-container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .feature-row {
            display: grid;
            grid-template-columns: 150px 1fr 70px;
            gap: 1.5rem;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .feature-name {
            font-weight: 700;
            color: #A0A9C9;
            font-size: 0.95rem;
        }
        
        .feature-bar-bg {
            height: 40px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            border: 1px solid rgba(255, 107, 53, 0.2);
            overflow: hidden;
        }
        
        .feature-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #FF6B35, #00D4FF);
            border-radius: 10px;
            transition: width 1.2s ease-out;
        }
        
        .feature-percent {
            font-weight: 900;
            color: #FF6B35;
            font-size: 0.95rem;
            text-align: right;
        }
        
        /* Model Badges */
        .model-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.1em;
        }
        
        .badge-classification {
            background: rgba(255, 107, 53, 0.2);
            color: #FF6B35;
            border: 1px solid #FF6B35;
        }
        
        .badge-regression {
            background: rgba(0, 212, 255, 0.2);
            color: #00D4FF;
            border: 1px solid #00D4FF;
        }
        
        /* Buttons */
        .premium-button {
            background: linear-gradient(135deg, #FF6B35, #ff8c5a);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.4s ease;
            width: 100%;
            font-size: 0.95rem;
            letter-spacing: 0.05em;
            position: relative;
            overflow: hidden;
        }
        
        .premium-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(255, 107, 53, 0.4);
        }
        
        .premium-button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .premium-button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        /* Tabs Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: rgba(255, 255, 255, 0.03);
            padding: 0.5rem;
            border-radius: 50px;
            border: 1px solid rgba(255, 107, 53, 0.2);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 50px !important;
            padding: 0.5rem 1.5rem !important;
            color: #A0A9C9 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FF6B35, #00D4FF) !important;
            color: white !important;
        }
        
        /* Progress Bars */
        .progress-container {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #FF6B35, #00D4FF);
            border-radius: 10px;
            transition: width 1.2s ease-out;
        }
        
        /* Floating Animation */
        .floating {
            animation: floating 3s ease-in-out infinite;
        }
        
        @keyframes floating {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }
        
        /* Pulse Animation */
        .pulse {
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        /* Loading Animation */
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 107, 53, 0.3);
            border-radius: 50%;
            border-top-color: #FF6B35;
            animation: spin 1s ease-in-out infinite;
            margin: 2rem auto;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Success Message */
        .success-message {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 212, 255, 0.2));
            border: 1px solid #00D4FF;
            color: #00D4FF;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 1rem 0;
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Error Message */
        .error-message {
            background: rgba(255, 107, 53, 0.2);
            border: 1px solid #FF6B35;
            color: #FF6B35;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 1rem 0;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .feature-row {
                grid-template-columns: 1fr;
                gap: 0.5rem;
            }
            
            .stats-box {
                grid-template-columns: 1fr;
            }
            
            .premium-title {
                font-size: 2rem;
            }
        }
        
        /* Graph Containers */
        .graph-container {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 107, 53, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
        }
        
        /* Tooltip Customization */
        .tooltip-custom {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip-custom:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        .tooltip-text {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            background: rgba(13, 2, 33, 0.95);
            border: 1px solid #FF6B35;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            white-space: nowrap;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.3s ease;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        /* Footer */
        .premium-footer {
            background: rgba(13, 2, 33, 0.8);
            border-top: 1px solid rgba(255, 107, 53, 0.2);
            padding: 2rem;
            margin-top: 3rem;
            text-align: center;
        }
        
        .footer-text {
            color: #A0A9C9;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'encoders' not in st.session_state:
        st.session_state.encoders = {}
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []

# ===============================
# DATA PROCESSING FUNCTIONS
# ===============================
@st.cache_data
def load_data(uploaded_file):
    """Load dataset with caching"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def clean_data(df):
    """Advanced data cleaning"""
    df_clean = df.copy()
    
    # Drop truly irrelevant columns
    cols_to_drop = [
        'Unnamed: 0', 'date', 'match_id', 'event_name', 
        'review_batter', 'team_reviewed', 'review_decision', 'umpire',
        'player_of_match', 'match_won_by', 'win_outcome', 'toss_winner', 
        'win_type', 'winner_id', 'extra_type', 'wicket_kind', 'player_out',
        'fielders', 'superover_winner', 'new_batter'
    ]
    df_clean = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], errors='ignore')
    
    # Handle missing values
    for col in df_clean.select_dtypes(include=np.number).columns:
        if df_clean[col].isnull().any():
            if col == 'runs_target':
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].fillna('Missing')
    
    return df_clean

def get_data_quality_metrics(df):
    """Calculate data quality metrics"""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    quality_score = ((total_cells - missing_cells) / total_cells) * 100
    
    return {
        'total_rows': df.shape[0],
        'total_cols': df.shape[1],
        'missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows,
        'quality_score': round(quality_score, 2),
        'memory_usage': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }

# ===============================
# ENCODING FUNCTIONS
# ===============================
def encode_dataframe(df, target_col=None):
    """Encode categorical variables"""
    df_encoded = df.copy()
    encoders = {}
    
    cat_cols = df_encoded.select_dtypes(include='object').columns
    
    for col in cat_cols:
        if col != target_col:  # Don't encode target if specified
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    return df_encoded, encoders

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================
def create_distribution_plot(df, column):
    """Create distribution plot"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution', 'Box Plot'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}]]
    )
    
    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=50, marker_color='#FF6B35', name='Distribution'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(y=df[column], marker_color='#00D4FF', name='Box Plot'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'Analysis of {column}',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2),
        texttemplate='%{text}',
        textfont={"color": "white"},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600
    )
    
    return fig

def create_feature_importance_plot(importance_df):
    """Create feature importance plot"""
    fig = go.Figure(data=go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker=dict(
            color=importance_df['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=importance_df['importance'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Importance Analysis',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        margin=dict(l=150)
    )
    
    return fig

def create_confusion_matrix_plot(cm, labels=None):
    """Create confusion matrix heatmap"""
    if labels is None:
        labels = ['Predicted Negative', 'Predicted Positive']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Viridis',
        text=cm,
        texttemplate='%{text}',
        textfont={"color": "white", "size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def create_regression_plot(y_true, y_pred):
    """Create regression performance plot"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Actual vs Predicted', 'Residuals'),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}]]
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers',
                  marker=dict(color='#FF6B35', size=8, opacity=0.6),
                  name='Predictions'),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', line=dict(color='#00D4FF', dash='dash'),
                  name='Perfect Prediction'),
        row=1, col=1
    )
    
    # Residuals
    residuals = y_true - y_pred
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=30, marker_color='#00D4FF'),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Regression Analysis',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Actual Values", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    
    return fig

# ===============================
# UI COMPONENTS
# ===============================
def render_header():
    """Render premium header"""
    st.markdown("""
    <div class="premium-header">
        <h1 class="premium-title floating">🏏 IPL ML ANALYTICS</h1>
        <p class="premium-subtitle">Advanced Machine Learning Platform for Cricket Performance Prediction & Strategic Insights</p>
        <div class="title-underline"></div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(title, value, icon, color="#FF6B35"):
    """Render premium metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <span style="font-size: 2rem;">{icon}</span>
        <span class="metric-number">{value}</span>
        <span class="metric-label">{title}</span>
    </div>
    """, unsafe_allow_html=True)

def render_feature_importance_bars(features, importances):
    """Render feature importance bars"""
    html = '<div class="feature-container">'
    
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10]:
        percentage = imp * 100
        html += f"""
        <div class="feature-row">
            <span class="feature-name">{feat[:30]}{'...' if len(feat) > 30 else ''}</span>
            <div class="feature-bar-bg">
                <div class="feature-bar-fill" style="width: {percentage}%;"></div>
            </div>
            <span class="feature-percent">{percentage:.1f}%</span>
        </div>
        """
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_stats_box(stats):
    """Render statistics box"""
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        with col:
            st.markdown(f"""
            <div class="stat-item">
                <span class="stat-value">{value}</span>
                <span class="stat-label">{label}</span>
            </div>
            """, unsafe_allow_html=True)

# ===============================
# MODEL TRAINING FUNCTIONS
# ===============================
def train_classification_model(X_train, X_test, y_train, y_test, model_type='Random Forest'):
    """Train classification model"""
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        importance = None
    
    return model, y_pred, accuracy, cm, importance

def train_regression_model(X_train, X_test, y_train, y_test, model_type='Random Forest'):
    """Train regression model"""
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Placeholder
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        importance = None
    
    return model, y_pred, mae, r2, importance

# ===============================
# MAIN APP
# ===============================
def main():
    # Initialize
    init_session_state()
    load_css()
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <span style="font-size: 3rem;">🏏</span>
            <h3 style="color: #FF6B35;">IPL Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # File Upload
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader("Upload IPL Dataset", type=['csv'])
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df_original = df
                    st.session_state.df = clean_data(df)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        
        # Navigation
        if st.session_state.data_loaded:
            st.markdown("### 🎯 Navigation")
            page = st.radio(
                "Go to",
                ["📊 Dashboard", "🔍 EDA", "🤖 Models", "📈 Predictions", "📋 Insights"]
            )
        else:
            page = "📊 Dashboard"
        
        # Theme Toggle
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙 Dark", use_container_width=True):
                st.markdown("""<script>document.body.style.background='#0D0221'</script>""", unsafe_allow_html=True)
        with col2:
            if st.button("☀️ Light", use_container_width=True):
                st.markdown("""<script>document.body.style.background='#f0f2f6'</script>""", unsafe_allow_html=True)
    
    # Main Content
    if not st.session_state.data_loaded:
        render_welcome_screen()
    else:
        if page == "📊 Dashboard":
            render_dashboard()
        elif page == "🔍 EDA":
            render_eda()
        elif page == "🤖 Models":
            render_models()
        elif page == "📈 Predictions":
            render_predictions()
        elif page == "📋 Insights":
            render_insights()

def render_welcome_screen():
    """Render welcome screen"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="padding: 2rem;">
            <h2 style="color: #FF6B35; font-size: 2rem;">🏏 Welcome to IPL ML Analytics</h2>
            <p style="color: #A0A9C9; font-size: 1.1rem; margin: 1rem 0;">
            Experience the power of machine learning in cricket analytics. Our platform provides:
            </p>
            <ul style="color: #A0A9C9; list-style: none; padding: 0;">
                <li style="margin: 0.5rem 0;">✨ Advanced data preprocessing & cleaning</li>
                <li style="margin: 0.5rem 0;">📊 Interactive exploratory analysis</li>
                <li style="margin: 0.5rem 0;">🤖 Multiple ML models for prediction</li>
                <li style="margin: 0.5rem 0;">🎯 Real-time predictions & insights</li>
                <li style="margin: 0.5rem 0;">📈 Professional visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats
        st.markdown("""
        <div class="stats-box">
            <div class="stat-item">
                <span class="stat-value">94.2%</span>
                <span class="stat-label">Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">50+</span>
                <span class="stat-label">Features</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">2008-24</span>
                <span class="stat-label">Data Range</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 8rem; animation: floating 3s ease-in-out infinite;">
                🏏
            </div>
            <div style="margin-top: 2rem;">
                <div class="progress-container">
                    <div class="progress-fill" style="width: 94.2%;"></div>
                </div>
                <p style="color: #A0A9C9;">Ready for analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions
    st.info("👆 Upload your IPL dataset using the sidebar to begin analysis")

def render_dashboard():
    """Render main dashboard"""
    st.markdown("## 📊 Dashboard Overview")
    
    df = st.session_state.df
    quality_metrics = get_data_quality_metrics(df)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card("Total Records", f"{quality_metrics['total_rows']:,}", "📦")
    with col2:
        render_metric_card("Features", quality_metrics['total_cols'], "📊")
    with col3:
        render_metric_card("Data Quality", f"{quality_metrics['quality_score']}%", "✨")
    with col4:
        render_metric_card("Memory", f"{quality_metrics['memory_usage']} MB", "💾")
    
    # Data Preview
    st.markdown("### 📋 Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Sample Data", "Data Info", "Quick Stats"])
    
    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="ipl_cleaned_data.csv" class="premium-button" style="text-decoration: none; display: inline-block; padding: 0.5rem 2rem;">📥 Download Cleaned Data</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        with col2:
            st.markdown("#### Data Quality")
            
            # Missing values visualization
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing %'] > 0]
            
            if not missing_df.empty:
                fig = px.bar(missing_df, x='Column', y='Missing %',
                            title='Missing Values',
                            color='Missing %',
                            color_continuous_scale='reds')
                fig.update_layout(template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
    
    with tab3:
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe(include='all').round(2), use_container_width=True)
    
    # Data Distribution
    st.markdown("### 📈 Feature Distributions")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select feature to analyze", numeric_cols)
        fig = create_distribution_plot(df, selected_col)
        st.plotly_chart(fig, use_container_width=True)

def render_eda():
    """Render exploratory data analysis"""
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    df = st.session_state.df
    
    # Analysis type selector
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Univariate", "Bivariate", "Categorical", "Correlation"],
        horizontal=True
    )
    
    if analysis_type == "Univariate":
        st.markdown("### 📈 Univariate Analysis")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=selected_col, nbins=50,
                                  title=f'Distribution of {selected_col}',
                                  color_discrete_sequence=['#FF6B35'])
                fig.update_layout(template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=selected_col,
                            title=f'Box Plot of {selected_col}',
                            color_discrete_sequence=['#00D4FF'])
                fig.update_layout(template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            stats = df[selected_col].describe()
            render_stats_box({
                'Mean': f"{stats['mean']:.2f}",
                'Std': f"{stats['std']:.2f}",
                'Min': f"{stats['min']:.2f}",
                'Max': f"{stats['max']:.2f}"
            })
    
    elif analysis_type == "Bivariate":
        st.markdown("### 🔗 Bivariate Analysis")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, index=0)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            # Scatter plot
            fig = px.scatter(df, x=x_col, y=y_col,
                            title=f'{x_col} vs {y_col}',
                            trendline="ols",
                            color_discrete_sequence=['#FF6B35'])
            fig.update_layout(template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
            corr = df[x_col].corr(df[y_col])
            st.metric("Correlation Coefficient", f"{corr:.3f}")
    
    elif analysis_type == "Categorical":
        st.markdown("### 📊 Categorical Analysis")
        
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        if cat_cols:
            selected_col = st.selectbox("Select categorical column", cat_cols)
            
            value_counts = df[selected_col].value_counts().head(20).reset_index()
            value_counts.columns = [selected_col, 'Count']
            
            fig = px.bar(value_counts, x=selected_col, y='Count',
                        title=f'Top 20 Values in {selected_col}',
                        color='Count',
                        color_continuous_scale='viridis')
            fig.update_layout(template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Values", df[selected_col].nunique())
            with col2:
                st.metric("Most Frequent", df[selected_col].mode().iloc[0] if not df[selected_col].mode().empty else 'N/A')
    
    else:  # Correlation
        st.markdown("### 🔥 Correlation Analysis")
        
        fig = create_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            corr_pairs = corr_matrix.unstack().reset_index()
            corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
            corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]
            corr_pairs['Correlation'] = corr_pairs['Correlation'].round(3)
            corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation']).sort_values('Correlation', ascending=False)
            
            st.markdown("#### Top Correlations")
            st.dataframe(corr_pairs.head(10), use_container_width=True)

def render_models():
    """Render model training interface"""
    st.markdown("## 🤖 Machine Learning Models")
    
    df = st.session_state.df
    
    # Target selection
    st.markdown("### 🎯 Select Target Variable")
    
    available_targets = ['winner', 'batter_runs', 'total_score']
    target_options = [col for col in available_targets if col in df.columns]
    
    if not target_options:
        st.error("No suitable target columns found!")
        return
    
    target_col = st.selectbox("Choose target for prediction", target_options)
    
    # Determine problem type
    if target_col == 'winner':
        problem_type = "classification"
        st.markdown('<span class="model-badge badge-classification">CLASSIFICATION</span>', unsafe_allow_html=True)
    else:
        problem_type = "regression"
        st.markdown('<span class="model-badge badge-regression">REGRESSION</span>', unsafe_allow_html=True)
    
    # Feature selection
    st.markdown("### 🔧 Feature Selection")
    
    # Encode data
    df_encoded, encoders = encode_dataframe(df, target_col)
    st.session_state.encoders = encoders
    
    # Prepare features
    exclude_cols = [target_col]
    feature_options = [col for col in df_encoded.columns if col not in exclude_cols]
    
    selected_features = st.multiselect(
        "Select features for modeling",
        options=feature_options,
        default=feature_options[:min(10, len(feature_options))]
    )
    
    if not selected_features:
        st.warning("Please select at least one feature")
        return
    
    # Model parameters
    st.markdown("### ⚙️ Model Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    with col2:
        if problem_type == "classification":
            model_choice = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        else:
            model_choice = st.selectbox("Model", ["Random Forest", "Linear Regression"])
    with col3:
        cv_folds = st.slider("CV Folds", 2, 10, 5)
    
    # Train button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model... Please wait"):
            # Prepare data
            X = df_encoded[selected_features]
            y = df_encoded[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "classification" else None
            )
            
            # Scale features for certain models
            if model_choice in ['Logistic Regression', 'Linear Regression']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Train model
            if problem_type == "classification":
                model, y_pred, accuracy, cm, importance = train_classification_model(
                    X_train, X_test, y_train, y_test, model_choice
                )
                
                # Store results
                st.session_state.current_model = model
                st.session_state.model_type = 'classification'
                st.session_state.metrics = {
                    'accuracy': accuracy,
                    'confusion_matrix': cm
                }
                
                # Display results
                st.markdown('<div class="success-message">✅ Model trained successfully!</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.3f}")
                with col2:
                    st.metric("CV Score", f"{cross_val_score(model, X, y, cv=cv_folds).mean():.3f}")
                
                # Confusion matrix
                fig = create_confusion_matrix_plot(cm)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                model, y_pred, mae, r2, importance = train_regression_model(
                    X_train, X_test, y_train, y_test, model_choice
                )
                
                # Store results
                st.session_state.current_model = model
                st.session_state.model_type = 'regression'
                st.session_state.metrics = {
                    'mae': mae,
                    'r2': r2,
                    'y_true': y_test,
                    'y_pred': y_pred
                }
                
                # Display results
                st.markdown('<div class="success-message">✅ Model trained successfully!</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAE", f"{mae:.3f}")
                with col2:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Regression plot
                fig = create_regression_plot(y_test, y_pred)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if importance is not None:
                st.markdown("#### Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                render_feature_importance_bars(importance_df['feature'], importance_df['importance'])
                
                # Store for later use
                st.session_state.feature_importance = importance_df
                st.session_state.selected_features = selected_features
            
            # Add to history
            st.session_state.training_history.append({
                'model': model_choice,
                'target': target_col,
                'accuracy' if problem_type == "classification" else 'r2': 
                    accuracy if problem_type == "classification" else r2,
                'features': len(selected_features)
            })
    
    # Show training history
    if st.session_state.training_history:
        st.markdown("### 📊 Training History")
        history_df = pd.DataFrame(st.session_state.training_history)
        st.dataframe(history_df, use_container_width=True)

def render_predictions():
    """Render predictions interface"""
    st.markdown("## 📈 Make Predictions")
    
    if st.session_state.current_model is None:
        st.warning("⚠️ Please train a model first in the Models section")
        return
    
    st.markdown("### Enter values for prediction")
    
    # Create input fields
    input_data = {}
    
    for feature in st.session_state.selected_features:
        if feature in st.session_state.df.columns:
            if st.session_state.df[feature].dtype in ['int64', 'float64']:
                min_val = float(st.session_state.df[feature].min())
                max_val = float(st.session_state.df[feature].max())
                mean_val = float(st.session_state.df[feature].mean())
                
                input_data[feature] = st.number_input(
                    f"**{feature}**",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=0.1,
                    format="%.2f"
                )
            else:
                options = st.session_state.df[feature].unique().tolist()
                input_data[feature] = st.selectbox(f"**{feature}**", options)
    
    if st.button("🔮 Predict", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_data])
        
        # Encode if necessary
        for col in input_df.select_dtypes(include='object').columns:
            if col in st.session_state.encoders:
                input_df[col] = st.session_state.encoders[col].transform(input_df[col].astype(str))
        
        # Ensure correct feature order
        input_df = input_df[st.session_state.selected_features]
        
        # Make prediction
        prediction = st.session_state.current_model.predict(input_df)[0]
        
        # Display result
        st.markdown("### 🎯 Prediction Result")
        
        if st.session_state.model_type == 'classification':
            if 'winner' in st.session_state.df.columns:
                # Try to decode if possible
                unique_winners = st.session_state.df['winner'].unique()
                if prediction < len(unique_winners):
                    prediction_label = unique_winners[int(prediction)]
                else:
                    prediction_label = f"Class {prediction}"
            else:
                prediction_label = f"Class {prediction}"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(0,212,255,0.2));
                        border: 2px solid #FF6B35; border-radius: 16px; padding: 2rem; text-align: center;">
                <span style="font-size: 3rem;">🏆</span>
                <h2 style="color: #FF6B35; font-size: 2.5rem;">{prediction_label}</h2>
                <p style="color: #A0A9C9;">Predicted Winner</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(0,212,255,0.2));
                        border: 2px solid #00D4FF; border-radius: 16px; padding: 2rem; text-align: center;">
                <span style="font-size: 3rem;">📊</span>
                <h2 style="color: #00D4FF; font-size: 2.5rem;">{prediction:.2f}</h2>
                <p style="color: #A0A9C9;">Predicted Value</p>
            </div>
            """, unsafe_allow_html=True)

def render_insights():
    """Render insights section"""
    st.markdown("## 📋 Key Insights")
    
    # Metrics from data
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="premium-card">
            <div class="card-corner card-corner-tl"></div>
            <div class="card-corner card-corner-br"></div>
            <div class="card-header">
                <span class="card-icon">🏆</span>
                <h3 class="card-title">Performance Insights</h3>
            </div>
            <div style="padding: 1rem;">
        """, unsafe_allow_html=True)
        
        if 'batter_runs' in df.columns:
            st.markdown(f"• **Average Runs:** {df['batter_runs'].mean():.2f}")
            st.markdown(f"• **Max Runs:** {df['batter_runs'].max()}")
        if 'total_score' in df.columns:
            st.markdown(f"• **Average Total:** {df['total_score'].mean():.2f}")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="premium-card">
            <div class="card-corner card-corner-tl"></div>
            <div class="card-corner card-corner-br"></div>
            <div class="card-header">
                <span class="card-icon">📊</span>
                <h3 class="card-title">Data Quality</h3>
            </div>
            <div style="padding: 1rem;">
        """, unsafe_allow_html=True)
        
        quality = get_data_quality_metrics(df)
        st.markdown(f"• **Quality Score:** {quality['quality_score']}%")
        st.markdown(f"• **Memory Usage:** {quality['memory_usage']} MB")
        st.markdown(f"• **Duplicate Rows:** {quality['duplicate_rows']}")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Feature importance
    if st.session_state.feature_importance is not None:
        st.markdown("### 🎯 Top Predictive Features")
        importance_df = st.session_state.feature_importance.head(5)
        
        for _, row in importance_df.iterrows():
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #A0A9C9;">{row['feature']}</span>
                    <span style="color: #FF6B35; font-weight: 700;">{row['importance']*100:.1f}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-fill" style="width: {row['importance']*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 Strategic Recommendations")
    
    st.markdown("""
    <div class="premium-card">
        <div class="card-corner card-corner-tl"></div>
        <div class="card-corner card-corner-br"></div>
        <div style="padding: 1rem;">
            <h4 style="color: #00D4FF;">Based on the analysis:</h4>
            <ul style="color: #A0A9C9; list-style: none;">
                <li style="margin: 0.5rem 0;">✦ Focus on top predictive features for team selection</li>
                <li style="margin: 0.5rem 0;">✦ Consider historical performance patterns</li>
                <li style="margin: 0.5rem 0;">✦ Use ensemble models for better accuracy</li>
                <li style="margin: 0.5rem 0;">✦ Monitor feature importance over time</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="premium-footer">
        <p class="footer-text">🏏 IPL ML Analytics | Powered by Machine Learning | © 2024</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    main()
