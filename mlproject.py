# ===============================
# IPL ML ANALYTICS - OPTIMIZED VERSION
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, r2_score, 
    confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression
import base64
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels, but provide fallback if not available
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="IPL ML Analytics - Optimized",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CACHING - BIGGEST PERFORMANCE FIX
# ===============================
@st.cache_data
def load_data(uploaded_file):
    """Load dataset with caching - prevents reloading"""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

@st.cache_data
def clean_data(df):
    """Clean and preprocess data with caching"""
    if df is None:
        return None
    
    df_clean = df.copy()
    
    # Drop irrelevant columns
    cols_to_drop = [
        'Unnamed: 0', 'date', 'match_id', 'event_name', 'review_batter', 
        'team_reviewed', 'review_decision', 'umpire', 'player_of_match', 
        'match_won_by', 'win_outcome', 'toss_winner', 'win_type', 'winner_id',
        'extra_type', 'wicket_kind', 'player_out', 'fielders', 'superover_winner', 'new_batter'
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

@st.cache_data
def sample_data(df, sample_size=5000):
    """Sample dataset for faster processing"""
    if df is not None and len(df) > sample_size:
        return df.sample(sample_size, random_state=42)
    return df

@st.cache_resource
def train_classification_model(_X_train, _X_test, y_train, y_test, model_type='Random Forest'):
    """Train classification model with caching - prevents retraining"""
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    
    model.fit(_X_train, y_train)
    y_pred = model.predict(_X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        importance = None
    
    return model, y_pred, accuracy, cm, importance

@st.cache_resource
def train_regression_model(_X_train, _X_test, y_train, y_test, model_type='Random Forest'):
    """Train regression model with caching - prevents retraining"""
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(_X_train, y_train)
    y_pred = model.predict(_X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        importance = None
    
    return model, y_pred, mae, r2, importance

@st.cache_data
def encode_dataframe(df, target_col=None):
    """Encode categorical variables with caching"""
    if df is None:
        return None, {}
    
    df_encoded = df.copy()
    encoders = {}
    
    cat_cols = df_encoded.select_dtypes(include='object').columns
    
    for col in cat_cols:
        if col != target_col:
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    return df_encoded, encoders

@st.cache_data
def get_data_quality_metrics(df):
    """Calculate data quality metrics with caching"""
    if df is None:
        return {}
    
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
# CUSTOM CSS
# ===============================
def load_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0D0221 0%, #110A31 50%, #1A0F3B 100%);
            color: #FFFFFF;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .premium-header {
            text-align: center;
            padding: 2rem 0;
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
        }
        
        .title-underline {
            width: 80px;
            height: 4px;
            background: linear-gradient(90deg, #FF6B35, #00D4FF);
            margin: 1rem auto;
            border-radius: 2px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 107, 53, 0.2);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            transition: all 0.3s ease;
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
        }
        
        .success-message {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 212, 255, 0.2));
            border: 1px solid #00D4FF;
            color: #00D4FF;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #FF6B35, #ff8c5a);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(255, 107, 53, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================
@st.cache_data
def create_distribution_plot(df, column):
    """Create distribution plot with caching"""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Distribution', 'Box Plot'),
                        specs=[[{'type': 'histogram'}, {'type': 'box'}]])
    
    fig.add_trace(go.Histogram(x=df[column], nbinsx=50, marker_color='#FF6B35'), row=1, col=1)
    fig.add_trace(go.Box(y=df[column], marker_color='#00D4FF'), row=1, col=2)
    
    fig.update_layout(template='plotly_dark', height=400, showlegend=False,
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return fig

@st.cache_data
def create_correlation_heatmap(df):
    """Create correlation heatmap with caching"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale='RdBu', zmid=0, text=corr_matrix.round(2),
        texttemplate='%{text}', textfont={"color": "white"}))
    
    fig.update_layout(template='plotly_dark', height=600,
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return fig

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
def init_session_state():
    defaults = {
        'data_loaded': False,
        'df': None,
        'df_original': None,
        'models_trained': False,
        'current_model': None,
        'model_type': None,
        'encoders': {},
        'feature_importance': None,
        'selected_features': [],
        'training_history': [],
        'statsmodels_available': STATSMODELS_AVAILABLE,
        'eda_done': False,
        'model_trained': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ===============================
# MAIN APP
# ===============================
def main():
    # Initialize
    init_session_state()
    load_css()
    
    # Header
    st.markdown("""
    <div class="premium-header">
        <h1 class="premium-title">🏏 IPL ML ANALYTICS</h1>
        <div class="title-underline"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader("Upload IPL Dataset", type=['csv'])
        
        if uploaded_file is not None:
            # Use spinner for loading
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df_original = df
                    
                    # Sample option for large datasets
                    sample_size = st.slider("Sample size (for faster processing)", 
                                           min_value=1000, max_value=20000, value=5000, step=1000)
                    
                    df_sampled = sample_data(df, sample_size)
                    st.session_state.df = clean_data(df_sampled)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded: {len(st.session_state.df)} rows")
        
        # Navigation
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 🎯 Navigation")
            page = st.radio("Go to", ["📊 Dashboard", "🔍 EDA", "🤖 Models", "📈 Predictions"])
        else:
            page = "📊 Dashboard"
    
    # Main content
    if not st.session_state.data_loaded:
        show_welcome()
    else:
        if page == "📊 Dashboard":
            show_dashboard()
        elif page == "🔍 EDA":
            show_eda()
        elif page == "🤖 Models":
            show_models()
        elif page == "📈 Predictions":
            show_predictions()

def show_welcome():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="padding: 2rem;">
            <h2 style="color: #FF6B35;">🏏 Welcome to IPL ML Analytics</h2>
            <p style="color: #A0A9C9;">Upload your IPL dataset to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; font-size: 5rem;">🏏</div>
        """, unsafe_allow_html=True)
    
    st.info("👆 Upload your IPL dataset using the sidebar")

def show_dashboard():
    st.markdown("## 📊 Dashboard")
    
    df = st.session_state.df
    quality = get_data_quality_metrics(df)
    
    # Metrics
    cols = st.columns(4)
    metrics = [
        ("Total Records", f"{quality['total_rows']:,}", "📦"),
        ("Features", quality['total_cols'], "📊"),
        ("Quality", f"{quality['quality_score']}%", "✨"),
        ("Memory", f"{quality['memory_usage']} MB", "💾")
    ]
    
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 2rem;">{icon}</span>
                <span class="metric-number">{value}</span>
                <span class="metric-label">{label}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Data preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="ipl_data.csv" style="background: #FF6B35; color: white; padding: 0.5rem 2rem; border-radius: 50px; text-decoration: none; display: inline-block;">📥 Download Data</a>'
        st.markdown(href, unsafe_allow_html=True)

def show_eda():
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    df = st.session_state.df
    
    # Use button to trigger EDA - prevents running on every rerun
    if st.button("🚀 Run EDA", type="primary"):
        with st.spinner("Running EDA..."):
            time.sleep(1)  # Visual feedback
            
            # Analysis type selector
            analysis_type = st.radio("Select Analysis", 
                                    ["Univariate", "Bivariate", "Correlation"],
                                    horizontal=True)
            
            if analysis_type == "Univariate":
                st.markdown("### 📈 Univariate Analysis")
                
                # Limit to first 5 numeric columns for performance
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:5]
                
                if numeric_cols:
                    selected_col = st.selectbox("Select column", numeric_cols)
                    fig = create_distribution_plot(df, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Bivariate":
                st.markdown("### 🔗 Bivariate Analysis")
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis", numeric_cols[:5], index=0)
                    with col2:
                        y_col = st.selectbox("Y-axis", numeric_cols[:5], index=min(1, 4))
                    
                    if STATSMODELS_AVAILABLE:
                        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                                       color_discrete_sequence=['#FF6B35'])
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col,
                                       color_discrete_sequence=['#FF6B35'])
                    
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    corr = df[x_col].corr(df[y_col])
                    st.metric("Correlation", f"{corr:.3f}")
            
            else:  # Correlation
                st.markdown("### 🔥 Correlation Analysis")
                fig = create_correlation_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns")

def show_models():
    st.markdown("## 🤖 Machine Learning Models")
    
    df = st.session_state.df
    
    # Target selection
    available_targets = ['winner', 'batter_runs', 'total_score']
    target_options = [col for col in available_targets if col in df.columns]
    
    if not target_options:
        st.error("No suitable target columns found")
        return
    
    target_col = st.selectbox("🎯 Select Target", target_options)
    problem_type = "classification" if target_col == 'winner' else "regression"
    
    # Feature selection - limit to 15 features max for performance
    df_encoded, encoders = encode_dataframe(df, target_col)
    st.session_state.encoders = encoders
    
    exclude_cols = [target_col]
    feature_options = [col for col in df_encoded.columns if col not in exclude_cols][:15]
    
    selected_features = st.multiselect("🔧 Select Features", 
                                      options=feature_options,
                                      default=feature_options[:min(5, len(feature_options))])
    
    if not selected_features:
        st.warning("Select at least one feature")
        return
    
    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 10, 40, 20) / 100
    with col2:
        if problem_type == "classification":
            model_choice = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        else:
            model_choice = st.selectbox("Model", ["Random Forest", "Linear Regression"])
    
    # Train button - only runs when clicked
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            X = df_encoded[selected_features]
            y = df_encoded[target_col]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if problem_type == "classification" else None
            )
            
            # Scale for linear models
            if model_choice in ['Logistic Regression', 'Linear Regression']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Train
            if problem_type == "classification":
                model, y_pred, accuracy, cm, importance = train_classification_model(
                    X_train, X_test, y_train, y_test, model_choice
                )
                
                st.session_state.current_model = model
                st.session_state.model_type = 'classification'
                
                st.markdown('<div class="success-message">✅ Model trained successfully!</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("CV Score", f"{cross_val_score(model, X, y, cv=5).mean():.3f}")
                
                # Confusion matrix
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=['Pred No', 'Pred Yes'], y=['Actual No', 'Actual Yes'],
                    colorscale='Viridis', text=cm, texttemplate='%{text}'))
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                model, y_pred, mae, r2, importance = train_regression_model(
                    X_train, X_test, y_train, y_test, model_choice
                )
                
                st.session_state.current_model = model
                st.session_state.model_type = 'regression'
                
                st.markdown('<div class="success-message">✅ Model trained successfully!</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:.3f}")
                col2.metric("R²", f"{r2:.3f}")
                
                # Regression plot
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                        marker_color='#FF6B35'), row=1, col=1)
                fig.add_trace(go.Histogram(x=y_test - y_pred, marker_color='#00D4FF'), row=1, col=2)
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance - limit to top 10
            if importance is not None:
                st.markdown("#### Top Features")
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importance
                }).sort_values('importance', ascending=False).head(10)
                
                for _, row in importance_df.iterrows():
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{row['feature'][:30]}</span>
                            <span style="color: #FF6B35;">{row['importance']*100:.1f}%</span>
                        </div>
                        <div style="height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                            <div style="height: 10px; width: {row['importance']*100}%; 
                                      background: linear-gradient(90deg, #FF6B35, #00D4FF); 
                                      border-radius: 5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state.feature_importance = importance_df
                st.session_state.selected_features = selected_features
            
            st.session_state.model_trained = True

def show_predictions():
    st.markdown("## 📈 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train a model first in the Models section")
        return
    
    st.markdown("### Enter values")
    
    # Create input fields
    input_data = {}
    
    for feature in st.session_state.selected_features[:5]:  # Limit to 5 features for UI
        if feature in st.session_state.df.columns:
            if st.session_state.df[feature].dtype in ['int64', 'float64']:
                min_val = float(st.session_state.df[feature].min())
                max_val = float(st.session_state.df[feature].max())
                mean_val = float(st.session_state.df[feature].mean())
                
                input_data[feature] = st.number_input(
                    feature, min_value=min_val, max_value=max_val, value=mean_val
                )
            else:
                options = st.session_state.df[feature].unique().tolist()[:10]
                input_data[feature] = st.selectbox(feature, options)
    
    if st.button("🔮 Predict", type="primary"):
        with st.spinner("Predicting..."):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            for col in input_df.select_dtypes(include='object').columns:
                if col in st.session_state.encoders:
                    input_df[col] = st.session_state.encoders[col].transform(
                        input_df[col].astype(str)
                    )
            
            input_df = input_df[st.session_state.selected_features]
            prediction = st.session_state.current_model.predict(input_df)[0]
            
            # Display
            if st.session_state.model_type == 'classification':
                unique_winners = st.session_state.df['winner'].unique() if 'winner' in st.session_state.df.columns else []
                if prediction < len(unique_winners):
                    pred_label = unique_winners[int(prediction)]
                else:
                    pred_label = f"Class {prediction}"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(0,212,255,0.2));
                            border: 2px solid #FF6B35; border-radius: 16px; padding: 2rem; text-align: center;">
                    <h2 style="color: #FF6B35;">🏆 {pred_label}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(0,212,255,0.2));
                            border: 2px solid #00D4FF; border-radius: 16px; padding: 2rem; text-align: center;">
                    <h2 style="color: #00D4FF;">📊 {prediction:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    main()
