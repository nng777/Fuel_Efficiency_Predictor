import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from fuel_efficiency_predictor import FuelEfficiencyPredictor

# Configure page
st.set_page_config(
    page_title="Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictor():
    """Load and initialize the predictor with cached data."""
    predictor = FuelEfficiencyPredictor()
    predictor.load_data()
    predictor.prepare_data()
    predictor.train_models()
    predictor.evaluate_models()
    return predictor

def show_home_page(predictor):
    st.markdown('<h2 class="section-header">Welcome to the Fuel Efficiency Predictor!</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This interactive application teaches you machine learning fundamentals using a **real-world problem**: 
        predicting car fuel efficiency (MPG - Miles Per Gallon).
        
        ### What you'll learn:
        - 📊 **Data Exploration**: Understanding your dataset
        - 🔧 **Data Preprocessing**: Preparing data for machine learning
        - 🤖 **Model Training**: Building Linear Regression and Random Forest models
        - 📏 **Model Evaluation**: Comparing model performance
        - 🔮 **Making Predictions**: Using trained models on new data
        
        ### Dataset Features:
        - **Cylinders**: Number of engine cylinders
        - **Displacement**: Engine displacement (cubic inches)
        - **Horsepower**: Engine horsepower
        - **Weight**: Vehicle weight (pounds)
        - **Acceleration**: Time to accelerate 0-60 mph
        - **Model Year**: Year of manufacture (70-82 represents 1970-1982)
        
        **Target Variable**: MPG (Miles Per Gallon)
        """)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Size", f"{len(predictor.data)} cars")
        st.metric("Features", len(predictor.feature_names))
        st.metric("Models Trained", len(predictor.models))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("### Quick Dataset Stats")
        st.write(f"**Average MPG**: {predictor.data['mpg'].mean():.1f}")
        st.write(f"**MPG Range**: {predictor.data['mpg'].min():.1f} - {predictor.data['mpg'].max():.1f}")
        st.write(f"**Most Common Cylinders**: {predictor.data['cylinders'].mode()[0]}")

def show_predictions(predictor):
    st.markdown('<h2 class="section-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    st.markdown("### Enter car specifications to predict fuel efficiency:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cylinders = st.selectbox("Number of Cylinders:", [4, 6, 8], index=0)
        displacement = st.slider("Engine Displacement (cubic inches):", 100, 400, 200)
        horsepower = st.slider("Horsepower:", 60, 250, 120)
    
    with col2:
        weight = st.slider("Weight (pounds):", 1800, 4500, 2800)
        acceleration = st.slider("Acceleration (0-60 mph time in seconds):", 8.0, 25.0, 15.0)
        model_year = st.slider("Model Year:", 70, 82, 78)
        st.caption("Note: 70-82 represents 1970-1982")
    
    # Create car features dictionary
    car_features = {
        'cylinders': cylinders,
        'displacement': displacement,
        'horsepower': horsepower,
        'weight': weight,
        'acceleration': acceleration,
        'model_year': model_year
    }
    
    # Make predictions
    st.subheader("Predictions:")
    
    col1, col2 = st.columns(2)
    
    # Linear Regression prediction
    with col1:
        new_car_df = pd.DataFrame([car_features])
        new_car_scaled = predictor.scaler.transform(new_car_df)
        lr_prediction = predictor.models['Linear Regression'].predict(new_car_scaled)[0]
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Linear Regression", f"{lr_prediction:.1f} MPG")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Random Forest prediction
    with col2:
        rf_prediction = predictor.models['Random Forest'].predict(new_car_df)[0]
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Random Forest", f"{rf_prediction:.1f} MPG")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Average prediction
    avg_prediction = (lr_prediction + rf_prediction) / 2
    st.markdown(f"### 🎯 Average Prediction: **{avg_prediction:.1f} MPG**")

def main():
    # Title
    st.markdown('<h1 class="main-header">🚗 Fuel Efficiency Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Learn Machine Learning with Real-World Car Data!")
    
    # Initialize predictor
    with st.spinner("Loading models and data..."):
        predictor = load_predictor()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "🔮 Make Predictions"]
    )
    
    if page == "🏠 Home":
        show_home_page(predictor)
    elif page == "🔮 Make Predictions":
        show_predictions(predictor)

if __name__ == "__main__":
    main() 