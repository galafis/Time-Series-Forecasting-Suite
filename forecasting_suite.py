#!/usr/bin/env python3
"""
Time Series Forecasting Suite
Comprehensive time series analysis and forecasting toolkit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Comprehensive time series forecasting toolkit."""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
    
    def load_data(self, data=None, date_column='date', value_column='value'):
        """Load time series data."""
        if data is None:
            # Generate sample data
            dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
            trend = np.linspace(100, 200, len(dates))
            seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            noise = np.random.normal(0, 5, len(dates))
            values = trend + seasonal + noise
            
            self.data = pd.DataFrame({
                'date': dates,
                'value': values
            })
        else:
            self.data = data.copy()
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data = self.data.rename(columns={date_column: 'date', value_column: 'value'})
        
        self.data = self.data.sort_values('date').reset_index(drop=True)
        return self.data
    
    def analyze_components(self):
        """Analyze time series components."""
        # Calculate moving averages
        self.data['ma_7'] = self.data['value'].rolling(window=7).mean()
        self.data['ma_30'] = self.data['value'].rolling(window=30).mean()
        self.data['ma_365'] = self.data['value'].rolling(window=365).mean()
        
        # Calculate differences
        self.data['diff_1'] = self.data['value'].diff()
        self.data['diff_7'] = self.data['value'].diff(7)
        
        # Seasonal decomposition (simplified)
        self.data['trend'] = self.data['ma_365']
        self.data['seasonal'] = self.data['value'] - self.data['trend']
        self.data['residual'] = self.data['value'] - self.data['ma_7']
        
        return self.data
    
    def create_features(self, forecast_days=30):
        """Create features for machine learning models."""
        df = self.data.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_models(self, test_size=0.2):
        """Train multiple forecasting models."""
        # Prepare data
        df = self.create_features()
        
        # Define features and target
        feature_cols = [col for col in df.columns if col not in ['date', 'value']]
        X = df[feature_cols]
        y = df['value']
        
        # Split data
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            self.models[name] = model
            self.metrics[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
        
        return self.models, self.metrics
    
    def forecast(self, days=30, model_name='Random Forest'):
        """Generate forecasts."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        # Get last known data point
        last_date = self.data['date'].max()
        last_data = self.create_features().iloc[-1:].copy()
        
        forecasts = []
        forecast_dates = []
        
        for i in range(days):
            # Predict next value
            feature_cols = [col for col in last_data.columns if col not in ['date', 'value']]
            X_pred = last_data[feature_cols]
            pred_value = model.predict(X_pred)[0]
            
            # Update date
            next_date = last_date + timedelta(days=i+1)
            forecast_dates.append(next_date)
            forecasts.append(pred_value)
            
            # Update features for next prediction (simplified)
            last_data['lag_1'] = pred_value
            last_data['month'] = next_date.month
            last_data['day'] = next_date.day
            last_data['dayofweek'] = next_date.weekday()
            last_data['dayofyear'] = next_date.timetuple().tm_yday
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts
        })
        
        self.forecasts[model_name] = forecast_df
        return forecast_df
    
    def create_forecast_plot(self, model_name='Random Forest', days_history=90):
        """Create interactive forecast plot."""
        if model_name not in self.forecasts:
            self.forecast(model_name=model_name)
        
        # Get recent historical data
        recent_data = self.data.tail(days_history)
        forecast_data = self.forecasts[model_name]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['value'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Add vertical line at forecast start
        last_date = self.data['date'].max()
        fig.add_vline(x=last_date, line_dash="dot", line_color="gray", 
                     annotation_text="Forecast Start")
        
        fig.update_layout(
            title=f'Time Series Forecast - {model_name}',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_components_plot(self):
        """Create time series components plot."""
        self.analyze_components()
        
        fig = go.Figure()
        
        # Original series
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data['value'],
            mode='lines',
            name='Original',
            line=dict(color='blue')
        ))
        
        # Trend
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='red')
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data['ma_30'],
            mode='lines',
            name='30-day MA',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Time Series Components',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            height=500
        )
        
        return fig

def main():
    """Streamlit application."""
    st.set_page_config(
        page_title="Time Series Forecasting Suite",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Time Series Forecasting Suite")
    st.markdown("Comprehensive time series analysis and forecasting toolkit")
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data options
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        data = forecaster.load_data()
        st.sidebar.success("Sample data loaded successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            date_col = st.sidebar.selectbox("Date Column", data.columns)
            value_col = st.sidebar.selectbox("Value Column", data.columns)
            data = forecaster.load_data(data, date_col, value_col)
            st.sidebar.success("Data loaded successfully!")
        else:
            st.warning("Please upload a CSV file or use sample data")
            return
    
    # Forecasting parameters
    forecast_days = st.sidebar.slider("Forecast Days", 1, 90, 30)
    model_choice = st.sidebar.selectbox("Model", ['Random Forest', 'Linear Regression'])
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Overview")
        st.write(f"**Data Points:** {len(data)}")
        st.write(f"**Date Range:** {data['date'].min().date()} to {data['date'].max().date()}")
        st.write(f"**Mean Value:** {data['value'].mean():.2f}")
        st.write(f"**Std Deviation:** {data['value'].std():.2f}")
    
    with col2:
        st.subheader("📈 Recent Values")
        recent_data = data.tail(5)[['date', 'value']]
        recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)
    
    # Train models
    if st.button("🚀 Train Models & Generate Forecast"):
        with st.spinner("Training models..."):
            models, metrics = forecaster.train_models()
            
            # Display metrics
            st.subheader("📊 Model Performance")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.round(3), use_container_width=True)
            
            # Generate forecast
            forecast_data = forecaster.forecast(days=forecast_days, model_name=model_choice)
            
            # Display forecast plot
            st.subheader(f"🔮 {forecast_days}-Day Forecast")
            forecast_plot = forecaster.create_forecast_plot(model_name=model_choice)
            st.plotly_chart(forecast_plot, use_container_width=True)
            
            # Display forecast values
            st.subheader("📋 Forecast Values")
            forecast_display = forecast_data.copy()
            forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
            forecast_display['forecast'] = forecast_display['forecast'].round(2)
            st.dataframe(forecast_display, use_container_width=True)
    
    # Components analysis
    st.subheader("🔍 Time Series Components")
    components_plot = forecaster.create_components_plot()
    st.plotly_chart(components_plot, use_container_width=True)
    
    # Raw data
    with st.expander("📋 View Raw Data"):
        display_data = data.copy()
        display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)

if __name__ == "__main__":
    main()

