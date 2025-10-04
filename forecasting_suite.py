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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

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
        
        self.data = self.data.sort_values('date').set_index('date')
        self.data = self.data.asfreq('D') # Ensure daily frequency
        self.data['value'] = self.data['value'].interpolate(method='linear') # Handle missing values
        return self.data
    
    def analyze_components(self):
        """Analyze time series components using seasonal decomposition."""
        if self.data is None or self.data.empty:
            raise ValueError("Data not loaded. Please load data first.")
        
        # Ensure data has enough periods for decomposition
        if len(self.data) < 2 * 365: # At least two years for yearly seasonality
            st.warning("Not enough data for meaningful seasonal decomposition. Using simplified components.")
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
        else:
            # Perform seasonal decomposition
            try:
                decomposition = seasonal_decompose(self.data['value'], model='additive', period=365)
                self.data['trend'] = decomposition.trend
                self.data['seasonal'] = decomposition.seasonal
                self.data['residual'] = decomposition.resid
            except Exception as e:
                st.error(f"Error during seasonal decomposition: {e}. Using simplified components.")
                # Fallback to simplified components if decomposition fails
                self.data['ma_7'] = self.data['value'].rolling(window=7).mean()
                self.data['ma_30'] = self.data['value'].rolling(window=30).mean()
                self.data['ma_365'] = self.data['value'].rolling(window=365).mean()
                self.data['trend'] = self.data['ma_365']
                self.data['seasonal'] = self.data['value'] - self.data['trend']
                self.data['residual'] = self.data['value'] - self.data['ma_7']
        
        return self.data
    
    def create_features(self):
        """Create features for machine learning models."""
        df = self.data.copy()
        df = df.reset_index() # Reset index to use date column as a feature
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        
        # Drop rows with NaN values created by feature engineering
        df = df.dropna()
        
        return df.set_index('date') # Set date back as index
    
    def fit_arima(self, data, order=(5,1,0), seasonal_order=(0,0,0,0)):
        """Fit an ARIMA model."""
        model = ARIMA(data["value"], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        self.models["ARIMA"] = model_fit
        return model_fit
    def fit_exponential_smoothing(self, data, seasonal='add', seasonal_periods=7):
        """Fit an Exponential Smoothing model."""
        model = ExponentialSmoothing(data["value"], seasonal_periods=seasonal_periods, trend='add', seasonal=seasonal)
        model_fit = model.fit()
        self.models["Exponential Smoothing"] = model_fit
        return model_fit
    def split_data(self, df, test_size=0.2):
        """Split data into training and testing sets."""
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df

    def train_models(self, test_size=0.2):
        """Train multiple forecasting models."""
        # Prepare data for ML models
        df_ml = self.create_features()
        
        # Define features and target for ML models
        feature_cols = [col for col in df_ml.columns if col not in ['value']]
        X = df_ml[feature_cols]
        y = df_ml['value']
        
        # Split data for ML models
        split_idx = int(len(df_ml) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train ML models
        ml_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in ml_models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
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
            
        # Train ARIMA model
        try:
            arima_model_fit = self.fit_arima(self.data.loc[y_train.index])
            arima_forecast_train = arima_model_fit.predict(start=y_train.index[0], end=y_train.index[-1])
            arima_forecast_test = arima_model_fit.predict(start=y_test.index[0], end=y_test.index[-1])
            
            arima_train_mae = mean_absolute_error(y_train, arima_forecast_train)
            arima_test_mae = mean_absolute_error(y_test, arima_forecast_test)
            arima_train_rmse = np.sqrt(mean_squared_error(y_train, arima_forecast_train))
            arima_test_rmse = np.sqrt(mean_squared_error(y_test, arima_forecast_test))
            
            self.metrics['ARIMA'] = {
                'train_mae': arima_train_mae,
                'test_mae': arima_test_mae,
                'train_rmse': arima_train_rmse,
                'test_rmse': arima_test_rmse
            }
        except Exception as e:
            st.warning(f"ARIMA model training failed: {e}")
            
        # Train Exponential Smoothing model
        try:
            es_model_fit = self.fit_exponential_smoothing(self.data.loc[y_train.index])
            es_forecast_train = es_model_fit.predict(start=y_train.index[0], end=y_train.index[-1])
            es_forecast_test = es_model_fit.predict(start=y_test.index[0], end=y_test.index[-1])
            
            es_train_mae = mean_absolute_error(y_train, es_forecast_train)
            es_test_mae = mean_absolute_error(y_test, es_forecast_test)
            es_train_rmse = np.sqrt(mean_squared_error(y_train, es_forecast_train))
            es_test_rmse = np.sqrt(mean_squared_error(y_test, es_forecast_test))
            
            self.metrics['Exponential Smoothing'] = {
                'train_mae': es_train_mae,
                'test_mae': es_test_mae,
                'train_rmse': es_train_rmse,
                'test_rmse': es_test_rmse
            }
        except Exception as e:
            st.warning(f"Exponential Smoothing model training failed: {e}")
        
        return self.models, self.metrics
    
    def forecast(self, days=30, model_name='Random Forest'):
        """Generate forecasts."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        if model_name in ['Linear Regression', 'Random Forest']:
            # Get last known data point and create future features
            last_date = self.data.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
            
            # Create a DataFrame for future features
            future_df = pd.DataFrame(index=future_dates)
            future_df['value'] = np.nan # Placeholder for value
            
            # Combine historical data with future dates to generate features
            combined_df = pd.concat([self.data, future_df])
            combined_df = combined_df.reset_index().rename(columns={'index': 'date'}) # Reset index for feature creation
            combined_df = self.create_features_for_forecast(combined_df) # New helper function
            combined_df = combined_df.set_index('date')
            
            # Filter for future features
            X_pred = combined_df.loc[future_dates]
            feature_cols = [col for col in X_pred.columns if col not in ['value']]
            X_pred = X_pred[feature_cols].dropna() # Drop NaNs from feature creation
            
            if X_pred.empty:
                st.warning("Could not generate future features for ML models. Forecast might be empty.")
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecast': np.nan
                })
            else:
                forecasts = model.predict(X_pred)
                forecast_df = pd.DataFrame({
                    'date': X_pred.index,
                    'forecast': forecasts
                })
        
        elif model_name == 'ARIMA':
            forecast_result = model.forecast(steps=days)
            forecast_df = pd.DataFrame({
                'date': forecast_result.index,
                'forecast': forecast_result.values
            })
        
        elif model_name == 'Exponential Smoothing':
            forecast_result = model.forecast(steps=days)
            forecast_df = pd.DataFrame({
                'date': forecast_result.index,
                'forecast': forecast_result.values
            })
        
        else:
            raise ValueError("Unsupported model for forecasting.")
        
        self.forecasts[model_name] = forecast_df
        return forecast_df

    def create_features_for_forecast(self, df):
        """Helper to create features for forecasting future values with ML models."""
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week
        
        # Lag features (fill NaNs for future dates with last known value)
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df['value'].shift(lag).ffill()
        
        # Rolling statistics (fill NaNs for future dates with last known value)
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean().ffill()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std().ffill()
        return df

    def create_forecast_plot(self, model_name='Random Forest', days_history=90):
        """Create interactive forecast plot."""
        if model_name not in self.forecasts:
            self.forecast(model_name=model_name)
        
        # Get recent historical data
        recent_data = self.data.tail(days_history).reset_index()
        forecast_data = self.forecasts[model_name].reset_index()
        
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
        last_date = self.data.index.max()
#        fig.add_vline(x=last_date.to_pydatetime(), line_dash="dot", line_color="gray", 
#                     annotation_text="Forecast Start")
        
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
            x=self.data.index,
            y=self.data['value'],
            mode='lines',
            name='Original',
            line=dict(color='blue')
        ))
        
        # Trend
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='red')
        ))
        
        # Seasonal (if available)
        if 'seasonal' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['seasonal'],
                mode='lines',
                name='Seasonal',
                line=dict(color='green')
            ))
        
        # Residual (if available)
        if 'residual' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['residual'],
                mode='lines',
                name='Residual',
                line=dict(color='purple')
            ))
        
        fig.update_layout(
            title='Time Series Components',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            height=500
        )
        
        return fig

    def create_metrics_table(self):
        """Create a table of model metrics."""
        metrics_data = []
        for model_name, metrics in self.metrics.items():
            metrics_data.append({
                'Model': model_name,
                'Train MAE': f"{metrics['train_mae']:.2f}",
                'Test MAE': f"{metrics['test_mae']:.2f}",
                'Train RMSE': f"{metrics['train_rmse']:.2f}",
                'Test RMSE': f"{metrics['test_rmse']:.2f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        return metrics_df


# Streamlit Application
def main():
    st.set_page_config(layout="wide", page_title="Time Series Forecasting Suite")
    st.title("📈 Time Series Forecasting Suite")

    forecaster = TimeSeriesForecaster()

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        date_col = st.sidebar.selectbox("Select Date Column", data.columns)
        value_col = st.sidebar.selectbox("Select Value Column", [col for col in data.columns if col != date_col])
        forecaster.load_data(data, date_column=date_col, value_column=value_col)
        st.sidebar.success("Data loaded successfully!")
    else:
        st.sidebar.info("No CSV uploaded. Using sample data.")
        forecaster.load_data() # Load sample data if no file is uploaded

    if forecaster.data is not None:
        st.subheader("Raw Data Preview")
        st.write(forecaster.data.head())

        st.subheader("Time Series Components")
        st.plotly_chart(forecaster.create_components_plot(), use_container_width=True)

        st.sidebar.header("Model Training")
        test_size = st.sidebar.slider("Test Size (proportion)", 0.1, 0.5, 0.2, 0.05)
        if st.sidebar.button("Train Models & Generate Forecast"):
            with st.spinner("Training models and generating forecasts..."):
                forecaster.train_models(test_size=test_size)
                st.sidebar.success("Models trained and forecasts generated!")

            st.subheader("Model Performance Metrics")
            st.dataframe(forecaster.create_metrics_table())

            st.subheader("Forecast Visualization")
            model_choice = st.selectbox("Select Model for Forecast Visualization", list(forecaster.models.keys()))
            days_to_forecast = st.slider("Days to Forecast", 7, 365, 30, 7)
            days_history = st.slider("Days of History to Display", 30, 365, 90, 30)
            
            if model_choice and forecaster.models:
                forecaster.forecast(days=days_to_forecast, model_name=model_choice)
                st.plotly_chart(forecaster.create_forecast_plot(model_name=model_choice, days_history=days_history), use_container_width=True)
            else:
                st.warning("Please train models first.")

if __name__ == "__main__":
    main()
