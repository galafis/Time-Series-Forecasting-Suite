import pandas as pd
from forecasting_suite import TimeSeriesForecaster

def test_load_data():
    forecaster = TimeSeriesForecaster()
    df = forecaster.load_data()
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'value' in df.columns

def test_create_features():
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    df_features = forecaster.create_features()
    assert not df_features.empty
    assert 'dayofweek' in df_features.columns
    assert 'quarter' in df_features.columns
    assert 'month' in df_features.columns
    assert 'year' in df_features.columns
    assert 'dayofyear' in df_features.columns
    assert 'weekofyear' in df_features.columns
    assert 'lag_1' in df_features.columns
    assert 'rolling_mean_7' in df_features.columns

def test_train_models():
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    models, metrics = forecaster.train_models()
    assert 'Linear Regression' in models
    assert 'Random Forest' in models
    assert 'ARIMA' in models
    assert 'Exponential Smoothing' in models
    assert 'Linear Regression' in metrics
    assert 'Random Forest' in metrics
    assert 'ARIMA' in metrics
    assert 'Exponential Smoothing' in metrics

def test_forecast():
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    forecaster.train_models()
    
    # Test ML model forecast
    forecast_rf = forecaster.forecast(days=30, model_name='Random Forest')
    assert not forecast_rf.empty
    assert 'forecast' in forecast_rf.columns
    
    # Test ARIMA forecast
    forecast_arima = forecaster.forecast(days=30, model_name='ARIMA')
    assert not forecast_arima.empty
    assert 'forecast' in forecast_arima.columns
    
    # Test Exponential Smoothing forecast
    forecast_es = forecaster.forecast(days=30, model_name='Exponential Smoothing')
    assert not forecast_es.empty
    assert 'forecast' in forecast_es.columns

