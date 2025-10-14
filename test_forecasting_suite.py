import pandas as pd
import numpy as np
import pytest
from forecasting_suite import TimeSeriesForecaster


def test_load_data():
    forecaster = TimeSeriesForecaster()
    df = forecaster.load_data()
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'value' in df.columns


def test_load_custom_data():
    """Test loading custom CSV data."""
    forecaster = TimeSeriesForecaster()
    # Create sample custom data
    custom_data = pd.DataFrame({
        'custom_date': pd.date_range(start='2023-01-01', periods=100),
        'custom_value': np.random.randn(100) + 100
    })
    df = forecaster.load_data(custom_data, date_column='custom_date', value_column='custom_value')
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'value' in df.columns
    assert len(df) == 100


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


def test_analyze_components():
    """Test time series decomposition."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    result = forecaster.analyze_components()
    assert 'trend' in result.columns
    assert 'seasonal' in result.columns
    assert 'residual' in result.columns


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


def test_metrics_values():
    """Test that metrics have proper values."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    models, metrics = forecaster.train_models()
    for model_name, metric in metrics.items():
        assert 'train_mae' in metric
        assert 'test_mae' in metric
        assert 'train_rmse' in metric
        assert 'test_rmse' in metric
        assert metric['train_mae'] >= 0
        assert metric['test_mae'] >= 0
        assert metric['train_rmse'] >= 0
        assert metric['test_rmse'] >= 0


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


def test_forecast_different_horizons():
    """Test forecasting with different time horizons."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    forecaster.train_models()

    # Test different forecast horizons
    for days in [7, 30, 90]:
        forecast = forecaster.forecast(days=days, model_name='Random Forest')
        assert len(forecast) <= days  # May be less due to feature engineering


def test_forecast_without_training():
    """Test that forecast raises error if model not trained."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()

    with pytest.raises(ValueError, match="Model .* not trained"):
        forecaster.forecast(days=30, model_name='Random Forest')


def test_create_forecast_plot():
    """Test forecast plot creation."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    forecaster.train_models()
    forecaster.forecast(days=30, model_name='Random Forest')

    fig = forecaster.create_forecast_plot(model_name='Random Forest')
    assert fig is not None
    assert len(fig.data) >= 2  # At least historical and forecast traces


def test_create_components_plot():
    """Test components plot creation."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()

    fig = forecaster.create_components_plot()
    assert fig is not None
    assert len(fig.data) >= 2  # At least original and trend


def test_split_data():
    """Test data splitting functionality."""
    forecaster = TimeSeriesForecaster()
    forecaster.load_data()
    df_features = forecaster.create_features()

    train_df, test_df = forecaster.split_data(df_features, test_size=0.2)
    total_len = len(train_df) + len(test_df)
    assert abs(len(test_df) / total_len - 0.2) < 0.01  # Check split ratio


def test_data_preprocessing():
    """Test that data preprocessing handles missing values."""
    forecaster = TimeSeriesForecaster()
    # Create data with missing values
    dates = pd.date_range(start='2023-01-01', periods=100)
    values = np.random.randn(100) + 100
    values[10:15] = np.nan  # Add missing values

    data = pd.DataFrame({'date': dates, 'value': values})
    df = forecaster.load_data(data)

    # Check that missing values are filled
    assert not df['value'].isna().any()
