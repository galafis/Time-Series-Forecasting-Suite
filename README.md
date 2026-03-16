<div align="center">

# Time Series Forecasting Suite

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-13%20passed-brightgreen?style=for-the-badge)](test_forecasting_suite.py)

Suite completa de previsao de series temporais com interface Streamlit e multiplos modelos.

Comprehensive time series forecasting suite with Streamlit interface and multiple models.

[Portugues](#portugues) | [English](#english)

</div>

---

## Portugues

### Sobre

Suite profissional de previsao de series temporais construida com Streamlit, oferecendo uma interface web interativa para upload de dados, treinamento de multiplos modelos e geracao de previsoes com exportacao de resultados. O sistema implementa quatro modelos de previsao: **Linear Regression** e **Random Forest** (via scikit-learn) para abordagem de machine learning com feature engineering temporal automatizado, e **ARIMA** e **Exponential Smoothing** (via statsmodels) para abordagem estatistica classica. Inclui decomposicao sazonal, comparacao automatica de modelos e exportacao de metricas.

### Tecnologias

| Tecnologia | Versao | Finalidade |
|------------|--------|------------|
| **Python** | 3.10+ | Linguagem principal |
| **Streamlit** | 1.28+ | Interface web interativa |
| **Pandas** | 1.3+ | Manipulacao de dados |
| **NumPy** | 1.21+ | Computacao numerica |
| **scikit-learn** | 1.0+ | Modelos de machine learning |
| **statsmodels** | 0.13+ | Modelos estatisticos (ARIMA, ETS) |
| **Plotly** | 5.0+ | Visualizacoes interativas |
| **pytest** | 7.0+ | Framework de testes |

### Arquitetura

```mermaid
graph TD
    A[Usuario] -->|Upload CSV / Dados padrao| B[Streamlit UI]
    B --> C[TimeSeriesForecaster]
    C --> D[Carregamento de Dados]
    D --> E[Preprocessamento]
    E --> F[Feature Engineering]
    E --> G[Decomposicao Sazonal]

    F --> H{Modelos ML}
    H --> I[Linear Regression]
    H --> J[Random Forest]

    E --> K{Modelos Estatisticos}
    K --> L[ARIMA]
    K --> M[Exponential Smoothing]

    I --> N[Avaliacao]
    J --> N
    L --> N
    M --> N

    N --> O[Comparacao de Modelos]
    N --> P[Geracao de Previsoes]
    P --> Q[Visualizacao Plotly]
    O --> Q
    Q --> R[Exportacao CSV/JSON]

    style C fill:#FF4B4B,color:#fff,stroke:#cc3c3c
    style H fill:#F7931E,color:#fff,stroke:#c57518
    style K fill:#3776AB,color:#fff,stroke:#2c5d88
```

### Fluxo de Previsao

```mermaid
sequenceDiagram
    participant U as Usuario
    participant UI as Streamlit UI
    participant F as TimeSeriesForecaster
    participant ML as Modelos ML
    participant ST as Modelos Estatisticos

    U->>UI: Upload CSV ou usar dados amostra
    UI->>F: load_data(data, date_col, value_col)
    F->>F: Preprocessar (interpolacao, freq diaria)
    F-->>UI: Exibir dados carregados

    UI->>F: create_components_plot()
    F->>F: analyze_components()
    F->>F: seasonal_decompose()
    F-->>UI: Grafico de componentes

    U->>UI: Clicar "Treinar e Prever"
    UI->>F: train_models(test_size=0.2)
    F->>F: create_features() - lags, rolling, temporal
    F->>ML: LinearRegression.fit(), RandomForest.fit()
    F->>ST: ARIMA.fit(), ExponentialSmoothing.fit()
    ML-->>F: Metricas train/test (MAE, RMSE)
    ST-->>F: Metricas train/test (MAE, RMSE)

    UI->>F: forecast(days, model_name)
    F->>F: Gerar features futuras
    F-->>UI: DataFrame com previsoes

    UI->>UI: Plotly chart + tabela de metricas
    UI->>UI: Comparacao de modelos (bar chart)
    U->>UI: Download CSV / JSON
```

### Estrutura do Projeto

```
Time-Series-Forecasting-Suite/
├── forecasting_suite.py       # App Streamlit + classe TimeSeriesForecaster (~593 linhas)
├── test_forecasting_suite.py  # Suite de testes com 13 testes (~173 linhas)
├── examples/
│   ├── README.md              # Guia dos datasets de exemplo
│   ├── sales_data.csv         # Vendas varejo com sazonalidade
│   ├── temperature_data.csv   # Dados climaticos
│   └── stock_price_data.csv   # Dados financeiros com volatilidade
├── requirements.txt           # Dependencias Python
├── Dockerfile                 # Containerizacao
├── CONTRIBUTING.md            # Guia para contribuidores
├── CHANGELOG.md               # Historico de versoes
├── LICENSE                    # Licenca MIT
└── README.md                  # Documentacao
```

### Inicio Rapido

```bash
# Clonar o repositorio
git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
cd Time-Series-Forecasting-Suite

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Executar a aplicacao
streamlit run forecasting_suite.py
```

### Docker

```bash
# Build da imagem
docker build -t forecasting-suite .

# Executar container
docker run -p 8501:8501 forecasting-suite

# Acessar em http://localhost:8501
```

### Testes

```bash
# Executar suite completa (13 testes)
python -m pytest test_forecasting_suite.py -v

# Com cobertura
python -m pytest test_forecasting_suite.py --cov=forecasting_suite --cov-report=html

# Teste especifico
python -m pytest test_forecasting_suite.py::test_train_models -v
```

### Uso Programatico

```python
from forecasting_suite import TimeSeriesForecaster

# Inicializar
forecaster = TimeSeriesForecaster()

# Carregar dados (ou usar dados amostra)
df = forecaster.load_data()

# Treinar todos os modelos
models, metrics = forecaster.train_models(test_size=0.2)

# Gerar previsao
forecast = forecaster.forecast(days=30, model_name='Random Forest')

# Visualizar
fig = forecaster.create_forecast_plot('Random Forest')
fig.show()
```

### Benchmarks

| Operacao | Tempo Medio | Dataset |
|----------|-------------|---------|
| Carregamento + preprocessamento | ~100 ms | 1826 dias |
| Feature engineering | ~200 ms | 1826 dias |
| Treino Linear Regression | ~50 ms | 80% split |
| Treino Random Forest | ~2 s | 80% split, 100 arvores |
| Treino ARIMA(5,1,0) | ~3 s | 80% split |
| Treino Exponential Smoothing | ~1 s | 80% split |
| Previsao 30 dias | ~100 ms | Random Forest |
| Decomposicao sazonal | ~500 ms | 1826 dias |

### Aplicabilidade

| Setor | Caso de Uso | Descricao |
|-------|-------------|-----------|
| **Financas** | Projecao de ativos | Previsao de precos com Random Forest e ARIMA |
| **Varejo** | Previsao de vendas | Planejamento de estoque com sazonalidade detectada |
| **Meteorologia** | Projecao climatica | Analise de tendencias e padroes ciclicos |
| **Manufatura** | Previsao de demanda | Dimensionamento de producao com multiplos horizontes |
| **Saude** | Series epidemiologicas | Modelagem de curvas de incidencia |
| **Energia** | Consumo futuro | Projecao de carga para planejamento de rede |

---

## English

### About

Professional time series forecasting suite built with Streamlit, offering an interactive web interface for data upload, multi-model training and forecast generation with results export. The system implements four forecasting models: **Linear Regression** and **Random Forest** (via scikit-learn) for a machine learning approach with automated temporal feature engineering, and **ARIMA** and **Exponential Smoothing** (via statsmodels) for classical statistical approaches. Includes seasonal decomposition, automatic model comparison and metrics export.

### Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core language |
| **Streamlit** | 1.28+ | Interactive web interface |
| **Pandas** | 1.3+ | Data manipulation |
| **NumPy** | 1.21+ | Numerical computing |
| **scikit-learn** | 1.0+ | Machine learning models |
| **statsmodels** | 0.13+ | Statistical models (ARIMA, ETS) |
| **Plotly** | 5.0+ | Interactive visualizations |
| **pytest** | 7.0+ | Testing framework |

### Architecture

```mermaid
graph TD
    A[User] -->|Upload CSV / Default data| B[Streamlit UI]
    B --> C[TimeSeriesForecaster]
    C --> D[Data Loading]
    D --> E[Preprocessing]
    E --> F[Feature Engineering]
    E --> G[Seasonal Decomposition]

    F --> H{ML Models}
    H --> I[Linear Regression]
    H --> J[Random Forest]

    E --> K{Statistical Models}
    K --> L[ARIMA]
    K --> M[Exponential Smoothing]

    I --> N[Evaluation]
    J --> N
    L --> N
    M --> N

    N --> O[Model Comparison]
    N --> P[Forecast Generation]
    P --> Q[Plotly Visualization]
    O --> Q
    Q --> R[CSV/JSON Export]

    style C fill:#FF4B4B,color:#fff,stroke:#cc3c3c
    style H fill:#F7931E,color:#fff,stroke:#c57518
    style K fill:#3776AB,color:#fff,stroke:#2c5d88
```

### Forecasting Flow

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant F as TimeSeriesForecaster
    participant ML as ML Models
    participant ST as Statistical Models

    U->>UI: Upload CSV or use sample data
    UI->>F: load_data(data, date_col, value_col)
    F->>F: Preprocess (interpolation, daily freq)
    F-->>UI: Display loaded data

    UI->>F: create_components_plot()
    F->>F: analyze_components()
    F->>F: seasonal_decompose()
    F-->>UI: Components chart

    U->>UI: Click "Train & Forecast"
    UI->>F: train_models(test_size=0.2)
    F->>F: create_features() - lags, rolling, temporal
    F->>ML: LinearRegression.fit(), RandomForest.fit()
    F->>ST: ARIMA.fit(), ExponentialSmoothing.fit()
    ML-->>F: Train/test metrics (MAE, RMSE)
    ST-->>F: Train/test metrics (MAE, RMSE)

    UI->>F: forecast(days, model_name)
    F->>F: Generate future features
    F-->>UI: Forecast DataFrame

    UI->>UI: Plotly chart + metrics table
    UI->>UI: Model comparison (bar chart)
    U->>UI: Download CSV / JSON
```

### Project Structure

```
Time-Series-Forecasting-Suite/
├── forecasting_suite.py       # Streamlit app + TimeSeriesForecaster class (~593 lines)
├── test_forecasting_suite.py  # Test suite with 13 tests (~173 lines)
├── examples/
│   ├── README.md              # Example datasets guide
│   ├── sales_data.csv         # Retail sales with seasonality
│   ├── temperature_data.csv   # Climate data
│   └── stock_price_data.csv   # Financial data with volatility
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization
├── CONTRIBUTING.md            # Developer guide
├── CHANGELOG.md               # Version history
├── LICENSE                    # MIT License
└── README.md                  # Documentation
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
cd Time-Series-Forecasting-Suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run forecasting_suite.py
```

### Docker

```bash
# Build image
docker build -t forecasting-suite .

# Run container
docker run -p 8501:8501 forecasting-suite

# Access at http://localhost:8501
```

### Tests

```bash
# Run full suite (13 tests)
python -m pytest test_forecasting_suite.py -v

# With coverage
python -m pytest test_forecasting_suite.py --cov=forecasting_suite --cov-report=html

# Specific test
python -m pytest test_forecasting_suite.py::test_train_models -v
```

### Programmatic Usage

```python
from forecasting_suite import TimeSeriesForecaster

# Initialize
forecaster = TimeSeriesForecaster()

# Load data (or use sample data)
df = forecaster.load_data()

# Train all models
models, metrics = forecaster.train_models(test_size=0.2)

# Generate forecast
forecast = forecaster.forecast(days=30, model_name='Random Forest')

# Visualize
fig = forecaster.create_forecast_plot('Random Forest')
fig.show()
```

### Benchmarks

| Operation | Avg Time | Dataset |
|-----------|----------|---------|
| Loading + preprocessing | ~100 ms | 1826 days |
| Feature engineering | ~200 ms | 1826 days |
| Linear Regression training | ~50 ms | 80% split |
| Random Forest training | ~2 s | 80% split, 100 trees |
| ARIMA(5,1,0) training | ~3 s | 80% split |
| Exponential Smoothing training | ~1 s | 80% split |
| 30-day forecast | ~100 ms | Random Forest |
| Seasonal decomposition | ~500 ms | 1826 days |

### Applicability

| Sector | Use Case | Description |
|--------|----------|-------------|
| **Finance** | Asset projection | Price forecasting with Random Forest and ARIMA |
| **Retail** | Sales forecasting | Inventory planning with detected seasonality |
| **Meteorology** | Climate projection | Trend analysis and cyclic patterns |
| **Manufacturing** | Demand forecasting | Production sizing with multiple horizons |
| **Healthcare** | Epidemiological series | Incidence curve modeling |
| **Energy** | Future consumption | Load projection for grid planning |

---

## Autor / Author

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

## Licenca / License

MIT License - veja [LICENSE](LICENSE) para detalhes / see [LICENSE](LICENSE) for details.
