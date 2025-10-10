# Time Series Forecasting Suite

[English](#english) | [Português](#português)

## English

### Overview
Comprehensive time series forecasting suite with multiple algorithms, interactive visualizations, and a professional web interface. This toolkit is designed for accurate time series prediction and analysis, featuring robust models like ARIMA, Exponential Smoothing, Linear Regression, and Random Forest, alongside advanced analytical capabilities.

### Features
- **Multiple Algorithms**: Implements ARIMA, Exponential Smoothing, Linear Regression, and Random Forest for diverse forecasting needs.
- **Interactive Dashboard**: A user-friendly web interface built with Streamlit, offering real-time updates and interactive controls.
- **Data Visualization**: Utilizes Plotly for professional, interactive charts, enabling detailed exploration of historical data and forecasts.
- **Forecast Accuracy Metrics**: Provides key evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for model performance assessment.
- **Seasonal Decomposition**: Analyzes time series into trend, seasonal, and residual components to understand underlying patterns.
- **Data Handling**: Supports uploading custom CSV files and includes a sample dataset for immediate use.
- **Export Results**: Allows downloading of forecast data and visualizations.
- **Model Comparison**: Facilitates side-by-side analysis of different algorithm performances.

### Technologies Used
- **Python 3.8+**
- **Streamlit**: For building the interactive web application.
- **Pandas**: Essential for data manipulation and analysis.
- **NumPy**: Provides powerful numerical computing capabilities.
- **Scikit-learn**: Offers various machine learning algorithms, including Linear Regression and RandomForestRegressor.
- **Plotly**: For creating rich, interactive data visualizations.
- **Statsmodels**: For statistical modeling, including ARIMA and Exponential Smoothing.

### Installation

To set up the Time Series Forecasting Suite, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
    cd Time-Series-Forecasting-Suite
    ```

2.  **Install dependencies**:
    Ensure you have `pip` installed. Then, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    Launch the Streamlit application from your terminal:
    ```bash
    streamlit run forecasting_suite.py
    ```

4.  **Access the application**:
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501` or `http://localhost:8502`).

### Usage

#### Web Interface
Once the application is running, you can interact with it through your web browser:

1.  **Data Input**: Choose to use the provided sample data or upload your own CSV file. If uploading, select the appropriate date and value columns.
2.  **Data Exploration**: Review the data overview, recent values, and time series components plots to understand your data.
3.  **Model Selection**: From the sidebar, select your preferred forecasting algorithm (e.g., Random Forest, ARIMA).
4.  **Generate Forecast**: Adjust the number of forecast days using the slider and click the "🚀 Train Models & Generate Forecast" button.
5.  **Evaluate Results**: Observe the model performance metrics (MAE, RMSE) and the interactive forecast plot.
6.  **Export**: Download the forecast data and charts as needed.

### Forecasting Models

#### ARIMA (AutoRegressive Integrated Moving Average)
-   **Description**: A statistical model that uses past values to predict future ones, suitable for data with trends and seasonality.
-   **Application**: Best for stationary time series with identifiable trends and seasonal patterns.

#### Exponential Smoothing
-   **Description**: A family of models that predict future values based on weighted averages of past observations, with weights decaying exponentially as observations get older.
-   **Application**: Effective for data with trend and/or seasonal components, offering simple, double, and triple (Holt-Winters) variations.

#### Machine Learning Models
-   **Linear Regression**: A straightforward model that predicts future values by fitting a linear equation to the observed data.
-   **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting. Excellent for capturing non-linear relationships.
-   **Feature Engineering**: Both Linear Regression and Random Forest models leverage engineered features such as lag values and rolling statistics to enhance prediction capabilities.

### Functionalities

#### Data Preprocessing
-   **Missing Value Handling**: Automatic interpolation (linear method) to fill gaps in the time series data.
-   **Frequency Enforcement**: Ensures data is uniformly sampled (e.g., daily frequency) for consistent analysis.

#### Visualization
-   **Interactive Time Series Plots**: Powered by Plotly, these charts allow zooming and panning for detailed data inspection.
-   **Forecast Plots**: Visual representation of predictions, showing historical data, forecasted values, and a clear demarcation of the forecast start.
-   **Time Series Components Plot**: Displays the decomposed trend, seasonal, and residual components of the time series.

#### Model Evaluation
-   **Accuracy Metrics**: Provides Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for both training and testing datasets to assess model accuracy.

### Sample Datasets
-   The application includes a built-in sample dataset for immediate demonstration and testing purposes.

### Contributing
We welcome contributions to enhance the Time Series Forecasting Suite! Please follow these guidelines:

1.  **Fork the repository**.
2.  **Create a feature branch** (`git checkout -b feature/your-feature-name`).
3.  **Commit your changes** (`git commit -am 'Add new feature'`).
4.  **Push to the branch** (`git push origin feature/your-feature-name`).
5.  **Create a Pull Request**.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Português

### Visão Geral
Suite abrangente de previsão de séries temporais com múltiplos algoritmos, visualizações interativas e uma interface web profissional. Este kit de ferramentas foi projetado para previsão e análise precisas de séries temporais, apresentando modelos robustos como ARIMA, Suavização Exponencial, Regressão Linear e Random Forest, juntamente com capacidades analíticas avançadas.

### Funcionalidades
-   **Múltiplos Algoritmos**: Implementa ARIMA, Suavização Exponencial, Regressão Linear e Random Forest para diversas necessidades de previsão.
-   **Dashboard Interativo**: Uma interface web amigável construída com Streamlit, oferecendo atualizações em tempo real e controles interativos.
-   **Visualização de Dados**: Utiliza Plotly para gráficos profissionais e interativos, permitindo a exploração detalhada de dados históricos e previsões.
-   **Métricas de Precisão de Previsão**: Fornece métricas de avaliação chave, como Erro Absoluto Médio (MAE) e Raiz do Erro Quadrático Médio (RMSE) para avaliação do desempenho do modelo.
-   **Decomposição Sazonal**: Analisa séries temporais em componentes de tendência, sazonalidade e resíduos para entender padrões subjacentes.
-   **Manipulação de Dados**: Suporta o upload de arquivos CSV personalizados e inclui um conjunto de dados de exemplo para uso imediato.
-   **Exportar Resultados**: Permite o download de dados de previsão e visualizações.
-   **Comparação de Modelos**: Facilita a análise lado a lado do desempenho de diferentes algoritmos.

### Tecnologias Utilizadas
-   **Python 3.8+**
-   **Streamlit**: Para a construção da aplicação web interativa.
-   **Pandas**: Essencial para manipulação e análise de dados.
-   **NumPy**: Fornece poderosas capacidades de computação numérica.
-   **Scikit-learn**: Oferece vários algoritmos de machine learning, incluindo Regressão Linear e RandomForestRegressor.
-   **Plotly**: Para a criação de visualizações de dados ricas e interativas.
-   **Statsmodels**: Para modelagem estatística, incluindo ARIMA e Suavização Exponencial.

### Instalação

Para configurar a Time Series Forecasting Suite, siga estes passos:

1.  **Clone o repositório**:
    ```bash
    git clone https://github.com/galafis/Time-Series-Forecasting-Suite.git
    cd Time-Series-Forecasting-Suite
    ```

2.  **Instale as dependências**:
    Certifique-se de ter o `pip` instalado. Em seguida, instale os pacotes Python necessários:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a aplicação**:
    Inicie a aplicação Streamlit a partir do seu terminal:
    ```bash
    streamlit run forecasting_suite.py
    ```

4.  **Execute os testes**:
    ```bash
    python3 -m pytest test_forecasting_suite.py
    ```

5.  **Acesse a aplicação**:
    Abra seu navegador e navegue para a URL local fornecida pelo Streamlit (geralmente `http://localhost:8501` ou `http://localhost:8502`).

### Uso

#### Interface Web
Uma vez que a aplicação esteja em execução, você pode interagir com ela através do seu navegador web:

1.  **Entrada de Dados**: Escolha usar os dados de exemplo fornecidos ou faça o upload do seu próprio arquivo CSV. Se estiver fazendo o upload, selecione as colunas de data e valor apropriadas.
2.  **Exploração de Dados**: Revise a visão geral dos dados, valores recentes e gráficos de componentes da série temporal para entender seus dados.
3.  **Seleção de Modelo**: Na barra lateral, selecione seu algoritmo de previsão preferido (por exemplo, Random Forest, ARIMA).
4.  **Gerar Previsão**: Ajuste o número de dias de previsão usando o controle deslizante e clique no botão "🚀 Treinar Modelos e Gerar Previsão".
5.  **Avaliar Resultados**: Observe as métricas de desempenho do modelo (MAE, RMSE) e o gráfico de previsão interativo.
6.  **Exportar**: Baixe os dados de previsão e os gráficos conforme necessário.

### Modelos de Previsão

#### ARIMA (AutoRegressive Integrated Moving Average)
-   **Descrição**: Um modelo estatístico que usa valores passados para prever futuros, adequado para dados com tendências e sazonalidade.
-   **Aplicação**: Melhor para séries temporais estacionárias com tendências e padrões sazonais identificáveis.

#### Suavização Exponencial
-   **Descrição**: Uma família de modelos que preveem valores futuros com base em médias ponderadas de observações passadas, com pesos decaindo exponencialmente à medida que as observações envelhecem.
-   **Aplicação**: Eficaz para dados com componentes de tendência e/ou sazonalidade, oferecendo variações simples, duplas e triplas (Holt-Winters).

#### Modelos de Machine Learning
-   **Regressão Linear**: Um modelo direto que prevê valores futuros ajustando uma equação linear aos dados observados.
-   **Random Forest**: Um método de aprendizado de conjunto que constrói múltiplas árvores de decisão e mescla suas previsões para melhorar a precisão e controlar o overfitting. Excelente para capturar relações não lineares.
-   **Engenharia de Features**: Ambos os modelos de Regressão Linear e Random Forest utilizam features engenheiradas, como valores de lag e estatísticas móveis, para aprimorar as capacidades de previsão.

### Funcionalidades

#### Pré-processamento de Dados
-   **Tratamento de Valores Ausentes**: Interpolação automática (método linear) para preencher lacunas nos dados da série temporal.
-   **Imposição de Frequência**: Garante que os dados sejam amostrados uniformemente (por exemplo, frequência diária) para análise consistente.

#### Visualização
-   **Gráficos Interativos de Séries Temporais**: Alimentados por Plotly, esses gráficos permitem zoom e panorâmica para inspeção detalhada dos dados.
-   **Gráficos de Previsão**: Representação visual das previsões, mostrando dados históricos, valores previstos e uma clara demarcação do início da previsão.
-   **Gráfico de Componentes da Série Temporal**: Exibe os componentes de tendência, sazonalidade e resíduos decompostos da série temporal.

#### Avaliação de Modelo
-   **Métricas de Precisão**: Fornece Erro Absoluto Médio (MAE) e Raiz do Erro Quadrático Médio (RMSE) para conjuntos de dados de treinamento e teste para avaliar a precisão do modelo.

### Datasets de Exemplo
-   A aplicação inclui um conjunto de dados de exemplo integrado para fins de demonstração e teste imediatos.

### Contribuindo
Aceitamos contribuições para aprimorar a Time Series Forecasting Suite! Por favor, siga estas diretrizes:

1.  **Faça um fork do repositório**.
2.  **Crie uma branch de feature** (`git checkout -b feature/seu-nome-da-feature`).
3.  **Commit suas mudanças** (`git commit -am 'Adicionar nova feature'`).
4.  **Push para a branch** (`git push origin feature/seu-nome-da-feature`).
5.  **Crie um Pull Request**.

### Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
