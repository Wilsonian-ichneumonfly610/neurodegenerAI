# NeuroDegenerAI Architecture

## System Architecture Overview

```mermaid
flowchart TB
    %% Data Sources
    ADNI[ADNI Dataset<br/>MRI & Tabular Data] --> NeuroIngest[Data Ingestion]
    Reddit[Reddit API] --> TrendIngest[Social Media Stream]
    X[X/Twitter API] --> TrendIngest

    %% NeuroDegenerAI Pipeline
    NeuroIngest --> NeuroPreprocess[Data Preprocessing]
    NeuroPreprocess --> NeuroFeatures[Feature Engineering]
    NeuroFeatures --> NeuroTrain[Model Training<br/>LightGBM/XGBoost/CNN]
    NeuroTrain --> NeuroModels[Saved Models]
    NeuroModels --> NeuroAPI[NeuroDegenerAI API<br/>FastAPI]

    %% Trend Detector Pipeline
    TrendIngest --> TrendEmbed[Text Embeddings<br/>sentence-transformers]
    TrendEmbed --> TrendCluster[Topic Clustering<br/>HDBSCAN/BERTopic]
    TrendCluster --> TrendTopics[Trend Analysis]
    TrendTopics --> TrendAPI[Trend Detector API<br/>FastAPI]

    %% APIs
    NeuroAPI --> Hub[Unified Dashboard<br/>Streamlit Hub]
    TrendAPI --> Hub

    %% User Interfaces
    Hub --> NeuroUI[NeuroDegenerAI UI<br/>Port 8501]
    Hub --> TrendUI[Trend Detector UI<br/>Port 8502]
    Hub --> MainUI[Main Hub UI<br/>Port 8503]

    %% External Access
    NeuroUI --> User[Users/Researchers]
    TrendUI --> User
    MainUI --> User

    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef api fill:#e8f5e8
    classDef ui fill:#fff3e0
    classDef user fill:#ffebee

    class ADNI,Reddit,X dataSource
    class NeuroIngest,NeuroPreprocess,NeuroFeatures,NeuroTrain,TrendIngest,TrendEmbed,TrendCluster,TrendTopics processing
    class NeuroAPI,TrendAPI api
    class Hub,NeuroUI,TrendUI,MainUI ui
    class User user
```

## Technology Stack

### NeuroDegenerAI
- **Data**: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Models**: LightGBM, XGBoost, ResNet18 CNN
- **Interpretability**: Grad-CAM, SHAP
- **API**: FastAPI with Pydantic schemas
- **UI**: Streamlit with interactive visualizations

### Real-Time Trend Detector
- **Data Sources**: Reddit API, X/Twitter API
- **NLP**: sentence-transformers, BERTopic, HDBSCAN
- **Clustering**: UMAP dimensionality reduction
- **API**: FastAPI with real-time streaming
- **UI**: Streamlit with auto-refresh dashboard

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions
- **Deployment**: Google Cloud Run
- **Monitoring**: Structured logging, health checks
- **Development**: Poetry, pre-commit hooks, pytest

## Data Flow

1. **NeuroDegenerAI**: ADNI data → preprocessing → feature engineering → model training → API → UI
2. **Trend Detector**: Social media streams → embeddings → clustering → trend analysis → API → UI
3. **Unified Hub**: Aggregates both services with health monitoring and navigation

## Key Features

- **Production Ready**: Docker containers, CI/CD, monitoring
- **Demo Mode**: Works without API keys or real data
- **Interpretable AI**: Grad-CAM heatmaps, SHAP explanations
- **Real-time**: Live social media trend detection
- **Scalable**: Microservices architecture, cloud deployment ready
