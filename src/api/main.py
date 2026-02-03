"""
API FastAPI para previsão de preços da VALE3 usando LSTM.

Endpoints:
- GET  /                  → Informações gerais da API
- GET  /health            → Health check
- GET  /modelo/info       → Informações do modelo
- POST /predict           → Previsão com dados fornecidos
- GET  /predict/latest    → Previsão com dados mais recentes
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import tensorflow as tf
import joblib
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# =============================================================================
# CONFIGURAÇÃO DE CAMINHOS
# =============================================================================

# Detecta se está rodando local ou no Lambda
if os.path.exists('/var/task'):
    # AWS Lambda
    BASE_PATH = '/var/task'
else:
    # Local - ajusta caminho relativo
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_PATH, 'models', 'lstm_vale3.h5')
SCALER_PATH = os.path.join(BASE_PATH, 'models', 'scaler.joblib')
CONFIG_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'config.json')
METRICS_PATH = os.path.join(BASE_PATH, 'models', 'metricas.json')

# =============================================================================
# INICIALIZAÇÃO DA API
# =============================================================================

app = FastAPI(
    title="API LSTM VALE3",
    description="API para previsão de preços da VALE3 usando modelo LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# CARREGAMENTO DO MODELO (uma única vez na inicialização)
# =============================================================================

print(f"Carregando modelo de: {MODEL_PATH}")

try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    with open(METRICS_PATH, 'r') as f:
        metricas = json.load(f)

    MODELO_CARREGADO = True
    print("✓ Modelo carregado com sucesso!")
except Exception as e:
    print(f"✗ Erro ao carregar modelo: {e}")
    MODELO_CARREGADO = False
    modelo = None
    scaler = None
    config = {'janela_temporal': 60}
    metricas = {}

# =============================================================================
# SCHEMAS (Pydantic)
# =============================================================================

class PrecosInput(BaseModel):
    """Schema para entrada de preços históricos."""
    precos: List[float] = Field(
        ...,
        description="Lista com os últimos 60 preços de fechamento em R$",
        min_items=60,
        max_items=60
    )

    class Config:
        schema_extra = {
            "example": {
                "precos": [55.0 + i * 0.1 for i in range(60)]
            }
        }

class PrevisaoOutput(BaseModel):
    """Schema para saída da previsão."""
    preco_previsto: float = Field(..., description="Preço previsto para o próximo dia em R$")
    ultimo_preco: float = Field(..., description="Último preço utilizado na previsão")
    variacao_percentual: float = Field(..., description="Variação percentual esperada")
    direcao: str = Field(..., description="Direção esperada: alta, baixa ou estável")
    mae_modelo: float = Field(..., description="Erro médio absoluto do modelo em R$")
    mape_modelo: float = Field(..., description="Erro percentual médio do modelo")
    data_previsao: str = Field(..., description="Data/hora da previsão")
    ticker: str = Field(default="VALE3.SA", description="Ticker da ação")

class HealthOutput(BaseModel):
    """Schema para health check."""
    status: str
    modelo_carregado: bool
    timestamp: str

class ModeloInfoOutput(BaseModel):
    """Schema para informações do modelo."""
    nome: str
    versao: str
    janela_temporal: int
    metricas_teste: dict
    hiperparametros: dict

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def fazer_previsao(precos_historicos: List[float]) -> dict:
    """Faz previsão do próximo preço."""
    if not MODELO_CARREGADO:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    janela = config['janela_temporal']

    if len(precos_historicos) != janela:
        raise HTTPException(
            status_code=400, 
            detail=f"Esperado {janela} preços, recebido {len(precos_historicos)}"
        )

    # Prepara dados
    precos = np.array(precos_historicos).reshape(-1, 1)
    precos_norm = scaler.transform(precos)
    entrada = precos_norm.reshape(1, janela, 1)

    # Previsão
    previsao_norm = modelo.predict(entrada, verbose=0)
    previsao_reais = scaler.inverse_transform(previsao_norm)

    preco_previsto = float(previsao_reais[0, 0])
    ultimo_preco = precos_historicos[-1]
    variacao = ((preco_previsto - ultimo_preco) / ultimo_preco) * 100

    return {
        'preco_previsto': round(preco_previsto, 2),
        'ultimo_preco': round(ultimo_preco, 2),
        'variacao_percentual': round(variacao, 2),
        'direcao': 'alta' if variacao > 0 else 'baixa' if variacao < 0 else 'estável',
        'mae_modelo': round(metricas.get('teste', {}).get('mae_reais', 0), 2),
        'mape_modelo': round(metricas.get('teste', {}).get('mape', 0), 2),
        'data_previsao': datetime.now().isoformat(),
        'ticker': 'VALE3.SA'
    }

def buscar_dados_recentes(ticker: str = 'VALE3.SA', dias: int = 60) -> dict:
    """Busca dados recentes do Yahoo Finance."""
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=dias * 2)

    try:
        df = yf.download(
            ticker,
            start=data_inicio.strftime('%Y-%m-%d'),
            end=data_fim.strftime('%Y-%m-%d'),
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.tail(dias)

        if len(df) < dias:
            raise HTTPException(
                status_code=500,
                detail=f"Dados insuficientes. Esperado {dias}, obtido {len(df)}"
            )

        return {
            'precos': [float(p) for p in df['Close'].values.tolist()],
            'ultima_data': df.index[-1].strftime('%Y-%m-%d'),
            'ticker': ticker
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar dados: {str(e)}")

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Geral"])
async def root():
    """Endpoint raiz com informações gerais da API."""
    return {
        "nome": "API LSTM VALE3",
        "descricao": "API para previsão de preços da VALE3 usando modelo LSTM",
        "versao": "1.0.0",
        "endpoints": {
            "health": "/health",
            "modelo_info": "/modelo/info",
            "previsao_manual": "POST /predict",
            "previsao_automatica": "GET /predict/latest",
            "documentacao": "/docs"
        }
    }

@app.get("/health", response_model=HealthOutput, tags=["Sistema"])
async def health_check():
    """Verifica se a API está funcionando corretamente."""
    return {
        "status": "healthy" if MODELO_CARREGADO else "unhealthy",
        "modelo_carregado": MODELO_CARREGADO,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/modelo/info", response_model=ModeloInfoOutput, tags=["Modelo"])
async def modelo_info():
    """Retorna informações sobre o modelo treinado."""
    if not MODELO_CARREGADO:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    return {
        "nome": "LSTM VALE3",
        "versao": "1.0.0",
        "janela_temporal": config['janela_temporal'],
        "metricas_teste": {
            "mae_reais": metricas.get('teste', {}).get('mae_reais', 0),
            "rmse_reais": metricas.get('teste', {}).get('rmse_reais', 0),
            "mape": metricas.get('teste', {}).get('mape', 0)
        },
        "hiperparametros": metricas.get('hiperparametros', {})
    }

@app.post("/predict", response_model=PrevisaoOutput, tags=["Previsão"])
async def predict(dados: PrecosInput):
    """
    Faz previsão do próximo preço baseado nos preços fornecidos.

    Envie uma lista com exatamente 60 preços de fechamento (em R$).
    """
    resultado = fazer_previsao(dados.precos)
    return resultado

@app.get("/predict/latest", response_model=PrevisaoOutput, tags=["Previsão"])
async def predict_latest(ticker: str = "VALE3.SA"):
    """
    Faz previsão usando os dados mais recentes do Yahoo Finance.

    Este endpoint busca automaticamente os últimos 60 dias de dados.
    """
    # Busca dados
    dados = buscar_dados_recentes(ticker, config['janela_temporal'])

    # Faz previsão
    resultado = fazer_previsao(dados['precos'])
    resultado['ticker'] = ticker

    return resultado

# =============================================================================
# HANDLER PARA AWS LAMBDA (Mangum)
# =============================================================================

# Importação condicional do Mangum para AWS Lambda
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    # Mangum não instalado (desenvolvimento local)
    handler = None
