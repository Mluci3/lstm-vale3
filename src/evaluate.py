"""
=============================================================================
MÓDULO DE AVALIAÇÃO
Tech Challenge 4 - LSTM VALE3
=============================================================================

Calcula as métricas de avaliação (MAE, RMSE, MAPE) sempre na ESCALA REAL
(em R$), desnormalizando com o scaler antes de qualquer cálculo, e compara
o modelo com um baseline ingênuo de persistência.

Por que desnormalizar ANTES de calcular?
-----------------------------------------
MAE e RMSE sobre valores normalizados (0-1) e sobre valores em R$ têm
magnitudes diferentes, mas o MAPE é especialmente traiçoeiro: calculado
sobre valores normalizados próximos de 0 ele explode (divisão por números
pequenos) e deixa de representar "erro percentual sobre o preço". Portanto
todas as métricas aqui partem dos valores já em R$.

Baseline ingênuo (persistência)
--------------------------------
Para preço de ação, o baseline mais honesto é "o preço de amanhã é igual ao
de hoje". Qualquer modelo que não supere esse baseline não está agregando
valor para prever o NÍVEL do preço — por isso ele é reportado lado a lado.
"""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mape(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%), ignorando posições em que o valor real é zero."""
    mask = y_real != 0
    return float(np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100)


def metricas_em_reais(
    y_real_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    scaler,
) -> Dict[str, float]:
    """
    Desnormaliza previsões e valores reais e calcula MAE, RMSE e MAPE em R$.

    Parâmetros:
    -----------
    y_real_norm : np.ndarray
        Valores reais normalizados (0-1)
    y_pred_norm : np.ndarray
        Previsões do modelo normalizadas (0-1)
    scaler : MinMaxScaler
        O MESMO scaler usado no preprocessamento (ajustado só no treino)

    Retorna:
    --------
    dict com mae_reais, rmse_reais, mape (todos em escala real)
    """
    y_real = scaler.inverse_transform(np.asarray(y_real_norm).reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(np.asarray(y_pred_norm).reshape(-1, 1)).flatten()

    return {
        'mae_reais': float(mean_absolute_error(y_real, y_pred)),
        'rmse_reais': float(np.sqrt(mean_squared_error(y_real, y_pred))),
        'mape': _mape(y_real, y_pred),
    }


def baseline_persistencia(
    X: np.ndarray,
    y_norm: np.ndarray,
    scaler,
) -> Dict[str, float]:
    """
    Calcula as métricas do baseline ingênuo (previsão = último preço da
    janela de entrada) na escala real, para comparar com o modelo.

    Parâmetros:
    -----------
    X : np.ndarray
        Sequências de entrada, shape (n, janela, 1)
    y_norm : np.ndarray
        Valores reais normalizados correspondentes
    scaler : MinMaxScaler
    """
    ultimo_preco_norm = np.asarray(X)[:, -1, 0]
    return metricas_em_reais(y_norm, ultimo_preco_norm, scaler)


def avaliar_modelo(
    modelo,
    X: np.ndarray,
    y_norm: np.ndarray,
    scaler,
    nome_conjunto: str = "conjunto",
) -> Dict[str, float]:
    """
    Faz a inferência do modelo em X e retorna as métricas em R$.
    Loga um resumo comparando com o baseline de persistência.
    """
    y_pred_norm = modelo.predict(X, verbose=0).flatten()
    metricas = metricas_em_reais(y_norm, y_pred_norm, scaler)
    baseline = baseline_persistencia(X, y_norm, scaler)

    logger.info(
        f"[{nome_conjunto}] LSTM     -> "
        f"MAE=R${metricas['mae_reais']:.2f} "
        f"RMSE=R${metricas['rmse_reais']:.2f} "
        f"MAPE={metricas['mape']:.2f}%"
    )
    logger.info(
        f"[{nome_conjunto}] baseline -> "
        f"MAE=R${baseline['mae_reais']:.2f} "
        f"RMSE=R${baseline['rmse_reais']:.2f} "
        f"MAPE={baseline['mape']:.2f}%"
    )

    return metricas
