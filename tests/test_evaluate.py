"""Testes do módulo de avaliação (não dependem de TensorFlow)."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.evaluate import metricas_em_reais, baseline_persistencia, _mape


def _scaler_0_100():
    return MinMaxScaler().fit(np.array([[0.0], [100.0]]))


def test_mape_previsao_perfeita_e_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert _mape(y, y) == 0.0


def test_metricas_previsao_perfeita_sao_zero():
    scaler = _scaler_0_100()
    y_norm = np.array([0.1, 0.5, 0.9])
    m = metricas_em_reais(y_norm, y_norm, scaler)
    assert m["mae_reais"] == 0.0
    assert m["rmse_reais"] == 0.0
    assert m["mape"] == 0.0


def test_metricas_desnormalizam_para_reais():
    # Erro constante de 0.1 na escala normalizada vira R$ 10 na escala 0-100
    scaler = _scaler_0_100()
    y_norm = np.array([0.5, 0.5, 0.5])
    y_pred = np.array([0.6, 0.6, 0.6])
    m = metricas_em_reais(y_norm, y_pred, scaler)
    assert round(m["mae_reais"], 6) == 10.0


def test_baseline_persistencia_retorna_metricas():
    scaler = _scaler_0_100()
    X = np.random.rand(5, 60, 1)
    y = np.random.rand(5)
    m = baseline_persistencia(X, y, scaler)
    assert set(m) == {"mae_reais", "rmse_reais", "mape"}
