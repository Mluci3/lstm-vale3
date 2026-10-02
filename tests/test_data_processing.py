"""Testes da criação de sequências temporais (não dependem de TensorFlow)."""
import numpy as np

from src.data_processing import criar_sequencias


def test_criar_sequencias_shapes():
    dados = np.arange(100, dtype=float).reshape(-1, 1)
    X, y = criar_sequencias(dados, janela=10)

    # 100 dias, janela 10 -> 90 sequências
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)


def test_criar_sequencias_janela_desliza_corretamente():
    dados = np.arange(100, dtype=float).reshape(-1, 1)
    X, y = criar_sequencias(dados, janela=10)

    # A 1ª sequência usa os dias 0..9 para prever o dia 10
    assert X[0, 0, 0] == 0.0
    assert X[0, -1, 0] == 9.0
    assert y[0] == 10.0

    # A última sequência prevê o último dia (99)
    assert y[-1] == 99.0
