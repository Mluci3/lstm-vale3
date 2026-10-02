"""
=============================================================================
MÓDULO DE MODELO LSTM
Tech Challenge 4 - LSTM VALE3
=============================================================================

Responsável pela arquitetura e pelo treino do modelo LSTM. A lógica aqui
reproduz a do notebook `03_modelo_lstm.ipynb`, mas organizada em funções
reutilizáveis para que os notebooks fiquem apenas com a parte exploratória.

Exemplo de uso:
    from src.model import criar_modelo_lstm, treinar_modelo

    modelo = criar_modelo_lstm(input_shape=(60, 1))
    historico = treinar_modelo(modelo, X_train, y_train, X_val, y_val)
    modelo.save('models/lstm_vale3.h5', include_optimizer=False)
"""

import os
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# HIPERPARÂMETROS PADRÃO
# =============================================================================

HIPERPARAMETROS = {
    'lstm_units_1': 50,
    'lstm_units_2': 50,
    'dropout_rate': 0.2,
    'epochs': 100,          # máximo; o early stopping para antes
    'batch_size': 32,
    'learning_rate': 0.001,
    'patience': 15,         # épocas sem melhora antes de parar
}


# =============================================================================
# ARQUITETURA
# =============================================================================

def criar_modelo_lstm(
    input_shape: Tuple[int, int],
    lstm_units_1: int = HIPERPARAMETROS['lstm_units_1'],
    lstm_units_2: int = HIPERPARAMETROS['lstm_units_2'],
    dropout_rate: float = HIPERPARAMETROS['dropout_rate'],
    learning_rate: float = HIPERPARAMETROS['learning_rate'],
) -> tf.keras.Model:
    """
    Cria e compila o modelo LSTM empilhado (duas camadas) para regressão.

    Arquitetura:
        Input(input_shape)
        -> LSTM(lstm_units_1, return_sequences=True) -> Dropout
        -> LSTM(lstm_units_2)                        -> Dropout
        -> Dense(1)

    Parâmetros:
    -----------
    input_shape : tuple
        (timesteps, features), ex: (60, 1)
    lstm_units_1, lstm_units_2 : int
        Neurônios de cada camada LSTM
    dropout_rate : float
        Taxa de dropout (regularização)
    learning_rate : float
        Taxa de aprendizado do otimizador Adam

    Retorna:
    --------
    tf.keras.Model compilado (loss=MSE, métrica=MAE)
    """
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units_1, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1),
    ])

    modelo.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    logger.info("Modelo LSTM criado e compilado.")
    return modelo


# =============================================================================
# TREINO
# =============================================================================

def treinar_modelo(
    modelo: tf.keras.Model,
    X_train, y_train,
    X_val, y_val,
    epochs: int = HIPERPARAMETROS['epochs'],
    batch_size: int = HIPERPARAMETROS['batch_size'],
    patience: int = HIPERPARAMETROS['patience'],
    caminho_best: str = 'models/lstm_vale3_best.keras',
):
    """
    Treina o modelo com early stopping, checkpoint do melhor modelo e
    redução de learning rate em platôs.

    Retorna o objeto `History` do Keras.
    """
    os.makedirs(os.path.dirname(caminho_best) or '.', exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=caminho_best,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    historico = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(f"Treino concluído em {len(historico.history['loss'])} épocas.")
    return historico


def salvar_modelo(modelo: tf.keras.Model, caminho: str = 'models/lstm_vale3.h5') -> None:
    """
    Salva o modelo final.

    Nota: salvamos sem o estado do otimizador (`include_optimizer=False`).
    O modelo é usado apenas para inferência na API, então o otimizador é
    desnecessário — e mantê-lo pode causar erro de desserialização ao
    recarregar o `.h5` em outra versão do Keras/TensorFlow.
    """
    os.makedirs(os.path.dirname(caminho) or '.', exist_ok=True)
    incluir_otim = not caminho.endswith('.h5')
    modelo.save(caminho, include_optimizer=incluir_otim)
    logger.info(f"Modelo salvo em {caminho} (include_optimizer={incluir_otim}).")
