"""
=============================================================================
MÓDULO DE COLETA E PROCESSAMENTO DE DADOS
Tech Challenge 4 - LSTM VALE3
=============================================================================

Este módulo é responsável por:
1. Baixar dados históricos de ações via Yahoo Finance
2. Preprocessar os dados (limpeza, normalização)
3. Criar sequências temporais para o LSTM
4. Dividir em conjuntos de treino, validação e teste

CONCEITOS IMPORTANTES:
----------------------

1. SÉRIES TEMPORAIS
   Dados ordenados no tempo onde a ordem importa. Diferente de dados tabulares
   comuns onde cada linha é independente, em séries temporais o valor de hoje
   depende dos valores anteriores.

2. NORMALIZAÇÃO (MinMaxScaler)
   Transforma os dados para ficarem entre 0 e 1:
   
   x_normalizado = (x - x_min) / (x_max - x_min)
   
   Por que fazer isso?
   - Redes neurais usam funções de ativação (sigmoid, tanh) que funcionam
     melhor com valores pequenos
   - Evita que features com valores grandes dominem o aprendizado
   - Acelera a convergência do gradiente descendente

3. SEQUÊNCIAS TEMPORAIS (Janela Deslizante)
   O LSTM precisa receber uma sequência de valores para prever o próximo.
   Usamos uma "janela deslizante" que move pelos dados:
   
   Dados: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
   Janela de 5:
   
   Sequência 1: [10, 11, 12, 13, 14] → Prever: 15
   Sequência 2: [11, 12, 13, 14, 15] → Prever: 16
   Sequência 3: [12, 13, 14, 15, 16] → Prever: 17
   ...

4. DIVISÃO TEMPORAL (Train/Val/Test)
   IMPORTANTE: Em séries temporais, NÃO podemos embaralhar os dados!
   O teste deve ser sempre no FUTURO em relação ao treino.
   
   ├── Treino (80%) ──┼── Val (10%) ──┼── Teste (10%) ──┤
   Passado ─────────────────────────────────────► Futuro
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple, Optional
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Configurações padrão do projeto
CONFIG = {
    'ticker': 'VALE3.SA',           # Código da ação no Yahoo Finance
    'periodo_anos': 5,               # Quantos anos de histórico baixar
    'janela_temporal': 60,           # Dias usados para prever o próximo
    'feature_alvo': 'Close',         # Coluna que queremos prever
    'train_ratio': 0.8,              # 80% para treino
    'val_ratio': 0.1,                # 10% para validação
    'test_ratio': 0.1,               # 10% para teste
}


# =============================================================================
# FUNÇÕES DE COLETA DE DADOS
# =============================================================================

def baixar_dados_acao(
    ticker: str = CONFIG['ticker'],
    periodo_anos: int = CONFIG['periodo_anos'],
    salvar_csv: bool = True,
    caminho_saida: str = 'data/raw'
) -> pd.DataFrame:
    """
    Baixa dados históricos de uma ação do Yahoo Finance.
    
    EXPLICAÇÃO DO YFINANCE:
    -----------------------
    O yfinance é uma biblioteca que acessa a API do Yahoo Finance.
    Ela retorna um DataFrame com as colunas:
    - Open: Preço de abertura do dia
    - High: Maior preço do dia
    - Low: Menor preço do dia
    - Close: Preço de fechamento (ESTE É O MAIS USADO PARA PREVISÃO)
    - Adj Close: Fechamento ajustado (considera dividendos e splits)
    - Volume: Quantidade de ações negociadas
    
    POR QUE USAR 'CLOSE' E NÃO 'ADJ CLOSE'?
    Para este desafio, usaremos 'Close' por simplicidade. Em produção,
    'Adj Close' seria mais preciso pois considera eventos corporativos.
    
    Parâmetros:
    -----------
    ticker : str
        Código da ação (ex: 'VALE3.SA' para Vale, 'PETR4.SA' para Petrobras)
        O sufixo .SA indica ações da B3 (bolsa brasileira)
    
    periodo_anos : int
        Quantos anos de histórico baixar
    
    salvar_csv : bool
        Se True, salva os dados em CSV
    
    caminho_saida : str
        Pasta onde salvar o CSV
    
    Retorna:
    --------
    pd.DataFrame com os dados históricos
    """
    logger.info(f"Baixando dados de {ticker} dos últimos {periodo_anos} anos...")
    
    # Calcula as datas de início e fim
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=periodo_anos * 365)
    
    # Baixa os dados usando yfinance
    # O método download() é o mais comum, retorna DataFrame
    df = yf.download(
        tickers=ticker,
        start=data_inicio.strftime('%Y-%m-%d'),
        end=data_fim.strftime('%Y-%m-%d'),
        progress=False  # Desativa barra de progresso do yfinance
    )
    
    # Verifica se conseguiu baixar dados
    if df.empty:
        raise ValueError(f"Não foi possível baixar dados para {ticker}")
    
    logger.info(f"✓ Baixados {len(df)} registros de {df.index.min()} até {df.index.max()}")
    
    # Salva em CSV se solicitado
    if salvar_csv:
        os.makedirs(caminho_saida, exist_ok=True)
        nome_arquivo = f"{ticker.replace('.', '_').lower()}.csv"
        caminho_completo = os.path.join(caminho_saida, nome_arquivo)
        df.to_csv(caminho_completo)
        logger.info(f"✓ Dados salvos em {caminho_completo}")
    
    return df


def carregar_dados_csv(caminho: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV previamente salvo.
    
    Útil para não precisar baixar novamente (economia de tempo e banda).
    """
    logger.info(f"Carregando dados de {caminho}...")
    df = pd.read_csv(caminho, index_col=0, parse_dates=True)
    logger.info(f"✓ Carregados {len(df)} registros")
    return df


# =============================================================================
# FUNÇÕES DE ANÁLISE EXPLORATÓRIA
# =============================================================================

def analise_basica(df: pd.DataFrame) -> dict:
    """
    Realiza análise básica dos dados para entender o dataset.
    
    ESTA FUNÇÃO É EDUCATIVA - mostra informações importantes para
    entender os dados antes de modelar.
    
    O que analisamos:
    1. Período coberto pelos dados
    2. Valores mínimo, máximo, médio
    3. Volatilidade (desvio padrão)
    4. Dados faltantes
    """
    info = {
        'periodo_inicio': df.index.min(),
        'periodo_fim': df.index.max(),
        'total_registros': len(df),
        'colunas': list(df.columns),
        'dados_faltantes': df.isnull().sum().to_dict(),
        'estatisticas': df.describe().to_dict()
    }
    
    # Estatísticas específicas do preço de fechamento
    if 'Close' in df.columns:
        close = df['Close']
        info['preco_fechamento'] = {
            'minimo': float(close.min()),
            'maximo': float(close.max()),
            'media': float(close.mean()),
            'desvio_padrao': float(close.std()),
            'volatilidade_percentual': float((close.std() / close.mean()) * 100)
        }
    
    return info


def verificar_dados_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica e trata dados faltantes.
    
    ESTRATÉGIAS PARA DADOS FALTANTES:
    ---------------------------------
    1. Remover linhas (dropna) - Simples, mas perde dados
    2. Forward fill (ffill) - Usa o último valor conhecido
    3. Backward fill (bfill) - Usa o próximo valor conhecido
    4. Interpolação - Calcula valor intermediário
    
    Para séries temporais financeiras, forward fill é comum porque
    simula a realidade: se não houve negociação, o preço é o último conhecido.
    """
    dados_faltantes = df.isnull().sum()
    
    if dados_faltantes.any():
        logger.warning(f"Encontrados dados faltantes:\n{dados_faltantes[dados_faltantes > 0]}")
        
        # Usa forward fill seguido de backward fill para garantir
        df = df.ffill().bfill()
        logger.info("✓ Dados faltantes preenchidos com forward/backward fill")
    else:
        logger.info("✓ Nenhum dado faltante encontrado")
    
    return df


# =============================================================================
# FUNÇÕES DE PREPROCESSAMENTO
# =============================================================================

def normalizar_dados(
    dados: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
    salvar_scaler: bool = True,
    caminho_scaler: str = 'models/scaler.joblib'
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normaliza os dados usando MinMaxScaler.
    
    EXPLICAÇÃO DETALHADA DO MinMaxScaler:
    -------------------------------------
    
    Fórmula: x_norm = (x - x_min) / (x_max - x_min)
    
    Exemplo prático:
    - Dados originais: [50, 60, 70, 80, 90]
    - x_min = 50, x_max = 90
    - 50 normalizado: (50-50)/(90-50) = 0.0
    - 70 normalizado: (70-50)/(90-50) = 0.5
    - 90 normalizado: (90-50)/(90-50) = 1.0
    - Dados normalizados: [0.0, 0.25, 0.5, 0.75, 1.0]
    
    POR QUE SALVAR O SCALER?
    ------------------------
    O scaler guarda os valores de min e max do TREINO.
    Na hora de fazer previsões, precisamos:
    1. Normalizar os dados de entrada com o MESMO scaler
    2. Desnormalizar a previsão para obter o preço real
    
    Se não salvarmos, não conseguimos converter de volta!
    
    Parâmetros:
    -----------
    dados : np.ndarray
        Array com os valores a normalizar
    
    scaler : MinMaxScaler, opcional
        Se fornecido, usa este scaler (para validação/teste)
        Se None, cria um novo (para treino)
    
    Retorna:
    --------
    Tuple[np.ndarray, MinMaxScaler]: dados normalizados e o scaler usado
    """
    # Garante que os dados têm formato 2D (necessário para sklearn)
    # Reshape de (n,) para (n, 1)
    if len(dados.shape) == 1:
        dados = dados.reshape(-1, 1)
    
    if scaler is None:
        # Cria novo scaler e ajusta aos dados (fit_transform)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dados_normalizados = scaler.fit_transform(dados)
        logger.info(f"✓ Scaler criado - Min: {scaler.data_min_[0]:.2f}, Max: {scaler.data_max_[0]:.2f}")
        
        # Salva o scaler para uso futuro
        if salvar_scaler:
            os.makedirs(os.path.dirname(caminho_scaler), exist_ok=True)
            joblib.dump(scaler, caminho_scaler)
            logger.info(f"✓ Scaler salvo em {caminho_scaler}")
    else:
        # Usa scaler existente (apenas transform, sem fit)
        dados_normalizados = scaler.transform(dados)
    
    return dados_normalizados, scaler


def desnormalizar_dados(dados_normalizados: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Converte dados normalizados de volta para a escala original.
    
    QUANDO USAR:
    - Após a previsão do modelo (que retorna valor entre 0 e 1)
    - Para mostrar o preço real em reais (R$)
    
    Fórmula inversa: x_original = x_norm * (x_max - x_min) + x_min
    """
    if len(dados_normalizados.shape) == 1:
        dados_normalizados = dados_normalizados.reshape(-1, 1)
    
    return scaler.inverse_transform(dados_normalizados)


def criar_sequencias(
    dados: np.ndarray,
    janela: int = CONFIG['janela_temporal']
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências de entrada (X) e saída (y) para o LSTM.
    
    EXPLICAÇÃO DA JANELA DESLIZANTE:
    --------------------------------
    
    O LSTM precisa de sequências para aprender padrões temporais.
    Usamos uma "janela" que desliza pelos dados:
    
    Janela = 5 dias (exemplo simplificado)
    Dados: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    i=0: X=[10,11,12,13,14] → y=15  (usa dias 0-4 para prever dia 5)
    i=1: X=[11,12,13,14,15] → y=16  (usa dias 1-5 para prever dia 6)
    i=2: X=[12,13,14,15,16] → y=17  (usa dias 2-6 para prever dia 7)
    ...
    
    FORMATO DE SAÍDA PARA LSTM:
    ---------------------------
    O Keras LSTM espera entrada no formato: (samples, timesteps, features)
    
    - samples: número de sequências
    - timesteps: tamanho da janela (60 no nosso caso)
    - features: número de variáveis (1 = só preço de fechamento)
    
    Exemplo: Se temos 1000 dias de dados e janela de 60:
    - Teremos 940 sequências (1000 - 60)
    - Cada sequência tem 60 timesteps
    - Cada timestep tem 1 feature
    - Shape final: (940, 60, 1)
    
    Parâmetros:
    -----------
    dados : np.ndarray
        Dados normalizados (shape: (n, 1))
    
    janela : int
        Número de dias anteriores usados para prever o próximo
    
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]: X (sequências) e y (valores alvo)
    """
    X, y = [], []
    
    # Percorre os dados criando sequências
    for i in range(janela, len(dados)):
        # X: sequência dos últimos 'janela' dias
        X.append(dados[i - janela:i, 0])
        
        # y: valor do próximo dia (o que queremos prever)
        y.append(dados[i, 0])
    
    # Converte para numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X para formato LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    logger.info(f"✓ Criadas {len(X)} sequências")
    logger.info(f"  Shape X: {X.shape} (samples, timesteps, features)")
    logger.info(f"  Shape y: {y.shape} (samples,)")
    
    return X, y


def dividir_dados_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = CONFIG['train_ratio'],
    val_ratio: float = CONFIG['val_ratio']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em treino, validação e teste RESPEITANDO A ORDEM TEMPORAL.
    
    IMPORTANTE - POR QUE NÃO EMBARALHAR?
    ------------------------------------
    
    Em séries temporais, a ordem cronológica é fundamental!
    
    ERRADO (o que NUNCA fazer):
    - Embaralhar os dados antes de dividir
    - Usar train_test_split com shuffle=True
    - Isso causa "data leakage" - o modelo veria dados do futuro durante treino
    
    CERTO (o que fazemos aqui):
    - Dividir sequencialmente: treino = passado, teste = futuro
    - O modelo NUNCA vê dados do futuro durante o treino
    
    Visualização:
    
    |←────── Treino (80%) ──────→|←─ Val (10%) ─→|←─ Test (10%) ─→|
    Jan/2019 ───────────────────────────────────────────────► Dez/2024
    
    POR QUE VALIDAÇÃO SEPARADA DO TESTE?
    ------------------------------------
    - Validação: usada para ajustar hiperparâmetros (early stopping, etc.)
    - Teste: NUNCA tocado até a avaliação final
    - Se usássemos teste para ajustar parâmetros, teríamos overfitting indireto
    """
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Divisão sequencial (SEM embaralhar!)
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    logger.info(f"✓ Dados divididos:")
    logger.info(f"  Treino:    {len(X_train)} amostras ({len(X_train)/n_total*100:.1f}%)")
    logger.info(f"  Validação: {len(X_val)} amostras ({len(X_val)/n_total*100:.1f}%)")
    logger.info(f"  Teste:     {len(X_test)} amostras ({len(X_test)/n_total*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# FUNÇÃO PRINCIPAL - PIPELINE COMPLETO
# =============================================================================

def preparar_dados_completo(
    ticker: str = CONFIG['ticker'],
    periodo_anos: int = CONFIG['periodo_anos'],
    janela: int = CONFIG['janela_temporal'],
    usar_cache: bool = True
) -> dict:
    """
    Executa o pipeline completo de preparação de dados.
    
    Esta função orquestra todas as etapas:
    1. Baixa ou carrega dados
    2. Limpa dados faltantes
    3. Normaliza
    4. Cria sequências
    5. Divide em treino/val/teste
    
    Parâmetros:
    -----------
    ticker : str
        Código da ação
    
    periodo_anos : int
        Anos de histórico
    
    janela : int
        Tamanho da janela temporal
    
    usar_cache : bool
        Se True, tenta carregar dados salvos antes de baixar
    
    Retorna:
    --------
    dict com todos os dados preparados e metadados
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE PREPARAÇÃO DE DADOS")
    logger.info("=" * 60)
    
    # 1. Coleta de dados
    caminho_cache = f"data/raw/{ticker.replace('.', '_').lower()}.csv"
    
    if usar_cache and os.path.exists(caminho_cache):
        df = carregar_dados_csv(caminho_cache)
    else:
        df = baixar_dados_acao(ticker, periodo_anos)
    
    # 2. Limpeza
    df = verificar_dados_faltantes(df)
    
    # 3. Extrai preço de fechamento
    precos = df['Close'].values
    
    # 4. Normalização
    precos_norm, scaler = normalizar_dados(precos)
    
    # 5. Criação de sequências
    X, y = criar_sequencias(precos_norm, janela)
    
    # 6. Divisão temporal
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados_temporal(X, y)
    
    # Prepara resultado
    resultado = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'df_original': df,
        'config': {
            'ticker': ticker,
            'janela': janela,
            'periodo_anos': periodo_anos,
            'total_amostras': len(X)
        }
    }
    
    logger.info("=" * 60)
    logger.info("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 60)
    
    return resultado


# =============================================================================
# EXECUÇÃO DIRETA (para testes)
# =============================================================================

if __name__ == "__main__":
    # Teste do pipeline
    resultado = preparar_dados_completo()
    
    print("\n" + "=" * 60)
    print("RESUMO DOS DADOS PREPARADOS")
    print("=" * 60)
    print(f"Ticker: {resultado['config']['ticker']}")
    print(f"Janela temporal: {resultado['config']['janela']} dias")
    print(f"Total de sequências: {resultado['config']['total_amostras']}")
    print(f"\nShapes:")
    print(f"  X_train: {resultado['X_train'].shape}")
    print(f"  X_val:   {resultado['X_val'].shape}")
    print(f"  X_test:  {resultado['X_test'].shape}")
