# 📈 Tech Challenge 4 - Previsão de Preços VALE3 com LSTM

## Descrição

API para previsão de preços de fechamento das ações da VALE3 utilizando redes neurais LSTM (Long Short-Term Memory). O projeto contempla toda a pipeline de desenvolvimento, desde a coleta de dados até o deploy em produção na AWS.

**🔗 API em Produção:** https://6qvbjbl3ie.execute-api.sa-east-1.amazonaws.com

> ⚠️ **Nota sobre cold start:** a API roda em AWS Lambda com container Docker (TensorFlow completo). Se ficar sem receber requisições por um tempo, a primeira chamada após esse período leva cerca de **8 a 10 segundos** (cold start: o container sobe e carrega o modelo). As chamadas seguintes respondem em menos de 100ms. A função foi configurada com memória suficiente (3 GB) para que esse cold start fique bem abaixo do limite de 30s do API Gateway.

---

## 🎯 Objetivos

- Coletar dados históricos de ações usando Yahoo Finance
- Desenvolver modelo LSTM para previsão de séries temporais
- Avaliar modelo com métricas apropriadas (MAE, RMSE, MAPE)
- Criar API RESTful para servir previsões
- Deploy em ambiente de nuvem (AWS Lambda)

---

## 📊 Métricas do Modelo

| Métrica | Valor |
|---------|-------|
| **MAE** | R$ 1,17 |
| **RMSE** | R$ 1,54 |
| **MAPE** | 1,84% |

### Comparação com um baseline ingênuo

Antes de confiar em qualquer modelo de série temporal, vale compará-lo com o baseline mais simples possível: prever que o preço de amanhã é igual ao de hoje ("persistência").

| | MAE | RMSE | MAPE |
|---|---|---|---|
| **LSTM** | R$ 1,17 | R$ 1,54 | 1,84% |
| **Baseline ingênuo** (hoje = previsão de amanhã) | R$ 0,51 | R$ 0,77 | 0,81% |

**O baseline ingênuo bate o LSTM nas três métricas, no conjunto de teste.** Isso não é um bug: é o comportamento esperado ao tentar prever o *nível* do preço de uma ação. Dia a dia, o preço de uma ação se aproxima de um [random walk](https://pt.wikipedia.org/wiki/Passeio_aleat%C3%B3rio) — a melhor estimativa para o valor de amanhã tende a ser o valor de hoje, e qualquer modelo que aprenda a "seguir" a série (em vez de prever a variação) vai naturalmente ficar perto do baseline, mas com um pouco de ruído extra vindo da própria arquitetura.

**O que isso significa na prática:**
- Para prever o **nível** do preço, o LSTM não agrega valor sobre o baseline ingênuo.
- Para ser útil de verdade, a próxima iteração deveria prever o **retorno** (variação percentual) em vez do preço absoluto, e ser avaliada por acerto de direção (alta/baixa) e pelo retorno de uma estratégia simulada — não só por erro absoluto de preço.
- Este projeto documenta esse resultado de forma transparente, em vez de esconder a comparação: entender os limites de um modelo é parte do trabalho.

---

## 🏗️ Arquitetura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Yahoo Finance │────▶│  Modelo LSTM    │────▶│   FastAPI       │
│   (Dados)       │     │  (Previsão)     │     │   (API REST)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   AWS Lambda    │
                                               │   + API Gateway │
                                               └─────────────────┘
```

---

## 🧠 Arquitetura do Modelo LSTM

```
Input (60 timesteps, 1 feature)
         │
         ▼
┌─────────────────────┐
│   LSTM (50 units)   │
│   return_sequences  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dropout (0.2)     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   LSTM (50 units)   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dropout (0.2)     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dense (1 unit)    │
└─────────────────────┘
```

**Hiperparâmetros:**
- Janela temporal: 60 dias
- Épocas: 35 (com early stopping)
- Batch size: 32
- Learning rate: 0.001
- Dropout: 20%

---

## 📁 Estrutura do Projeto

```
lstm-vale3/
├── data/
│   ├── raw/                    # Dados brutos
│   └── processed/              # Dados processados
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       └── config.json
├── models/
│   ├── lstm_vale3.h5           # Modelo treinado (formato H5)
│   ├── lstm_vale3.keras        # Modelo treinado (formato Keras)
│   ├── lstm_vale3_best.keras   # Melhor modelo durante treino
│   ├── scaler.joblib           # Normalizador
│   └── metricas.json           # Métricas do modelo
├── notebooks/
│   ├── 01_coleta_exploracao.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelo_lstm.ipynb
│   ├── 04_api_fastapi.ipynb
│   └── 05_deploy_aws.ipynb
├── src/                        # Lógica reutilizável (fora dos notebooks)
│   ├── data_processing.py      # Coleta, normalização, sequências e split temporal
│   ├── model.py                # Arquitetura e treino do LSTM
│   ├── evaluate.py             # Métricas em R$ e baseline de persistência
│   └── api/
│       ├── __init__.py
│       └── main.py             # API FastAPI
├── Dockerfile
├── requirements.txt
└── README.md
```

> **Organização do código:** a lógica central (dados, modelo e avaliação) vive em módulos Python em `src/`, reutilizáveis e testáveis. Os notebooks em `notebooks/` servem à exploração e à narrativa passo a passo, mas a implementação de referência está em `src/`.

---

## 🚀 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações da API |
| GET | `/health` | Health check |
| GET | `/modelo/info` | Métricas e hiperparâmetros |
| POST | `/predict` | Previsão com dados fornecidos |
| GET | `/docs` | Documentação Swagger |

### Exemplo de Uso - POST /predict

**Request:**
```bash
curl -X POST https://6qvbjbl3ie.execute-api.sa-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"precos": [52.5,53.1,52.8,53.5,54.0,53.8,54.2,54.5,54.1,53.9,54.3,54.8,55.0,54.7,55.2,55.5,55.1,54.9,55.3,55.8,56.0,55.7,56.2,56.5,56.1,55.9,56.3,56.8,57.0,56.7,57.2,57.5,57.1,56.9,57.3,57.8,58.0,57.7,58.2,58.5,58.1,57.9,58.3,58.8,59.0,58.7,59.2,59.5,59.1,58.9,59.3,59.8,60.0,59.7,60.2,60.5,60.1,59.9,60.3,60.8]}'
```

**Response:**
```json
{
  "preco_previsto": 59.59,
  "ultimo_preco": 60.8,
  "variacao_percentual": -1.99,
  "direcao": "baixa",
  "mae_modelo": 1.17,
  "mape_modelo": 1.84,
  "data_previsao": "2026-02-02T15:36:26.523023",
  "ticker": "VALE3.SA"
}
```

---

## 🛠️ Instalação Local

### Pré-requisitos
- Python 3.11+
- pip

### Passos

1. **Clone o repositório:**
```bash
git clone https://github.com/Mluci3/lstm-vale3.git
cd lstm-vale3
```

2. **Crie o ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute os notebooks em ordem:**
   - `01_coleta_exploracao.ipynb`
   - `02_preprocessamento.ipynb`
   - `03_modelo_lstm.ipynb`
   - `04_api_fastapi.ipynb`

5. **Rode a API localmente:**
```bash
cd src/api
uvicorn main:app --reload --port 8000
```

Acesse: http://localhost:8000/docs

---

## 🐳 Docker

### Build da imagem:
```bash
docker build --platform linux/amd64 -t lstm-vale3-api:latest .
```

### Executar localmente:
```bash
docker run -p 8000:8080 lstm-vale3-api:latest
```

---

## ☁️ Deploy AWS

O projeto está deployado na AWS usando:
- **AWS Lambda** - Execução serverless
- **API Gateway** - Endpoint HTTP
- **ECR** - Registro de imagens Docker

### Arquitetura AWS:
```
Cliente → API Gateway → Lambda (Container) → ECR (Imagem Docker)
```

---

## 📈 Pipeline de Dados

1. **Coleta:** Download de 5 anos de dados históricos da VALE3 via yfinance
2. **Limpeza:** Remoção de valores nulos, tratamento de outliers
3. **Normalização:** MinMaxScaler (0 a 1)
4. **Sequenciamento:** Janela deslizante de 60 dias
5. **Divisão:** 80% treino / 10% validação / 10% teste

---

## 📊 Resultados

### Previsão vs Real (Conjunto de Teste)
O modelo consegue capturar a tendência geral dos preços, com erro médio de R$ 1,17 — mas, como a seção de comparação com baseline acima mostra, ainda fica atrás da estratégia ingênua de repetir o último preço conhecido.

### Interpretação das Métricas
- **MAE (R$ 1,17):** Em média, o modelo erra R$ 1,17 para cima ou para baixo
- **MAPE (1,84%):** O erro representa ~1,8% do valor da ação
- **RMSE (R$ 1,54):** Por elevar os erros ao quadrado antes de tirar a média, penaliza desproporcionalmente os erros grandes — por isso é mais sensível a outliers do que o MAE (RMSE > MAE sugere a presença de alguns erros maiores no meio de erros pequenos)

### Limitações conhecidas
- **O LSTM não bate o baseline ingênuo** no conjunto de teste (ver comparação acima). Isso é esperado para previsão do nível de preço de uma ação e está documentado de forma transparente, não escondido.
- O `MinMaxScaler` é ajustado (fit) apenas no conjunto de treino e aplicado (transform) ao restante da série, evitando vazamento de dados do futuro (val/teste) para a escala usada no treino.

---

## 🔧 Tecnologias Utilizadas

| Categoria | Tecnologia |
|-----------|------------|
| Linguagem | Python 3.11 |
| Deep Learning | TensorFlow/Keras |
| API | FastAPI |
| Dados | yfinance, pandas, numpy |
| ML | scikit-learn |
| Deploy | Docker, AWS Lambda, API Gateway |
| Versionamento | Git/GitHub |

---

## 📝 Notebooks

| Notebook | Descrição |
|----------|-----------|
| `01_coleta_exploracao.ipynb` | Coleta de dados e análise exploratória |
| `02_preprocessamento.ipynb` | Normalização e criação de sequências |
| `03_modelo_lstm.ipynb` | Construção, treino e avaliação do LSTM |
| `04_api_fastapi.ipynb` | Criação da API e testes |
| `05_deploy_aws.ipynb` | Deploy na AWS (Lambda + API Gateway) |

---

## 👩‍💻 Autora

**Maria Araujo**

---

## 📄 Licença

Este projeto foi desenvolvido como parte do Tech Challenge da Pós-Graduação em Machine Learning Engineering.

---

## 🔗 Links Úteis

- **API em Produção:** https://6qvbjbl3ie.execute-api.sa-east-1.amazonaws.com
- **Documentação Swagger:** https://6qvbjbl3ie.execute-api.sa-east-1.amazonaws.com/docs
- **Health Check:** https://6qvbjbl3ie.execute-api.sa-east-1.amazonaws.com/health