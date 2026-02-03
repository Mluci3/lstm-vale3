# 📈 Tech Challenge 4 - Previsão de Preços VALE3 com LSTM

## Descrição

API para previsão de preços de fechamento das ações da VALE3 utilizando redes neurais LSTM (Long Short-Term Memory). O projeto contempla toda a pipeline de desenvolvimento, desde a coleta de dados até o deploy em produção na AWS.

**🔗 API em Produção:** https://6qvbjbl3ie.execute-api.sa-east-1.amazonaws.com

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
| **MAE** | R$ 1,47 |
| **RMSE** | R$ 1,86 |
| **MAPE** | 7,86% |

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
tech-challenge-4/
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
├── src/
│   └── api/
│       ├── __init__.py
│       └── main.py             # API FastAPI
├── Dockerfile
├── requirements.txt
└── README.md
```

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
  "mae_modelo": 1.47,
  "mape_modelo": 7.86,
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
git clone https://github.com/seu-usuario/tech-challenge-4.git
cd tech-challenge-4
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
O modelo consegue capturar a tendência geral dos preços, com erro médio de R$ 1,47.

### Interpretação das Métricas
- **MAE (R$ 1,47):** Em média, o modelo erra R$ 1,47 para cima ou para baixo
- **MAPE (7,86%):** O erro representa ~8% do valor da ação
- **RMSE (R$ 1,86):** Erros maiores são penalizados, indicando consistência

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