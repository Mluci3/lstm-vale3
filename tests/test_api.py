"""
Testes de integração da API (carregam o modelo real via TestClient).

Não testamos /predict/latest aqui porque ele depende de rede (Yahoo Finance)
e deixaria o CI instável.
"""
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["modelo_carregado"] is True


def test_root_lista_endpoints():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_modelo_info():
    r = client.get("/modelo/info")
    assert r.status_code == 200
    body = r.json()
    assert body["janela_temporal"] == 60
    assert "mape" in body["metricas_teste"]


def test_predict_com_60_precos():
    precos = [55.0 + i * 0.1 for i in range(60)]
    r = client.post("/predict", json={"precos": precos})
    assert r.status_code == 200
    body = r.json()
    assert "preco_previsto" in body
    assert body["ticker"] == "VALE3.SA"
    assert body["direcao"] in ("alta", "baixa", "estável")


def test_predict_rejeita_tamanho_errado():
    precos = [55.0 + i * 0.1 for i in range(59)]  # 59 em vez de 60
    r = client.post("/predict", json={"precos": precos})
    assert r.status_code == 422
