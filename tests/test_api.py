from fastapi.testclient import TestClient

from forecast.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_series_listing():
    assert "retail_sales" in client.get("/series").json()


def test_forecast_happy_path():
    r = client.post("/forecast", json={"series": "retail_sales", "horizon": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == 6 and len(body["forecast"]) == 6
    assert body["model_version"].startswith("gbm_lags-")
    assert body["forecast"][0]["date"] == "2026-01-01"


def test_forecast_returns_prediction_interval():
    body = client.post("/forecast", json={"series": "ecommerce_orders", "horizon": 24}).json()
    assert body["interval_level"] == 0.8
    pts = body["forecast"]
    assert all(p["lower"] <= p["value"] <= p["upper"] for p in pts)
    assert pts[-1]["upper"] - pts[-1]["lower"] >= pts[0]["upper"] - pts[0]["lower"]


def test_forecast_validates_horizon_and_series():
    assert client.post("/forecast", json={"series": "retail_sales", "horizon": 0}).status_code == 422
    assert client.post("/forecast", json={"series": "retail_sales", "horizon": 999}).status_code == 422
    assert client.post("/forecast", json={"series": "nope", "horizon": 3}).status_code == 404
