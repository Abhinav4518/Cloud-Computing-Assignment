import os
os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from service.app import app, state, validate_against_schema  # noqa: E402

client = TestClient(app)

def test_healthz_endpoint():
    resp = client.get("/healthz")
    assert resp.status_code == 200

def test_readyz_returns_503_when_not_loaded():
    state["ready"] = False
    resp = client.get("/readyz")
    assert resp.status_code == 503

def test_schema_validation_logic():
    valid = '{"intent": "weather_query", "date": null, "time": null, "place": "london"}'
    assert validate_against_schema(valid) is not None
    invalid_intent = '{"intent": "play_music", "date": null, "time": null, "place": null}'
    assert validate_against_schema(invalid_intent) is None
    extra_key = '{"intent": "alarm_set", "date": null, "time": null, "place": null, "extra": 1}'
    assert validate_against_schema(extra_key) is None

def test_fault_inject_returns_500():
    os.environ["FAULT_INJECT"] = "1"
    resp = client.post("/parse", json={"text": "set an alarm for six am"})
    assert resp.status_code == 500
    assert resp.json() == {"error": "invalid_model_output"}
    assert "X-Model-Version" in resp.headers
    assert "X-Served-By" in resp.headers
    os.environ["FAULT_INJECT"] = "0"
