from pathlib import Path
from fastapi.testclient import TestClient
from axonforge.api import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["project"] == "Connectome Project"
    assert body["subsystem"] == "AxonForge"
    assert body["version"] == "1.4.0"


def test_demo_run_redirect():
    c = TestClient(app)
    r = c.get("/demo_run", follow_redirects=False)
    assert r.status_code == 303
    st = c.get("/status").json()
    assert st["telemetry"]["total_volume"] > 0


def test_demo_run_return_to():
    client = TestClient(app)
    r = client.get(
        "/demo_run",
        params={"reset": True, "return_to": "http://127.0.0.1:8742/"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "http://127.0.0.1:8742/?ran=1"


def test_dashboard_wires_absolute_cta():
    html = Path("axonforge_dashboard/index.html").read_text(encoding="utf-8")
    assert "wireSubmit" in html
    assert "return_to" in html
    assert 'id="af-submit-run"' in html
