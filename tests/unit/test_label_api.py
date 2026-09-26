from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import label


def _client(tmp_path, monkeypatch):
    sample = tmp_path / "sample.csv"
    sample.write_text(
        "event_id,split,change_type,stratum_population\nCHG-000001,calibration,construction,10\n"
        "CHG-000002,heldout,water,5\n",
        encoding="utf-8",
    )
    pre = tmp_path / "pre.csv"
    pre.write_text(
        "event_id,real,true_type,note\nCHG-000001,no,,seasonal flooding\nCHG-000002,yes,water,x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(label, "SAMPLE", sample)
    monkeypatch.setattr(label, "PRELABELS", pre)
    monkeypatch.setattr(label, "HUMAN", tmp_path / "human.jsonl")
    app = FastAPI()
    app.include_router(label.router)
    return TestClient(app)


def test_heldout_items_are_blind(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    held = c.get("/label/items", params={"set": "heldout"}).json()["items"]
    assert held == [
        {"event_id": "CHG-000002", "human": None}
    ]  # no system_type, no prelabel even if one exists
    cal = c.get("/label/items", params={"set": "calibration"}).json()["items"][0]
    assert cal["system_type"] == "construction" and cal["prelabel"]["real"] == "no"


def test_save_validates_and_latest_label_wins(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert (
        c.post("/label/CHG-999999", json={"real": "yes", "true_type": "water", "labeller": "a"}).status_code
        == 404
    )
    assert c.post("/label/CHG-000002", json={"real": "maybe", "labeller": "a"}).status_code == 422
    assert (
        c.post("/label/CHG-000002", json={"real": "yes", "true_type": "lake", "labeller": "a"}).status_code
        == 422
    )
    c.post("/label/CHG-000002", json={"real": "yes", "true_type": "water", "labeller": "a"})
    c.post("/label/CHG-000002", json={"real": "no", "labeller": "b"})
    item = c.get("/label/items", params={"set": "heldout"}).json()["items"][0]
    assert item["human"]["real"] == "no" and item["human"]["labeller"] == "b"
