"""eval/run.py metrics: geometry matching, ranking metrics, change/no-change confusion."""

import sys
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import run


def test_ranked_hits_credits_each_relevant_once():
    rel = [box(0, 0, 1, 1), box(5, 5, 6, 6)]
    sites = [box(0.5, 0.5, 1.5, 1.5), box(0.2, 0.2, 0.4, 0.4), box(9, 9, 10, 10), box(5.5, 5.5, 7, 7)]
    assert run.ranked_hits(sites, rel) == [0, None, None, 1]


def test_retrieval_metrics_perfect_and_empty():
    m = run.retrieval_metrics([0, 1], n_relevant=2, k=2)
    assert m == {"p@2": 1.0, "recall@2": 1.0, "ap@2": 1.0, "ndcg@2": 1.0}
    z = run.retrieval_metrics([None, None], n_relevant=2, k=2)
    assert z["p@2"] == z["ap@2"] == z["ndcg@2"] == 0.0


def test_retrieval_metrics_rank_matters():
    early = run.retrieval_metrics([0, None, None], 1, 3)
    late = run.retrieval_metrics([None, None, 0], 1, 3)
    assert early["p@3"] == late["p@3"] and early["ndcg@3"] > late["ndcg@3"] and early["ap@3"] > late["ap@3"]


def _ev(eid, d, conf, geom, t="construction"):
    return {
        "event_id": eid,
        "earliest_supported_date": d,
        "confidence": conf,
        "change_type": t,
        "geometry": geom.__geo_interface__,
    }


def test_predict_case_window_and_geometry():
    evs = [
        _ev("A", "2023-05-01", 0.6, box(0, 0, 1, 1)),
        _ev("B", "2023-06-01", 0.9, box(0, 0, 1, 1)),
        _ev("C", "2025-01-01", 0.99, box(0, 0, 1, 1)),
        _ev("D", "2023-06-01", 0.99, box(5, 5, 6, 6)),
    ]
    assert run.predict_case(box(0.5, 0.5, 2, 2), "2023-01-01", "2023-12-31", evs)["event_id"] == "B"
    assert run.predict_case(box(3, 3, 4, 4), "2023-01-01", "2023-12-31", evs) is None


def test_change_metrics_confusion_type_and_date():
    p = {"event_id": "X", "change_type": "construction", "earliest_supported_date": "2023-01-11"}
    rows = [
        {"label": "change", "pred": p, "type": "construction", "earliest_date": "2023-01-01"},
        {"label": "change", "pred": None},
        {"label": "no_change", "pred": p},
        {"label": "no_change", "pred": None},
        {"label": "no_change", "pred": None},
    ]
    m = run.change_metrics(rows)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == pytest.approx(0.5)
    assert m["false_alarm_rate"] == pytest.approx(1 / 3)
    assert m["type_accuracy"] == 1.0 and m["earliest_date_abs_err_days"]["median"] == 10
