"""Offline tests for pn/explain.py (no API calls). Run: .venv/bin/python -m pytest tests/test_explain.py
or .venv/bin/python tests/test_explain.py"""
import json
import os
import shutil
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import make_synthetic_detail as syn  # noqa: E402
from pn import explain as ex  # noqa: E402


def _run():
    d = tempfile.mkdtemp(prefix="pn_explain_")
    syn.main(d)
    return d


def test_packet_blinding_and_split():
    rd = _run()
    det = ex.load_detail(rd, "L10/N100")
    pk = ex.build_packet(det, "blind")
    val_ids = {e["ex_id"] for e in det["examples"] if e["split"] == "validation"}
    assert 20 <= len(pk["windows"]) <= 40
    assert not val_ids & {w["ex_id"] for w in pk["windows"]}
    low = pk["text"].lower()
    for bad in ("openness", "neuroticism", "o_amp", "o_sup", "ctrl", "amplifier", "l10_n100", "top_pos", "o05"):
        assert bad not in low, bad
    assert "⟦" in pk["text"] and "FUTURE" in pk["text"]
    cats = Counter(w["category"] for w in pk["windows"])
    assert cats["high_pos"] and cats["high_neg"] and cats["random"] and cats["moderate"]
    assert len({w["domain"] for w in pk["windows"]}) >= 4
    shutil.rmtree(rd)


def test_shuffle_preserves_values_and_breaks_link():
    rd = _run()
    det = ex.load_detail(rd, "L10/N100")
    real = ex.build_packet(det, "blind")
    sh = ex.build_packet(det, "shuffled")
    assert sh["hash"] != real["hash"]
    vals = sorted(w["focus_value"] for w in sh["windows"])
    true = sorted(w["true_focus_value"] for w in sh["windows"])
    assert vals == true
    water_tokens = [w for w in sh["windows"] if w["focus_token"].strip() in syn.WATER]
    assert any(w["focus_value"] < 1.0 for w in water_tokens)
    assert "steering" not in sh["text"]
    shutil.rmtree(rd)


def test_ambiguous_mixes_two_neurons():
    rd = _run()
    pk = ex.build_ambiguous_packet(ex.load_detail(rd, "L10/N100"), ex.load_detail(rd, "L12/N200"))
    assert Counter(w["neuron"] for w in pk["windows"]).keys() == {"L10/N100", "L12/N200"}
    assert "L12" not in pk["text"]
    shutil.rmtree(rd)


def _fake_output(packet):
    wids = [w["wid"] for w in packet["windows"]]
    hi = [w["wid"] for w in packet["windows"] if w["category"] == "high_pos"]
    return {"label": "water words", "trigger_hypothesis": "Fires on words about water; negative on dryness words.",
            "output_hypothesis": "Promotes water vocabulary.", "alternatives": ["liquids", "nature"],
            "adapter_changes": "state C scales positive firing up", "supporting": hi, "contradicting": [wids[-1], "W99"],
            "proposed_tests": [
                {"discriminates": "water word vs neutral", "texts": [
                    {"text": "We swam in the lake all day.", "expected": "high", "target": "lake"},
                    {"text": "We sat in the office all day.", "expected": "low", "target": ""}]},
                {"discriminates": "dry vs neutral", "texts": [
                    {"text": "The desert was hot.", "expected": "negative", "target": "desert"},
                    {"text": "The room was hot.", "expected": "low", "target": ""}]},
                {"discriminates": "wrong prediction (should fail)", "texts": [
                    {"text": "The cat slept.", "expected": "high", "target": "cat"},
                    {"text": "The river ran.", "expected": "low", "target": ""}]},
            ], "confidence": "medium", "no_coherent_hypothesis": False}


def test_card_append_and_checker():
    rd = _run()
    det = ex.load_detail(rd, "L10/N100")
    pk = ex.build_packet(det, "blind")
    out = _fake_output(pk)
    assert ex.validate_labeler_output(out) == []
    bad = dict(out, confidence="sure")
    assert ex.validate_labeler_output(bad)
    res = {"model": "fake", "parsed": out, "usage": {"input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0,
                                                    "cost_usd": 0.0}}
    ver = ex.to_card_version(res, pk)
    assert ver["invalid_citations"] == ["W99"]
    assert all(s.startswith("L10_N100:e") for s in ver["supporting"])
    assert ex.append_card_version(rd, "L10/N100", ver) == 1
    assert ex.append_card_version(rd, "L10/N100", ver) == 2
    plan = ex.measurement_plan(ex.load_card(rd, "L10/N100"), det)
    assert sum(i["kind"] == "test" for i in plan) == 6 and any(i["kind"] == "validation" for i in plan)

    def keyword_sim(version, windows):
        return [8.0 if w["focus_token"].strip().lower() in syn.WATER else
                (-5.0 if w["focus_token"].strip().lower() in syn.DRY else 0.0) for w in windows]
    entries = ex.check_card(rd, "L10/N100", syn.fake_measure, simulate_fn=keyword_sim, source="fake_measure")
    results = [e["result"] for e in entries]
    assert results[:3] == ["pass", "pass", "fail"], results
    assert entries[-1]["test_id"].endswith("validation") and entries[-1]["metrics"]["spearman"] > 0.8
    card = ex.load_card(rd, "L10/N100")
    assert len(card["versions"]) == 2 and len(card["tests"]) == 4 and card["status"] == "mixed"
    shutil.rmtree(rd)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
