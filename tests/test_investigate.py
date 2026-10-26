"""Offline tests for pn/investigate.py (no API calls). Run: .venv/bin/python tests/test_investigate.py
(or with pytest). The batched-measurement test uses the tiny fixture (tests/make_tiny.py) when ../work/tiny exists."""
import asyncio
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import make_synthetic_detail as syn  # noqa: E402
from pn import explain as ex  # noqa: E402
from pn import investigate as inv  # noqa: E402

TINY = os.path.join(os.path.dirname(REPO), "work", "tiny")
PROGRESS_KEYS = {"started", "updated", "n_total", "n_done", "n_running", "n_failed", "cost_usd", "cost_cap_usd",
                 "est_cost_per_neuron", "eta_minutes", "by_kind", "by_trait"}
FINISHED_KEYS = {"id", "kind", "trait", "nickname", "one_liner", "status", "passes", "n_final_tests", "rounds",
                 "cost_usd", "finished_at"}
COUNT_KEYS = {"done", "held_up", "partly", "not_held", "no_coherent"}


def _run():
    d = tempfile.mkdtemp(prefix="pn_inv_")
    syn.main(d)
    return d


def _items():
    return [{"id": "L10/N100", "layer": 10, "n": 100, "kind": "trait", "trait": "openness"},
            {"id": "L12/N200", "layer": 12, "n": 200, "kind": "trait", "trait": "neuroticism"},
            {"id": "L14/N300", "layer": 14, "n": 300, "kind": "random", "trait": None}]


def _investigator(rd, llm=None, **kw):
    kw.setdefault("rounds", 3)
    kw.setdefault("max_cost", 1.0)
    return inv.Investigator(rd, llm or inv.FakeLLM(latency=(0, 0.002)), inv.FuncMeasurer(syn.fake_measure),
                            usage_log=os.path.join(rd, "usage.jsonl"), progress_every=0.05, log=lambda s: None, **kw)


def test_render_and_region():
    full, off = inv.render({"context": "assistant_reply", "user": "Hi", "text": "Oh no."})
    assert full == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nOh no."
    toks = ["<|im_start|>", "user", "\n", "Hi", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>",
            "\n\n", "</think>", "\n\n", "Oh", " no", "."]
    assert inv.region_start(toks, off) == 13
    assert inv.render({"context": "plain", "text": "abc"}) == (inv.SINK + "abc", len(inv.SINK))
    # scoring of a chat text looks only at the reply: the template tokens here carry a huge activation
    m = {"tokens": toks, "acts": [0, 50] + [0] * 11 + [3.0, 0.1, 0], "region_start": 13}
    v = inv.measure_value(m, {"text": "Oh no.", "expected": "high", "target": "Oh"})
    assert v["value"] == 3.0 and v["measured_at"] == "target" and v["max"] == 3.0
    v = inv.measure_value(m, {"text": "Oh no.", "expected": "low", "target": ""})
    assert v["value"] == 3.0


def test_scoring_rule():
    thr = {"p99": 1.0, "p01": -2.0}
    tests = [{"pair": 1, "expected": "high"}, {"pair": 1, "expected": "low"},
             {"pair": 2, "expected": "negative"}, {"pair": 2, "expected": "low"},
             {"pair": 3, "expected": "high"}, {"pair": 3, "expected": "low"}]
    vals = [{"value": 1.5, "max": 1.5, "min": 0}, {"value": 0.4, "max": 0.4, "min": -5},   # low ignores other sign
            {"value": -3, "max": 0, "min": -3}, {"value": 0.5, "max": 0.5, "min": -0.5},
            {"value": 0.2, "max": 0.2, "min": 0}, {"value": 0.3, "max": 0.3, "min": 0}]    # contrast wrong way
    got = [ok for ok, _ in inv.score_texts(tests, vals, thr)]
    assert got == [True, True, True, True, False, False], got
    assert inv.verdict(3, 4) == "held_up" and inv.verdict(2, 4) == "partly" and inv.verdict(1, 4) == "not_held"
    assert inv.verdict(2, 8) == "not_held" and inv.verdict(0, 0) == "no_coherent"


def test_thresholds_fallbacks():
    assert inv.thresholds({"quantiles_base": {"p99": 0.6, "p01": -3.6}})["source"] == "quantiles_base"
    assert inv.thresholds({"natural_range": {"p99": 0.7, "p01": -0.4}})["p99"] == 0.7
    t = inv.thresholds({"examples": [{"acts": {"base": list(range(-50, 51))}}]})
    assert t["p99"] >= 48 and t["p01"] <= -48


def _good_obj():
    tests = []
    for p in range(1, 5):
        tests += [{"pair": p, "context": "plain", "text": f"It was so deep {p}.", "expected": "high", "target": "deep"},
                  {"pair": p, "context": "plain", "text": f"It was deep {p}.", "expected": "low", "target": "x"}]
    return {"decision": "new", "label": "so + adj", "trigger_hypothesis": "t", "output_hypothesis": "o",
            "nickname": "so neuron", "one_liner": "x", "alternatives": ["a"], "adapter_changes": "B+1 stronger",
            "state_effects": [], "test_states": ["state B+1", "Z+1", "C-1", "D+1"], "supporting": ["W01", "W99"],
            "contradicting": [], "confidence": "medium", "no_coherent_hypothesis": False, "tests": tests}


def test_validator_repairs_and_errors():
    v1 = inv.make_validator(1, ["B+1", "C-1", "D+1"], {"W01", "W02"})
    o = _good_obj()
    assert v1(o) == []
    assert o["test_states"] == ["B+1", "C-1"] and o["supporting"] == ["W01"]
    assert all(t["target"] == "" for t in o["tests"] if t["expected"] == "low")
    o = _good_obj()
    o["tests"][0]["target"] = "missing"
    assert any("substring" in e for e in v1(o))
    o = _good_obj()
    o["tests"][1]["expected"] = "high"
    o["tests"][1]["target"] = "deep"
    assert any("pair 1" in e for e in v1(o))
    o = _good_obj()
    o["tests"] = o["tests"][:6]
    assert v1(o)
    o = _good_obj()
    o["tests"][2]["context"] = "assistant_reply"
    assert any("user message" in e for e in v1(o))
    v2 = inv.make_validator(2, ["B+1"], set())
    assert v2({"decision": "give up", "label": "none", "trigger_hypothesis": "", "tests": None}) == []
    o = _good_obj()
    assert any("decision" in e for e in v2(o))


def test_full_loop_resume_and_formats():
    rd = _run()
    try:
        card_before = ex.load_card(rd, "L10/N100")
        card_before["versions"].append({"v": 1, "kind": "automated", "label": "old"})
        ex.save_card(rd, card_before)
        items = _items()
        llm = inv.FakeLLM(latency=(0, 0.002))
        prog = asyncio.run(_investigator(rd, llm).run(items))
        assert set(prog) == PROGRESS_KEYS
        assert prog["n_done"] == 3 and prog["n_failed"] == 0 and prog["n_running"] == 0
        assert set(prog["by_kind"]) == {"trait", "random"} and set(prog["by_trait"]) == {"openness", "neuroticism"}
        for d in list(prog["by_kind"].values()) + list(prog["by_trait"].values()):
            assert set(d) == COUNT_KEYS
        assert prog["by_kind"]["random"]["done"] == 1
        idir = os.path.join(rd, "investigations")
        rows = [json.loads(x) for x in open(os.path.join(idir, "finished.jsonl"))]
        assert len(rows) == 3 and all(set(r) == FINISHED_KEYS for r in rows)
        assert {r["status"] for r in rows} <= set(inv.STATUS_KEYS)
        assert json.load(open(os.path.join(idir, "progress.json")))["n_done"] == 3
        # card: earlier versions kept, investigation version appended, tests carry rule and rounds
        card = ex.load_card(rd, "L10/N100")
        assert card["versions"][0]["label"] == "old"
        v = card["versions"][-1]
        assert v["kind"] == "investigation" and v["author"] == "fake/canned (investigator)"
        assert v["nickname"] and v["one_liner"] and v["v"] == 2
        it = [t for t in card["tests"] if t.get("rule") == "investigation"]
        assert {t["round"] for t in it} == {1, 2, 3} and all(t["card_version"] == 2 for t in it)
        assert sum(t["final"] for t in it) == 4
        row = next(r for r in rows if r["id"] == "L10/N100")
        assert card["status"] == inv.STATUS_TO_CARD[row["status"]]
        d = json.load(open(os.path.join(idir, "L10_N100.json")))
        assert d["finished"] and len(d["rounds"]) == 3 and d["rounds"][0]["attempts"][0]["content"]
        assert all("results" in r and len(r["results"]) == 8 for r in d["rounds"])
        assert "AVAILABLE:" not in d["rounds"][1]["user_tail"]   # stored tail only, not the packet again
        assert "Round 1 results" in d["rounds"][1]["user_tail"] and "Round 2 results" in d["rounds"][2]["user_tail"]
        assert "Round 1 results" not in d["rounds"][0]["user_tail"]
        assert "final round" in d["rounds"][2]["user_tail"]
        # no real lettered state name leaks into what the model sees
        for r in d["rounds"]:
            assert "o_amp" not in r["user_tail"] and "n_amp" not in r["user_tail"]
        usage = [json.loads(x) for x in open(os.path.join(rd, "usage.jsonl"))]
        assert all(u["component"] == "pn.investigate" for u in usage)
        n_calls = llm.calls
        # resume: nothing left to do, no calls
        llm2 = inv.FakeLLM(latency=(0, 0.002))
        prog2 = asyncio.run(_investigator(rd, llm2).run(items))
        assert llm2.calls == 0 and prog2["n_done"] == 3 and n_calls > 0
        # a finished file without its finished.jsonl line is reconciled; an unfinished one is redone
        lines = open(os.path.join(idir, "finished.jsonl")).read().splitlines()
        open(os.path.join(idir, "finished.jsonl"), "w").write("\n".join(lines[:1]) + "\n")
        d = json.load(open(os.path.join(idir, "L14_N300.json")))
        d["finished"] = False
        json.dump(d, open(os.path.join(idir, "L14_N300.json"), "w"))
        llm3 = inv.FakeLLM(latency=(0, 0.002))
        prog3 = asyncio.run(_investigator(rd, llm3).run(items))
        assert prog3["n_done"] == 3 and llm3.calls > 0
        ids = [json.loads(x)["id"] for x in open(os.path.join(idir, "finished.jsonl"))]
        assert sorted(ids) == sorted(i["id"] for i in items) and len(ids) == 3
    finally:
        shutil.rmtree(rd)


def test_give_up_and_invalid_retry_paths():
    rd = _run()
    try:
        llm = inv.FakeLLM(latency=(0, 0.002), give_up={"L14/N300"}, invalid_first={"L10/N100"})
        asyncio.run(_investigator(rd, llm).run(_items()))
        d = json.load(open(os.path.join(rd, "investigations", "L14_N300.json")))
        assert d["status"] == "no_coherent" and d["passes"] is None and len(d["rounds"]) == 2
        assert ex.load_card(rd, "L14/N300")["status"] == "untested"
        assert ex.load_card(rd, "L14/N300")["versions"][-1]["no_coherent_hypothesis"] is True
        d = json.load(open(os.path.join(rd, "investigations", "L10_N100.json")))
        assert len(d["rounds"][0]["attempts"]) == 2 and d["rounds"][0]["attempts"][0]["errors"]
        assert d["status"] != "no_coherent" and len(d["rounds"]) == 3
    finally:
        shutil.rmtree(rd)


def test_cost_cap_stops_launching():
    rd = _run()
    try:
        llm = inv.FakeLLM(latency=(0, 0.002), cost_per_call=0.01)
        # each neuron costs about 0.02-0.04; cap 0.05 with a 0.03 estimate and concurrency 1 allows one or two
        prog = asyncio.run(_investigator(rd, llm, concurrency=1, max_cost=0.05).run(_items(), prior_estimate=0.03))
        assert 1 <= prog["n_done"] < 3 and prog["cost_usd"] <= 0.05 * inv.HARD_CEILING
    finally:
        shutil.rmtree(rd)


class _FailingLLM(inv.FakeLLM):
    async def chat(self, messages, meta=None):
        if meta.get("neuron") == "L12/N200":
            raise inv.LLMError("boom")
        return await super().chat(messages, meta)


def test_failure_is_recorded_and_retried():
    rd = _run()
    try:
        prog = asyncio.run(_investigator(rd, _FailingLLM(latency=(0, 0.001))).run(_items()))
        assert prog["n_failed"] == 1 and prog["n_done"] == 2
        d = json.load(open(os.path.join(rd, "investigations", "L12_N200.json")))
        assert not d["finished"] and "boom" in d["error"]
        d["usage"]["cost_usd"] = 0.123   # pretend the failed attempt had spent money
        json.dump(d, open(os.path.join(rd, "investigations", "L12_N200.json"), "w"))
        llm = inv.FakeLLM(latency=(0, 0.001))
        prog = asyncio.run(_investigator(rd, llm).run(_items()))
        assert prog["n_done"] == 3 and prog["n_failed"] == 0 and llm.calls >= 2
        row = [json.loads(x) for x in open(os.path.join(rd, "investigations", "finished.jsonl"))][-1]
        assert row["id"] == "L12/N200" and row["cost_usd"] > 0.123
    finally:
        shutil.rmtree(rd)


def test_neuron_list_formats():
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "a.json")
        json.dump([[17, 7573], [3, 5], [17, 7573]], open(p, "w"))
        idx = {"L3/N5": {"kind": "random_control", "selected_for": []},
               "L17/N7573": {"kind": "candidate", "selected_for": ["neuroticism"]}}
        got = inv.load_neuron_list(p, idx)
        assert [g["id"] for g in got] == ["L17/N7573", "L3/N5"]
        assert got[0]["kind"] == "trait" and got[0]["trait"] == "neuroticism" and got[1]["kind"] == "random"
        got = inv.load_neuron_list(os.path.join(REPO, "runs", "v2_work", "pilot.json"))
        assert len(got) == 128 and {g["kind"] for g in got} <= {"trait", "random"}
    finally:
        shutil.rmtree(d)


def test_decode_letters():
    assert inv.decode_letters("state B+1 is stronger than C-0.5", {"n_amp": "B", "o_sup": "C"}) == \
        "n_amp+1 is stronger than o_sup-0.5"


def test_batched_matches_measurer_on_tiny():
    if not os.path.isdir(TINY):
        print("  (skipped: no tiny fixture; run tests/make_tiny.py)")
        return
    sm = inv.build_state_model(os.path.join(TINY, "base"), os.path.join(TINY, "adapters"), None, "cpu")
    bm = inv.BatchedMeasurer(sm, max_batch_tokens=64)   # small budget forces several chunks
    rep = inv.validate_batched(bm, [(0, 5), (3, 191), (2, 17)], ["base", "n_amp+1", "o_amp-0.5|ctrl+1"], tol=0.02)
    assert rep["ok"], rep
    # the async path batches across neurons and states and returns the same numbers as one-at-a-time requests

    async def go():
        bm.start()
        items = [{"context": "plain", "text": "The cat sat."},
                 {"context": "assistant_reply", "user": "Hi", "text": "Oh no, that is so hard to hear."}]
        jobs = [bm.measure(l, n, s, items) for l, n in [(0, 5), (3, 191)] for s in ["base", "n_amp+1"]]
        together = await asyncio.gather(*jobs)
        alone = [await bm.measure(l, n, s, items) for l, n in [(0, 5), (3, 191)] for s in ["base", "n_amp+1"]]
        return together, alone
    together, alone = asyncio.run(go())
    for a, b in zip(together, alone):
        for x, y in zip(a, b):
            assert x["tokens"] == y["tokens"] and x["region_start"] == y["region_start"]
            assert x["acts"][0] == 0.0
            assert max(abs(p - q) for p, q in zip(x["acts"], y["acts"])) <= 0.02 * max(max(map(abs, y["acts"])), 0.1)
    assert together[0][1]["region_start"] > 10
    assert bm.stats["forwards"] > 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
