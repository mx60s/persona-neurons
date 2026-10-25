"""Offline checks for pn/judge.py (no network): python tests/test_judge.py

Uses httpx.MockTransport in place of OpenRouter. Checks blinding (no state/prompt id in any request), common prompts
scored on both traits, caching (a rerun makes no requests), retry on 429 and on unparseable replies, usage logging,
and the summary layout.
"""
import json
import os
import sys
import tempfile

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pn import judge  # noqa: E402

PROMPTS = [
    {"id": "o01", "trait": "openness", "category": "imagination", "common": False, "smoke": True, "text": "Describe a city where the rain falls upward."},
    {"id": "o02", "trait": "openness", "category": "neutral", "common": True, "smoke": True, "text": "How many tablespoons are in a cup?"},
    {"id": "n01", "trait": "neuroticism", "category": "setbacks", "common": False, "smoke": True, "text": "I failed my driving test again."},
]
STATES = ["base", "o_amp+1", "n_amp+1"]


def responses():
    out = []
    for p in PROMPTS:
        for s in STATES:
            if p["trait"] == "openness" and s == "n_amp+1" and not p["common"]:
                continue
            if p["trait"] == "neuroticism" and s == "o_amp+1":
                continue
            # base and o_amp+1 give identical text for o02: should share one judge call
            text = "Sixteen." if p["id"] == "o02" and s != "n_amp+1" else f"reply variant {len(out)}"
            out.append({"prompt_id": p["id"], "state": s, "intervention": None, "text": text, "n_tokens": 5,
                        "finish": "eos", "seed": 0, "decoding": {"do_sample": False, "max_new_tokens": 160}})
    return out


def main():
    calls = {"n": 0, "bodies": [], "fail_once": True, "garble_once": True}

    def handler(request):
        body = json.loads(request.content)
        calls["n"] += 1
        calls["bodies"].append(body)
        if calls["fail_once"]:
            calls["fail_once"] = False
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        content = '{"reasoning": "ok", "score": 5, "coherence": 4}'
        if calls["garble_once"]:
            calls["garble_once"] = False
            content = "I think it is a five"
        return httpx.Response(200, json={"id": f"gen-{calls['n']}", "provider": "Mock",
                                         "choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 30, "cost": 0.0003}})

    judge.random.seed(0)
    orig_sleep = judge.asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    judge.asyncio.sleep = fast_sleep
    with tempfile.TemporaryDirectory() as d:
        beh = os.path.join(d, "behavior")
        os.makedirs(beh)
        json.dump(PROMPTS, open(os.path.join(beh, "prompts.json"), "w"))
        with open(os.path.join(beh, "responses.jsonl"), "w") as f:
            for r in responses():
                f.write(json.dumps(r) + "\n")
        usage = os.path.join(d, "usage.jsonl")
        args = ["--run", d, "--usage-log", usage, "--n-boot", "200"]
        summ = judge.main(args, transport=httpx.MockTransport(handler))

        scores = judge.read_jsonl(os.path.join(beh, "scores.jsonl"))
        n_resp = len(responses())
        n_common = sum(1 for r in responses() if r["prompt_id"] == "o02")
        assert len(scores) == n_resp + n_common, (len(scores), n_resp, n_common)
        assert {s["trait"] for s in scores if s["prompt_id"] == "o02"} == {"openness", "neuroticism"}
        assert all(s["trait"] == "neuroticism" for s in scores if s["prompt_id"] == "n01")
        # unique (content, trait) pairs: identical o02 text under base and o_amp+1 is judged once per trait
        texts = {(r["prompt_id"], r["text"]) for r in responses()}
        expected_unique = sum(2 if pid == "o02" else 1 for pid, _ in texts)
        billed = judge.read_jsonl(usage)
        assert len(billed) == expected_unique + 1, (len(billed), expected_unique)  # +1 for the garbled reply
        assert calls["n"] == expected_unique + 2  # +429 +garbled
        # blinding: no state ids, prompt ids or adapter names in any request
        for b in calls["bodies"]:
            msg = b["messages"][0]["content"]
            for bad in ("o_amp", "n_amp", "base\"", "o01", "n01", "suppressor", "amplifier", "adapter"):
                assert bad not in msg, bad
            assert b["temperature"] == 0 and b["max_tokens"] == 400 and b["reasoning"]["enabled"] is False
            assert b["usage"] == {"include": True}
        assert summ["traits"]["openness"]["o_amp+1"]["basis"] == "own_set"
        assert summ["traits"]["openness"]["n_amp+1"]["basis"] == "common"
        assert "diff_vs_base" in summ["traits"]["neuroticism"]["n_amp+1"]
        assert abs(sum(r["cost_usd"] for r in billed) - 0.0003 * len(billed)) < 1e-9

        # rerun: everything from cache, no new requests
        before = calls["n"]
        judge.main(args, transport=httpx.MockTransport(handler))
        assert calls["n"] == before
        assert len(judge.read_jsonl(os.path.join(beh, "scores.jsonl"))) == len(scores)

    # parser
    assert judge.parse_reply('```json\n{"reasoning": "x", "score": "6", "coherence": 5}\n```')[:2] == (6, 5)
    for bad in ('{"score": 8, "coherence": 3}', '{"score": 4}', "nothing"):
        try:
            judge.parse_reply(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    lo, hi = judge.bootstrap_ci([1, 2, 3, 4, 5, 6, 7], n_boot=500)
    assert lo < 4 < hi
    print("test_judge: all checks passed")


if __name__ == "__main__":
    main()
