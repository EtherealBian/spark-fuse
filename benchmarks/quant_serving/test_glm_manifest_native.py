import contextlib
import copy
import io
import unittest
from glm_manifest import LIMIT, audit, common, make, run, tasks


class NativeFixture:
    # Byte tokenizer only for protocol/oracle tests; never used for measurements.
    def __init__(self):
        self.previous = []
        self.eog = 999
        self.calls = []

    def __call__(self, path, p):
        if path == "apply-template":
            return {
                "prompt": "SYS\n"
                + p["messages"][0]["content"]
                + "\nUSER\n"
                + p["messages"][1]["content"]
                + "\nASSISTANT\n"
            }
        if path == "tokenize":
            assert p["parse_special"] is True and p["add_special"] is False
            return {"tokens": list(p["content"].encode())}
        if path == "detokenize":
            return {"content": bytes(p["tokens"]).decode()}
        assert (
            path == "completion"
            and set(p)
            == {
                "prompt",
                "n_predict",
                "id_slot",
                "cache_prompt",
                "temperature",
                "seed",
                "stream",
                "return_tokens",
            }
            and p["n_predict"] == LIMIT
        )
        self.calls.append(p["cache_prompt"])
        ids = p["prompt"]
        text = bytes(ids).decode()
        key = text.split("What is the code for ")[-1].split("?")[0]
        val = next(s.split(" | ")[1] for s in text.splitlines() if s.startswith(key + " | "))
        content = " " + val
        out = list(content.encode()) + [self.eog]
        hit = common(ids, self.previous) if p["cache_prompt"] else 0
        self.previous = ids + out
        return dict(
            id_slot=0,
            content=content,
            tokens=out,
            stop_type="eos",
            truncated=False,
            tokens_evaluated=len(ids),
            timings=dict(
                prompt_n=len(ids) - hit,
                cache_n=hit,
                predicted_n=len(out),
                prompt_ms=0,
                predicted_ms=0,
            ),
        )


class RevisionCanary(unittest.TestCase):
    def setUp(self):
        self.fake = NativeFixture()
        self.checkpoints = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.raw = run(self.fake, [999], lambda o: self.checkpoints.append(len(o["rows"])))

    def test_masks_absent_and_one_field_changes(self):
        self.assertEqual(self.checkpoints, list(range(1, 17)))
        self.assertEqual(len(self.fake.calls), 32)
        self.assertEqual(sum(self.fake.calls), 8)
        for n, p, v in tasks():
            new, answer = make(n, p, v)
            old, prev = make(n, p, 1 - v)
            self.assertNotEqual(answer, prev)
            self.assertEqual(new[0], old[0])
            lines = list(
                zip(new[1]["content"].splitlines(), old[1]["content"].splitlines(), strict=True)
            )
            diff = [(a, b) for a, b in lines if a != b]
            self.assertEqual(len(diff), 1)
            self.assertEqual(diff[0][0].split(" | ")[0], diff[0][1].split(" | ")[0])
        result = audit(self.raw, [999])
        self.assertEqual(result["exact_token_pairs"], 8)
        self.assertEqual(result["totals"]["cold"]["correct"], 8)

    def test_wrong_answer_is_data_not_a_discarded_run(self):
        x = self.raw["rows"][0]
        r = x["target"]
        old = make(16, "early", 1)[1]
        r["ids"] = list(old.encode()) + [999]
        r.update(text=old, decoded=old, predicted=len(r["ids"]))
        result = audit(self.raw, [999])
        self.assertEqual(result["totals"][x["arm"]]["stale"], 1)
        self.assertEqual(result["exact_token_pairs"], 7)
        r["text"] = r["decoded"] = "unrecognized!!"[:LIMIT]
        r["ids"] = list(r["text"].encode())
        r["predicted"] = LIMIT
        r["stop"] = "limit"
        result = audit(self.raw, [999])
        self.assertEqual(result["totals"][x["arm"]]["other"], 1)

    def test_reject_misbinding_counters_and_unreconciled_output(self):
        for field, value in [
            ("input_sha256", "0" * 64),
            ("slot", 1),
            ("cached", 1),
            ("processed", 3),
            ("evaluated", 0),
            ("predicted", True),
            ("truncated", True),
            ("stop", "limit"),
            ("decoded", "different"),
            ("wall_ms", float("inf")),
        ]:
            with self.subTest(field=field):
                a = copy.deepcopy(self.raw)
                a["rows"][0]["target"][field] = value
                with self.assertRaises(ValueError):
                    audit(a, [999])
        for kind in ("switch", "text", "ids", "mask"):
            with self.subTest(kind=kind):
                a = copy.deepcopy(self.raw)
                if kind == "switch":
                    a["rows"][0], a["rows"][1] = a["rows"][1], a["rows"][0]
                elif kind == "text":
                    a["inputs"]["16:late:0"]["rendered"] += "unbound"
                elif kind == "ids":
                    a["inputs"]["96:late:0"]["input_ids"][4] += 1
                else:
                    a["rows"][0]["target"]["ids"][0] = 999
                with self.assertRaises(ValueError):
                    audit(a, [999])
