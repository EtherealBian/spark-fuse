import copy
import json
import unittest
from stream_wave import audit, digest, SHAPES, stream, TOKENS


def fixture():
    out = {"inputs": {}, "waves": []}
    for i, (key, sizes) in enumerate(SHAPES.items()):
        for rep in (0, 1):
            spec = [[100 * i + 10 * rep + s] * n for s, n in enumerate(sizes)]
            out["inputs"][f"{key}:{rep}"] = spec
            for mode in ("serial", "parallel") if (i + rep) % 2 == 0 else ("parallel", "serial"):
                rows = []
                for j, s in enumerate([rep, 1 - rep]):
                    t = 250 * j if mode == "serial" else 3 * j
                    rows.append(
                        dict(
                            slot=s,
                            terminal_slot=s,
                            partial_slot_ids={"-1": TOKENS},
                            input_sha256=digest(spec[s]),
                            start_ns=t,
                            end_ns=t + 170,
                            events=[[t + 10 + k, 300 + k] for k in range(TOKENS)],
                            stop="limit",
                            predicted=TOKENS,
                            processed=sizes[s],
                            cached=0,
                        )
                    )
                out["waves"].append(dict(key=key, rep=rep, mode=mode, rows=rows))
    return out


class StreamEvidence(unittest.TestCase):
    def test_reads_native_records_and_preserves_ids(self):
        def line(**v):
            return b"data: " + json.dumps({"index": 0, **v}).encode() + b"\n"

        partial = line(id_slot=-1, stop=False, tokens=[11])
        end = line(
            id_slot=1,
            stop=True,
            tokens=[],
            truncated=False,
            stop_type="limit",
            timings={"predicted_n": 1, "prompt_n": 2, "cache_n": 0},
        )
        clock = iter([50, 70])
        v = stream([b"\n", partial, end, b"data: [DONE]"], lambda: next(clock), 1)
        self.assertEqual((v["events"], v["end_ns"]), ([[50, 11]], 70))
        self.assertEqual((v["terminal_slot"], v["partial_slot_ids"]), (1, {"-1": 1}))
        for bad in (
            [line(id_slot=0, stop=False, tokens=[11]), end],
            [line(id_slot=-2, stop=False, tokens=[11]), end],
            [partial, end.replace(b'"id_slot": 1', b'"id_slot": -1')],
            [partial, end.replace(b'"id_slot": 1', b'"id_slot": 0')],
            [partial, end.replace(b'"index": 0', b'"index": 1')],
            [partial.replace(b"[11]", b"[]"), end],
            [partial],
            [b"data: [DONE]"],
            [end, partial],
            [b"data: " + b'{"id_slot":0,"stop":false}'],
        ):
            with self.assertRaises(ValueError):
                stream(bad, lambda: 1, 1)

    def test_overlap_timestamps_counters_and_parity(self):
        d = fixture()
        base = audit(d, {151643})
        self.assertEqual(base["cases"]["balanced"]["exact_output_pairs"], 4)
        d["waves"][1]["rows"][0]["events"][0][1] = 120
        self.assertEqual(audit(d, {151643})["cases"]["balanced"]["exact_output_pairs"], 3)
        for field, v in [
            ("start_ns", 10000),
            ("end_ns", 0),
            ("cached", 4),
            ("terminal_slot", -1),
            ("partial_slot_ids", {"-1": TOKENS - 1}),
            ("partial_slot_ids", {"1": TOKENS}),
            ("stop", "eos"),
            ("processed", 4),
            ("predicted", 13),
            ("input_sha256", "x"),
        ]:
            x = copy.deepcopy(d)
            x["waves"][0]["rows"][0][field] = v
            with self.assertRaises(ValueError):
                audit(x, {151643})
        x = copy.deepcopy(d)
        x["waves"][1]["rows"][0]["events"][0][1] = 151643
        with self.assertRaises(ValueError):
            audit(x, {151643})
        x = copy.deepcopy(d)
        r = x["waves"][1]["rows"][1]
        r["start_ns"] += 300
        r["end_ns"] += 300
        for e in r["events"]:
            e[0] += 300
        with self.assertRaises(ValueError):
            audit(x, {151643})

    def test_queued_follower_includes_leader_but_request_ttft_does_not(self):
        raw = fixture()
        c = audit(raw, set())["cases"]["balanced"]
        self.assertAlmostEqual(c["serial"]["slot1_ttft_ms"], 10 / 1e6)
        w = c["paired_waves"][0]
        self.assertEqual(w["serial_leader"], 0)
        self.assertAlmostEqual(w["ready_to_first_saved_ms_by_slot"]["1"], 247 / 1e6)
        self.assertAlmostEqual(w["ready_to_first_saved_ms_by_slot"]["0"], 0.0)
        raw["waves"][1]["rows"][1]["events"][2][1] = 2000
        self.assertEqual(
            audit(raw, set())["cases"]["balanced"]["paired_waves"][0][
                "first_different_token_by_slot"
            ],
            {"0": None, "1": 3},
        )
