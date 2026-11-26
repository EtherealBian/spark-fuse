"""CPU-only protocol and counterexamples; never inference measurements."""

import copy, io, json, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from natural235 import CAP, KEYS, audit, collect, compare, digest, invoke, profile, verify_inputs

EOG = [151645]


def fixture():
    o = dict(inputs={}, witnesses={}, streams=[], warmup_excluded=1)
    for i, k in enumerate(KEYS):
        n = int(k.split(":")[0])
        ids = [i] * n
        o["inputs"][k] = ids
        o["witnesses"][k] = dict(
            input_sha256=digest(ids),
            profile=profile(False),
            processed=n,
            tokens_evaluated=n,
            cached=0,
            prompt_ms=5.0,
            predicted=3,
            tokens_predicted=3,
            stop="eos",
            ids=[10, 20, 151645],
            content=" OK",
        )
    for i, k in enumerate(KEYS + KEYS[::-1]):
        w = o["witnesses"][k]
        o["streams"].append(
            {q: v for q, v in w.items() if q not in ("ids", "content")}
            | dict(
                case=k,
                rep=i // 4,
                profile=profile(True),
                events=[[1, 1, 10, " "], [5, 2, 20, "OK"], [7, 3, 151645, ""]],
                end_ns=8,
            )
        )
    return o


class Panel(unittest.TestCase):
    def test_first_visible_not_first_id_and_positive_timing(self):
        b = fixture()
        blocks = [copy.deepcopy(b) for _ in range(4)]
        for i in (0, 3):
            for r in blocks[i]["streams"]:
                for event in r["events"]:
                    event[0] *= 2
                r["end_ns"] *= 2
        c = compare(blocks, EOG)
        self.assertTrue(c["qualified_for_tested_inputs"])
        self.assertEqual(c["block_observations"][1][KEYS[0]][0]["first_visible_ms"], 5e-6)

    def test_ordinal_gap_not_imputed(self):
        s = fixture()
        r = s["streams"][0]
        r["events"].pop(0)
        r["events"][0][3] = " OK"
        c = audit(s, EOG)[KEYS[0]][0]
        self.assertTrue(c["delivery_matches_witness"])
        self.assertFalse(c["all_sampled_ids_observed"])
        self.assertEqual((c["delivered_ids"], c["backend_predicted"]), (2, 3))
        self.assertFalse(compare([s] * 4, EOG)["exact_output_gate"])

    def test_wrong_counter_and_framing_rejected(self):
        for mutate in (
            lambda r: r.update(cached=1),
            lambda r: r.update(stop="limit"),
            lambda r: r["events"][0].__setitem__(1, True),
            lambda r: r["events"][1].__setitem__(1, 1),
            lambda r: r["events"][0].__setitem__(0, 40),
            lambda r: r["events"][0].__setitem__(2, 151645),
            lambda r: r["profile"].update(ignore_eos=True),
        ):
            s = fixture()
            mutate(s["streams"][0])
            with self.assertRaises(ValueError):
                audit(s, EOG)
        for msg in (b"data: [DONE]\n", b'data: {"stop": false}\n', b": ping\n"):
            with self.assertRaises(ValueError):
                collect(io.BytesIO(msg), lambda: 1)

    def test_divergence_is_a_reported_negative_not_an_invalid_run(self):
        blocks = [fixture() for _ in range(4)]
        s = blocks[2]
        s["witnesses"][KEYS[0]]["ids"][0] = 22
        self.assertFalse(compare(blocks, EOG)["exact_output_gate"])
        s = fixture()
        s["streams"][0]["events"][1][3] = ""
        s["streams"][0]["events"][0][3] = " "
        c = audit(s, EOG)[KEYS[0]][0]
        self.assertIsNone(c["first_visible_ms"])
        self.assertFalse(c["delivery_matches_witness"])

    def test_disjoint_native_oracle_and_literal_score(self):
        with self.assertRaises(ValueError):
            verify_inputs(fixture()["inputs"], [128247, 151643, 151645, 151662, 151663, 151664])
        b = fixture()
        b["witnesses"][KEYS[0]]["content"] = " AMBER\n"
        b["witnesses"][KEYS[2]]["content"] = "amber"
        c = compare([b] * 4, EOG)
        self.assertEqual(c["cases"][KEYS[0]]["case_sensitive_label_match_by_block"], [True] * 4)
        self.assertEqual(c["cases"][KEYS[2]]["case_sensitive_label_match_by_block"], [False] * 4)
        self.assertIsNone(c["cases"][KEYS[1]]["case_sensitive_label_match_by_block"])

    def test_limit_is_not_eos(self):
        s = fixture()
        for w in s["witnesses"].values():
            w.update(
                stop="limit",
                tokens_predicted=CAP,
                predicted=CAP,
                ids=[20] * CAP,
                content="OK" * CAP,
            )
        for r in s["streams"]:
            r.update(
                stop="limit",
                tokens_predicted=CAP,
                predicted=CAP,
                events=[[j, j, 20, "OK"] for j in range(1, CAP + 1)],
                end_ns=CAP + 1,
            )
        c = compare([s] * 4, EOG)
        self.assertFalse(c["observed_eos_gate"])
        self.assertTrue(c["exact_output_gate"])

    def test_native_http_natural_profile(self):
        captured = []

        class Server(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append(p)
                common = dict(index=0, id_slot=0, tokens_evaluated=len(p["prompt"]))
                end = dict(
                    **common,
                    stop=True,
                    truncated=False,
                    tokens_predicted=2,
                    stop_type="eos",
                    timings=dict(
                        cache_n=0, prompt_n=len(p["prompt"]), predicted_n=2, prompt_ms=2.0
                    ),
                )
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/event-stream" if p["stream"] else "application/json"
                )
                self.end_headers()
                if p["stream"]:
                    frames = [
                        dict(
                            common,
                            id_slot=-1,
                            stop=False,
                            tokens_predicted=i,
                            tokens=[t],
                            content=c,
                        )
                        for i, t, c in ((1, 20, "OK"), (2, 151645, ""))
                    ] + [dict(end, tokens=[], content="")]
                    self.wfile.write(
                        b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames)
                        + b"data: [DONE]\n\n"
                    )
                else:
                    self.wfile.write(
                        json.dumps(dict(end, tokens=[20, 151645], content="OK")).encode()
                    )

        with HTTPServer(("127.0.0.1", 0), Server) as h:
            t = threading.Thread(target=h.serve_forever, daemon=True)
            t.start()
            try:
                a = invoke(h.server_port, [1, 2], True)
                b = invoke(h.server_port, [1, 2], False)
            finally:
                h.shutdown()
                t.join(2)
        self.assertEqual([e[2] for e in a["events"]], b["ids"])
        self.assertEqual([v["stream"] for v in captured], [True, False])
        self.assertTrue(
            all(
                not set(v) & {"ignore_eos", "logit_bias", "stop"} and v["n_predict"] == 48
                for v in captured
            )
        )
