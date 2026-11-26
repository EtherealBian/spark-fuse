"""Loopback CPU protocol test. No fixture carries latency into a performance report."""

from contextlib import redirect_stdout
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
from time import sleep
from unittest import TestCase, mock
from prefix_replay import digest
from live72 import run
from live72_review import check_arm, inputs, review


class Wire(TestCase):
    @classmethod
    def setUpClass(cls):
        ref = Path("reports/qwen72b/assignment")
        cls.ref = ref
        cls.cases = inputs(ref)
        cls.eog = json.loads((ref / "runtime.json").read_bytes())["eog_token_ids"]
        idx = {}
        for c in cls.cases:
            h = next(x for x in c["arms"] if x["mode"] == "joint")
            for w in [*h["preparations"], *(z["response"] for z in h["requests"])]:
                idx[w["input_sha256"], w["observed_slot"], w["cache_prompt"]] = w

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                slot = body["id_slot"]
                w = idx[digest(body["prompt"]), slot, body["cache_prompt"]]
                tok = w["ids"]
                assert self.path == "/completion" and body["logit_bias"] == [
                    [t, False] for t in cls.eog
                ]
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/event-stream" if body["stream"] else "application/json"
                )
                self.end_headers()
                if body["stream"]:
                    assert len(tok) == 1 and slot == 1 and body["n_predict"] == 16

                    def send(obj):
                        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                        self.wfile.flush()

                    for _ in range(16):
                        send(dict(stop=False, index=0, id_slot=-1, tokens=tok))
                        sleep(0.0025)
                    send(
                        dict(
                            stop=True,
                            index=0,
                            id_slot=1,
                            tokens=[],
                            truncated=False,
                            stop_type="limit",
                            timings=dict(predicted_n=16, prompt_n=1024, cache_n=0),
                        )
                    )
                else:
                    if body["cache_prompt"]:
                        sleep(0.020)
                    self.wfile.write(
                        json.dumps(
                            dict(
                                id_slot=slot,
                                tokens=tok,
                                stop_type="limit",
                                timings=dict(
                                    predicted_n=len(tok),
                                    prompt_n=w["processed"],
                                    cache_n=w["reused"],
                                    prompt_ms=0.01,
                                    predicted_ms=0.01,
                                ),
                            )
                        ).encode()
                    )

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            t = Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                with (
                    tempfile.TemporaryDirectory() as td,
                    mock.patch.dict(os.environ, {"SPARKFUSE_EOG_IDS": json.dumps(cls.eog)}),
                    redirect_stdout(io.StringIO()),
                ):
                    p = Path(td) / "test-only.json"
                    run(server.server_port, ref, p, cls.eog)
                    cls.obj = json.loads(p.read_bytes())
            finally:
                server.shutdown()
                t.join(5)

    def test_full_protocol_and_occupancy(self):
        x = review(self.obj, self.ref, self.eog)
        self.assertTrue(x["matched_including_historical_targets"])
        self.assertEqual(x["timed_targets"], 48)
        self.assertEqual(x["by_first"]["poll-1ms"]["target_router_calls"]["fifo"], 6)
        self.assertGreater(x["by_first"]["fifo"]["target_router_calls"]["poll-1ms"], 6)

    def test_malformed_or_uncontended_cannot_pass(self):
        original = self.obj["warmup"]["fifo"]
        c = self.cases[0]
        for fn in (
            lambda r: r.update(arrival_ns=r["blocker"]["response"]["end_ns"]),
            lambda r: r["broker"].update(wait_calls=0),
            lambda r: r["broker"].update(active_slots=1),
            lambda r: r["targets"][1]["response"].update(reused=0),
            lambda r: r["blocker"]["response"].update(terminal_slot=-1),
            lambda r: r["blocker"]["response"]["events"].pop(),
            lambda r: r["targets"][0].update(start_ns=r["targets"][1]["end_ns"]),
        ):
            r = deepcopy(original)
            fn(r)
            with self.assertRaises(ValueError):
                check_arm(r, c, "fifo", self.eog)

    def test_negative_numeric_result_is_retained(self):
        obj = deepcopy(self.obj)
        obj["pairs"][0]["arms"][0]["row"]["targets"][0]["response"]["ids"][5] = 42
        res = review(obj, self.ref, self.eog)
        self.assertFalse(res["matched_including_historical_targets"])
        self.assertFalse(res["end_to_end_gate"])
        self.assertFalse(res["waiter_cpu_gate"])
        obj["pairs"].pop()
        with self.assertRaises(ValueError):
            review(obj, self.ref, self.eog)
