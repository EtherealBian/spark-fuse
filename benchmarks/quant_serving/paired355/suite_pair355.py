import copy, unittest
from pair355 import counter_snapshot, counters, reduce_block, schedule
from prefix_replay import digest
from review_pair355 import summarize
from runtime_pair355 import read_log


def snap(p, t, d, b, epoch="10"):
    return counter_snapshot(
        "\n".join(
            "llamacpp:" + k + " " + str(v)
            for k, v in {
                "prompt_tokens_total": p,
                "tokens_predicted_total": t,
                "n_decode_total": d,
                "n_busy_slots_per_decode": format(b / d, ".6g") if d else 0,
                "requests_processing": 0,
                "requests_deferred": 0,
            }.items()
        ),
        epoch,
    )


def fixture(block):
    raw = {
        "inputs": {f"{n}:{j}": [20 + j] * n for n in (128, 1024) for j in range(2)},
        "rows": [
            {"case": f"{n}:{j}", "ids": [40 + j] * 16, "rep": 0}
            for n in (128, 1024)
            for j in range(2)
        ],
    }

    def completion(n, j, slot, output):
        return dict(
            slot=slot,
            input_sha256=digest(raw["inputs"][f"{n}:{j}"]),
            ids=[40 + j] * output,
            stop="limit",
            truncated=False,
            processed=n,
            reused=0,
            predicted=output,
            prompt_ms=0.1,
            decode_ms=0.3,
        )

    data = dict(
        schema="glm355.paired-waves.v1",
        block=block,
        warm=[completion(128, 0, s, 3) for s in (0, 1)],
        waves=[],
    )
    p, t, d, b = 256, 6, 6, 6
    for rep, n, mode, lead in schedule(block):
        before = snap(p, t, d, b)
        p += 2 * n
        t += 32
        d += 20
        b += 20 if mode == "serial" else 32
        rows = []
        for j in range(2):
            start, end = (
                ((0, 10000000) if j == lead else (10000000, 20000000))
                if mode == "serial"
                else ((100, 10500000) if j == lead else (200, 13000000))
            )
            rows.append(
                dict(case=f"{n}:{j}", start_ns=start, end_ns=end, **completion(n, j, j ^ lead, 16))
            )
        data["waves"].append(
            dict(
                rep=rep,
                length=n,
                mode=mode,
                lead=lead,
                rows=rows,
                before=before,
                after=snap(p, t, d, b),
            )
        )
    return raw, data


class Paired(unittest.TestCase):
    def test_full_schedule_both_directions(self):
        blocks = []
        for i in (0, 1):
            raw, r = fixture(i)
            blocks.append(reduce_block(r, raw, [2], i))
        report = summarize(blocks)
        self.assertTrue(report["qualified_for_tested_workload"])
        self.assertEqual(len(report["lengths"]["1024"]["pairs"]), 4)

    def test_throughput_can_delay_first_completion(self):
        bs = []
        for i in (0, 1):
            raw, r = fixture(i)
            for w in r["waves"]:
                if w["mode"] == "together":
                    for c in w["rows"]:
                        c["end_ns"] += 1500000
            bs.append(reduce_block(r, raw, [2], i))
        h = summarize(bs)
        c = h["lengths"]["128"]
        self.assertTrue(c["all_four_pair_makespans_improve_at_least_5pct"])
        self.assertFalse(c["first_completion_within_10pct_all_pairs"])
        self.assertFalse(h["qualified_for_tested_workload"])

    def test_output_divergence_retained(self):
        bs = []
        for i in (0, 1):
            raw, r = fixture(i)
            r["waves"][0]["rows"][0]["ids"][4] = 77
            bs.append(reduce_block(r, raw, [2], i))
        h = summarize(bs)
        self.assertFalse(h["qualified_for_tested_workload"])
        self.assertFalse(h["lengths"]["128"]["exact_outputs"])

    def test_exact_counter_inverse_and_refusals(self):
        for d, b in ((23, 45), (4003, 4881), (710, 710)):
            self.assertEqual(counters(snap(100, 64, d, b))["busy_slot_sum"], b)
        for k, v in [
            ("requests_processing", "1"),
            ("n_busy_slots_per_decode", "1.4922"),
            ("n_decode_total", "13.5"),
            ("n_decode_total", "1000000"),
            ("prompt_tokens_total", "nan"),
        ]:
            x = snap(100, 32, 16, 24)
            x["values"][k] = v
            with self.subTest(k=k), self.assertRaises(ValueError):
                counters(x)
        x = snap(100, 32, 200000, 300000)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            counters(x)

    def test_boundary_slot_and_cache_corruption(self):
        raw, r = fixture(0)
        for change in [
            lambda x: x["waves"][1]["before"].update(epoch="2"),
            lambda x: x["waves"][2]["rows"][0].update(reused=True),
            lambda x: x["waves"][0]["rows"][1].update(slot=0),
            lambda x: x["waves"][0]["rows"][1].update(start_ns=3),
            lambda x: x["waves"][1]["after"]["values"].update(tokens_predicted_total="777"),
            lambda x: x["warm"][1].update(truncated=True),
            lambda x: x["waves"].reverse(),
        ]:
            y = copy.deepcopy(r)
            change(y)
            with self.assertRaises(ValueError):
                reduce_block(y, raw, [2], 0)

    def test_real_log_requires_two_partitions_not_double_memory(self):
        lines = [
            "NVIDIA GB10",
            "offloaded 94/94 layers to GPU",
            "CUDA0 model buffer size = 90512.32 MiB",
        ] + [
            f"llama_context: {k} = {v}"
            for k, v in [
                ("n_seq_max", 2),
                ("n_ctx", 4096),
                ("n_ctx_seq", 2048),
                ("n_batch", 512),
                ("n_ubatch", 128),
            ]
        ]
        log = "\n".join(lines)
        self.assertEqual(read_log(log, 94, 2, 128)["context"]["n_ctx_seq"], 2048)
        for old, new in [
            ("n_ctx_seq = 2048", "n_ctx_seq = 4096"),
            ("n_ctx = 4096", "n_ctx = 8192"),
            ("n_seq_max = 2", "n_seq_max = 1"),
        ]:
            with self.assertRaises(ValueError):
                read_log(log.replace(old, new), 94, 2, 128)
