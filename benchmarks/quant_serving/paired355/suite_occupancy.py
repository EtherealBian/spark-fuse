import copy
import itertools
import unittest
from occupancy_ledger import joint_bounds, native, timeline, divergence


def boundary(p, t, d, b):
    return dict(
        epoch="101",
        values=dict(
            prompt_tokens_total=str(p),
            tokens_predicted_total=str(t),
            n_decode_total=str(d),
            n_busy_slots_per_decode=format(b / d, ".6g"),
            requests_processing="0",
            requests_deferred="0",
        ),
    )


def fixture(mode="together"):
    rows = []
    base = {}
    serial = mode == "serial"
    for j, (a, b) in enumerate(((10, 80), (80 if serial else 12, 130))):
        case = f"128:{j}"
        base[case] = dict(ids=[j] * 16, input_sha256=str(j) * 64)
        rows.append(
            dict(
                case=case,
                slot=j,
                start_ns=a,
                end_ns=b,
                processed=128,
                predicted=16,
                reused=0,
                stop="limit",
                truncated=False,
                **base[case],
            )
        )
    return dict(
        rep=0,
        length=128,
        lead=0,
        mode=mode,
        rows=rows,
        before=boundary(256, 6, 6, 6),
        after=boundary(512, 38, 26, 26 if serial else 38),
    ), base


class Timelines(unittest.TestCase):
    def test_bounds_against_exhaustive_occupancy_histories(self):
        for d in range(1, 8):
            outcomes = {}
            for trace in itertools.product(range(3), repeat=d):
                outcomes.setdefault(sum(trace), []).append(trace.count(2))
            self.assertEqual(
                {b: joint_bounds(d, b) for b in outcomes},
                {b: (min(v), max(v)) for b, v in outcomes.items()},
            )

    def test_exact_float_text_inversion_and_invalid_value(self):
        for d, b in ((17, 22), (208, 317), (1200, 1491), (4096, 4311)):
            self.assertEqual(native(boundary(512, 38, d, b)), (512, 38, d, b))
        x = boundary(512, 38, 17, 22)
        x["values"]["n_busy_slots_per_decode"] = "1.29413"
        with self.assertRaisesRegex(ValueError, "unique"):
            native(x)
        for k, v in [
            ("n_decode_total", "100000"),
            ("requests_processing", "1"),
            ("prompt_tokens_total", "512.0"),
        ]:
            x = boundary(512, 38, 17, 22)
            x["values"][k] = v
            with self.assertRaises(ValueError):
                native(x)

    def test_client_overlap_without_proven_dual_slot(self):
        w, base = fixture()
        w["after"] = boundary(512, 38, 26, 26)
        x = timeline(w, base)
        self.assertEqual((x["client_overlap_ns"], x["dual_slot_calls_lower_bound"]), (68, 0))
        w["after"] = boundary(512, 38, 26, 38)
        x = timeline(w, base)
        self.assertEqual(
            (x["dual_slot_calls_lower_bound"], x["dual_slot_calls_upper_bound"]), (12, 16)
        )
        self.assertEqual(x["clients"][1]["service_ns"], 118)

    def test_malformed_serial_or_foreign_work_rejected(self):
        original, base = fixture("serial")
        self.assertEqual(timeline(original, base)["client_overlap_ns"], 0)
        for change in (
            lambda w: w["rows"][1].update(start_ns=1),
            lambda w: w["after"].update(epoch="2"),
            lambda w: w["rows"][0].update(slot=1),
            lambda w: w["after"]["values"].update(tokens_predicted_total="39"),
            lambda w: w["rows"][0].update(input_sha256="x"),
        ):
            w = copy.deepcopy(original)
            change(w)
            with self.assertRaises(ValueError):
                timeline(w, base)

    def test_divergent_output_is_retained_not_a_missing_result(self):
        w, base = fixture()
        w["rows"][0]["ids"] = list(w["rows"][0]["ids"])
        w["rows"][0]["ids"][10] = 17
        self.assertEqual(timeline(w, base)["clients"][0]["first_different_id_vs_one_slot"], 11)
        self.assertIsNone(divergence([1] * 16, [1] * 16))
