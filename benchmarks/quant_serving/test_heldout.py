import copy
import unittest
from heldout import ARMS, OUT, arm_order, audit, case_plan, digest


def fixture():
    r = []
    for ix, (n, domain) in enumerate(case_plan()):
        ids = [[ix + 9] * (n - 32) + [40 + j] * 32 for j in range(4)]
        case = dict(
            length=n, domain=domain, prompts=ids, prompt_sha256=[digest(q) for q in ids], runs=[]
        )
        r.append(case)

        def measure(n, hit, g):
            return dict(
                processed=n - hit,
                reused=hit,
                predicted=g,
                ids=[3] * g,
                stop="limit",
                prompt_ms=2,
                decode_ms=0 if g == 1 else 2,
                wall_ms=5,
            )

        for t in range(3):
            for a in arm_order(ix, t):
                pr = measure(n - 32, 0, 1) if a == "prepared" else None
                rr = [
                    measure(
                        n, n - 32 if a == "prepared" or (a == "on-demand" and j > 0) else 0, OUT
                    )
                    for j in range(4)
                ]
                case["runs"].append(
                    dict(
                        arm=a,
                        repeat=t,
                        primer=pr,
                        requests=rr,
                        cumulative_wall_ms=[5 * (j + 1 + int(pr is not None)) for j in range(4)],
                    )
                )
    return r


class Bursts(unittest.TestCase):
    def test_preparation_is_charged_and_heldout_ids_disjoint(self):
        r = fixture()
        x = audit(r, [], {151645})
        res = x["cases"][0]["arms"]
        self.assertEqual(set(res), set(ARMS))
        self.assertEqual(res["prepared"]["median_paired_wall_saved_vs_on_demand_ms"], [-5] * 4)
        self.assertEqual(res["on-demand"]["exact_vs_same_repeat_cold"], 12)
        with self.assertRaisesRegex(ValueError, "reused prompt"):
            audit(r, [dict(target=r[0]["prompt_sha256"][0])], {151645})
        for edit, exc in [
            (lambda p: p["requests"][0].update(reused=0), "counter"),
            (lambda p: p["cumulative_wall_ms"].__setitem__(0, 5), "clock"),
            (lambda p: p["primer"]["ids"].__setitem__(0, 151645), "token"),
        ]:
            z = copy.deepcopy(r)
            prepared = next(v for v in z[0]["runs"] if v["arm"] == "prepared")
            edit(prepared)
            with self.assertRaisesRegex(ValueError, exc):
                audit(z, [], {151645})

    def test_disagreement_is_counted_not_rejected(self):
        r = fixture()
        arm = next(v for v in r[0]["runs"] if v["arm"] == "on-demand")
        arm["requests"][2]["ids"][4] = 17
        self.assertEqual(
            audit(r, [], set())["cases"][0]["arms"]["on-demand"]["exact_vs_same_repeat_cold"], 11
        )
