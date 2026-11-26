import copy, json, unittest
from pathlib import Path
from glm_manifest import tasks, label
from audit_glm_manifest import compare


class Matched(unittest.TestCase):
    def reference(self):
        return json.loads(Path("reports/qwen235b/revision/summary.json").read_bytes())["serving"]

    def test_independent_verdicts_not_token_equivalence(self):
        q = self.reference()
        g = copy.deepcopy(q)
        k = label(*next(iter(tasks())))
        g["cases"][k]["arms"]["reuse"].update(verdict="other", natural_eos=False)
        r = compare(q, g)
        self.assertEqual(r["reuse"]["qwen_row_glm_column"]["correct"]["other"], 1)
        self.assertEqual(r["cold"]["qwen_row_glm_column"]["correct"]["correct"], 8)

    def test_reject_different_questions_or_targets(self):
        q = self.reference()
        k = label(*next(iter(tasks())))
        for change in ("expected", "stale", "missing"):
            g = copy.deepcopy(q)
            if change == "missing":
                del g["cases"][k]
            else:
                g["cases"][k][change] = "UNRELATED"
            with self.assertRaises(ValueError):
                compare(q, g)
