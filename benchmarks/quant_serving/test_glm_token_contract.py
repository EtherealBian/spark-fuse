from copy import deepcopy
import unittest
from glm_token_contract import compare, derive, digest


class TokenContract(unittest.TestCase):
    def fixture(self):
        ids = {"128:0": [0, 1, 2], "128:1": [0, 1, 3], "1024:0": [4, 5, 6], "1024:1": [4, 5, 7]}
        contract = dict(
            vocab_size=120,
            eog_token_ids=[15, 16],
            inputs={k: dict(tokens=len(v), input_sha256=digest(v)) for k, v in ids.items()},
        )
        return contract, dict(inputs=ids), dict(eog_token_ids=[15, 16])

    def test_refusal_of_unknown_bounded_header(self):
        with self.assertRaises(ValueError):
            derive(b"GGUF" + bytes(20))

    def test_exact_oracle_and_substitutions(self):
        contract, raw, rt = self.fixture()
        compare(contract, raw, rt)
        for change in [
            lambda x: x["inputs"]["128:0"].__setitem__(1, True),
            lambda x: x["inputs"]["128:0"].__setitem__(1, 3),
            lambda x: x["inputs"].pop("1024:0"),
            lambda x: x["inputs"].__setitem__("128:0", x["inputs"]["128:1"]),
            lambda x: x["inputs"]["128:1"].append(6),
        ]:
            obj = deepcopy(raw)
            change(obj)
            with self.assertRaises(ValueError):
                compare(contract, obj, rt)
        with self.assertRaises(ValueError):
            compare(contract, raw, {"eog_token_ids": [15]})
