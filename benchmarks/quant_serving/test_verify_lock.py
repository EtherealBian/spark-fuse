import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from verify_lock import verify


class Verify(unittest.TestCase):
    def test_hashes_real_bytes_with_mutation_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "lock.json"
            f = root / "sample.gguf"
            f.write_bytes(b"abcd")
            m = dict(
                model="synthetic",
                format="test",
                revision="a" * 40,
                files=[dict(name=f.name, bytes=4, sha256=hashlib.sha256(b"abcd").hexdigest())],
            )
            lock.write_text(json.dumps(m))
            a = verify(lock, root)
            self.assertEqual(a, verify(lock, f))
            self.assertEqual(a["total_bytes"], 4)
            f.write_bytes(b"xbcd")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify(lock, root)
            f.write_bytes(b"ab")
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                verify(lock, root)
            m["files"][0]["name"] = "../sample.gguf"
            lock.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "invalid entry"):
                verify(lock, root)
