"""A pinned download is still required to have a bounded, regular-file inventory."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location(
    "replay_prefix32", Path(__file__).with_name("replay-prefix32.py")
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ArchiveChecks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.archive = self.root / "release.zip"
        self.dest = self.root / "unpacked"
        self.dest.mkdir()
        self.meta = self.root / "evidence-archive.json"
        p = patch.object(replay, "CUT", self.root)
        p.start()
        self.addCleanup(p.stop)

    def build(self, name="cuts/logits/fixture", mode=0o100644):
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = mode << 16
        with zipfile.ZipFile(self.archive, "w") as f:
            f.writestr(info, b"abc")
        b = self.archive.read_bytes()
        self.meta.write_text(
            json.dumps(
                dict(
                    bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), members=1, expanded_bytes=3
                )
            )
        )

    def test_pinned_regular_file(self):
        self.build()
        replay.unpack(self.archive, self.dest, "cuts")
        self.assertEqual((self.dest / "cuts/logits/fixture").read_bytes(), b"abc")

    def test_tampered_download(self):
        self.build()
        self.archive.write_bytes(self.archive.read_bytes() + b"!")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replay.unpack(self.archive, self.dest, "cuts")
        self.assertFalse(list(self.dest.iterdir()))

    def test_rejects_escape_and_symlink_even_if_pinned(self):
        for name, mode in (
            ("cuts/../oops", 0o100644),
            ("/cuts/a", 0o100644),
            ("cuts/link", 0o120777),
            ("cuts/a:b", 0o100644),
        ):
            with self.subTest(name=name):
                self.build(name, mode)
                with self.assertRaisesRegex(ValueError, "non-regular or non-relative"):
                    replay.unpack(self.archive, self.dest, "cuts")
                self.assertFalse(list(self.dest.iterdir()))

    def test_rejects_wrong_inventory(self):
        self.build()
        d = json.loads(self.meta.read_text())
        d["members"] = 2
        self.meta.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "extent"):
            replay.unpack(self.archive, self.dest, "cuts")
