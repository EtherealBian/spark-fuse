import hashlib
import json
import tempfile
import warnings
import zipfile
from pathlib import Path
from unittest import TestCase, mock
import fork_witness as w


class Lossless(TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        p = mock.patch.dict(w.KINDS, {"kv": (1, 2), "logits": (1, 2)}, clear=True)
        p.start()
        self.addCleanup(p.stop)
        self.src = self.root / "source"
        self.src.mkdir()
        self.arc = self.root / "source.zip"
        self.dst = self.root / "dest"
        records = []
        for n in (128, 2048):
            for f in (1, 2):
                for b in (0, 1):
                    for t in (0, 1):
                        for a in w.ARMS:
                            r = {"cell": [n, f, b], "arm": a, "repeat": t}
                            records.append(r)
                            for kind in w.KINDS:
                                data = bytes([int(kind == "kv"), b, f, 10])
                                name = f"{n}-{f}-{b}-{t}-{a}-{kind}.bf16"
                                (self.src / name).write_bytes(data)
                                r[kind] = dict(sha256=w.digest(data), shape=[1, 2], bytes=4)
        (self.src / "index.json").write_text(
            json.dumps(dict(schema="sparkfuse.fork.attribution.v1", records=records))
        )
        for k in w.METADATA[1:]:
            (self.src / k).write_text("{}")
        self.seal = w.make(self.src, self.arc)
        self.setseal()

    def setseal(self):
        (self.src / "witness.json").write_text(json.dumps(self.seal))

    def test_all_96_observations_roundtrip_and_no_clobber(self):
        w.materialize(self.src, self.arc, self.dst)
        for p in self.src.glob("*.bf16"):
            self.assertEqual(p.read_bytes(), (self.dst / p.name).read_bytes())
        self.assertEqual((self.seal["unique_payloads"], self.seal["logical_files"]), (8, 96))
        with self.assertRaises(FileExistsError):
            w.materialize(self.src, self.arc, self.dst)

    def test_copy_fallback_on_no_hardlinks(self):
        with mock.patch("os.link", side_effect=OSError("unsupported")):
            w.materialize(self.src, self.arc, self.dst)
        self.assertEqual(len(list(self.dst.glob("*.bf16"))), 96)
        self.assertEqual(
            (self.dst / "128-1-0-1-restored-kv.bf16").read_bytes(), bytes([1, 0, 1, 10])
        )

    def test_pack_detects_modified_duplicate_and_does_not_replace_zip(self):
        with self.assertRaises(FileExistsError):
            w.make(self.src, self.arc)
        (self.src / "128-1-0-1-restored-kv.bf16").write_bytes(b"abcd")
        with self.assertRaisesRegex(ValueError, "changed source"):
            w.make(self.src, self.root / "second.zip")

    def test_archive_and_index_are_both_bound(self):
        original = self.arc.read_bytes()
        self.arc.write_bytes(original + b"after")
        with self.assertRaisesRegex(ValueError, "seal"):
            w.materialize(self.src, self.arc, self.dst)
        self.arc.write_bytes(original)
        p = self.src / "index.json"
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "seal"):
            w.materialize(self.src, self.arc, self.dst)

    def test_even_resealed_duplicate_or_traversal_rejected(self):
        original = self.arc.read_bytes()
        with zipfile.ZipFile(self.arc) as z:
            good = z.namelist()[0]
        for name in (good, "../escape"):
            with self.subTest(name=name):
                self.arc.write_bytes(original)
                with warnings.catch_warnings(), zipfile.ZipFile(self.arc, "a") as z:
                    warnings.simplefilter("ignore", UserWarning)
                    z.writestr(name, b"bad")
                self.seal.update(
                    archive_bytes=self.arc.stat().st_size,
                    archive_sha256=w.digest(self.arc.read_bytes()),
                )
                self.setseal()
                with self.assertRaisesRegex(ValueError, "membership"):
                    w.materialize(self.src, self.arc, self.dst)
                self.assertFalse(self.dst.exists())
                self.assertFalse((self.root / "escape").exists())

    def test_same_size_bad_payload_rejected_even_with_updated_zip_hash(self):
        with zipfile.ZipFile(self.arc) as z:
            rows = [(k, z.read(k)) for k in z.namelist()]
        with zipfile.ZipFile(self.arc, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for i, (k, b) in enumerate(rows):
                z.writestr(k, b if i else b"BAD!")
        self.seal.update(
            archive_bytes=self.arc.stat().st_size,
            archive_sha256=hashlib.sha256(self.arc.read_bytes()).hexdigest(),
        )
        self.setseal()
        with self.assertRaisesRegex(ValueError, "payload digest"):
            w.materialize(self.src, self.arc, self.dst)

    def test_typed_case_extent_and_file_membership(self):
        idx = json.loads((self.src / "index.json").read_text())
        for key, val in [("arm", "../unsafe"), ("repeat", False), ("cell", [128, True, 0])]:
            import copy

            bad = copy.deepcopy(idx)
            bad["records"][0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.layout(bad)
        idx["records"][0]["logits"]["bytes"] = 8
        with self.assertRaisesRegex(ValueError, "extent"):
            w.layout(idx)
        (self.src / "extra.bf16").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "membership"):
            w.make(self.src, self.root / "second.zip")

    def test_repacking_order_metadata_reproducible_on_same_encoder(self):
        fresh = self.root / "second.zip"
        self.assertEqual(w.make(self.src, fresh), self.seal)
        self.assertEqual(fresh.read_bytes(), self.arc.read_bytes())
