"""Archive the exact project sources used by fresh workers, including uncommitted changes."""

from hashlib import sha256
import io
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def capture(folder):
    paths = sorted(
        {
            p
            for pat in (
                "sparkfuse/*.py",
                "sparkfuse/csrc/*.cpp",
                "examples/*.py",
                "tools/*.py",
                "pyproject.toml",
            )
            for p in ROOT.glob(pat)
        }
    )
    files = [(p.relative_to(ROOT).as_posix(), p.read_bytes()) for p in paths]
    h = sha256()
    for name, data in files:
        h.update(name.encode() + b"\x00" + sha256(data).digest())
    digest = h.hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / (digest + ".tar.gz")
    if not archive.is_file():
        tmp = folder / (digest + f".{os.getpid()}.tmp")
        try:
            with tarfile.open(tmp, "w:gz") as out:
                for name, data in files:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    out.addfile(info, io.BytesIO(data))
            tmp.replace(archive)
        finally:
            tmp.unlink(missing_ok=True)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {
        "git_head_at_capture": rev.stdout.strip() if rev.returncode == 0 else None,
        "source_tree_sha256": digest,
        "archive": "sources/" + archive.name,
    }
