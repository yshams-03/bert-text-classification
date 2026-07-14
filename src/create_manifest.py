"""Create SHA-256 metadata for local model and processed-data artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIRS = (ROOT / "checkpoints", ROOT / "data" / "processed")
OUTPUT = ROOT / "artifacts" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for directory in ARTIFACT_DIRS:
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    files.append(
                        {
                            "path": path.relative_to(ROOT).as_posix(),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(files)} artifact entries to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
