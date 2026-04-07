"""
Launcher for Celonis extraction. Run from the project root:

    python main_ingest.py
    python main_ingest.py --data-model
    python main_ingest.py --all

The implementation is in src/ingestion/main_ingest.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    script = root / "src" / "ingestion" / "main_ingest.py"
    if not script.is_file():
        print(f"Error: missing {script}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(script), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
