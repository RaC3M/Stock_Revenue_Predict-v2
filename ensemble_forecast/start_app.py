from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SYSTEM_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SYSTEM_DIR / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(SYSTEM_DIR / "app.py"),
        "--server.port",
        "8501",
    ]
    subprocess.run(command, cwd=SYSTEM_DIR, check=True)


if __name__ == "__main__":
    main()
