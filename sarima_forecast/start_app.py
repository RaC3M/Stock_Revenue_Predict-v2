from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SYSTEM_DIR = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(SYSTEM_DIR / "app.py"),
            "--server.port",
            "8503",
        ],
        cwd=SYSTEM_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()

