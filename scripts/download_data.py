"""Download PTB-XL from PhysioNet (100Hz only, skipping 500Hz to save ~2GB)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
DATA_DIR = Path("data/raw/ptbxl")


def check_wget() -> bool:
    return shutil.which("wget") is not None


def download_ptbxl() -> None:
    if not check_wget():
        print("ERROR: wget is not installed.")
        print("Install it with: brew install wget")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "wget",
        "-r",                                       # recursive
        "-N",                                       # only fetch newer files
        "-c",                                       # resume partial downloads
        "-np",                                      # no parent directories
        "-nH",                                      # no host directories
        "--cut-dirs=3",                             # strip /files/ptb-xl/1.0.3
        "-X", "/files/ptb-xl/1.0.3/records500",    # skip 500Hz files
        "-P", str(DATA_DIR),
        PTBXL_URL,
    ]
    print(f"Downloading PTB-XL (100Hz) to {DATA_DIR}/")
    print("Expect ~1GB, 10-30 minutes depending on your connection.\n")
    subprocess.run(cmd, check=True)
    print(f"\nDone. Data is in {DATA_DIR}/")


if __name__ == "__main__":
    download_ptbxl()