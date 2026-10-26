from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OpenCV's public pedestrian sample")
    parser.add_argument("--output", type=Path, default=Path("contrib/bytetrack/artifacts/vtest.avi"))
    options = parser.parse_args()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SAMPLE_URL, timeout=60) as response:
        payload = response.read()
    options.output.write_bytes(payload)
    print(f"saved {options.output.resolve()}")
    print(f"sha256 {hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
