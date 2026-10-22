from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


SAMPLES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Ultralytics YOLO sample images")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/images"))
    parser.add_argument("--name", action="append", choices=sorted(SAMPLES), help="download only this sample; repeatable")
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    names = options.name or list(SAMPLES)
    for name in names:
        with urlopen(SAMPLES[name], timeout=30) as response:
            payload = response.read()
        output = options.output_dir / name
        output.write_bytes(payload)
        print(f"saved {output.resolve()}")
        print(f"sha256 {hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
