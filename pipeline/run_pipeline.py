"""
run_pipeline.py
Runs the full data pipeline in order:
  1. fetch_spotify.py  -- Spotify metadata (skips cached)
  2. fetch_lastfm.py   -- Last.fm tags (skips cached)
  3. clean_data.py     -- merge and flatten
  4. recommend.py      -- cosine similarity recommendations
  5. build_data.py     -- final JSON for the frontend
  6. fetch_covers.py   -- cover art (skips cached)

Usage:
  python pipeline/run_pipeline.py           skip already-cached albums
  python pipeline/run_pipeline.py --force   re-fetch everything from scratch
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
PIPELINE = Path(__file__).resolve().parent

# (label, script filename, whether --force applies to this step)
STEPS = [
    ("Spotify metadata",   "fetch_spotify.py",  True),
    ("Last.fm tags",       "fetch_lastfm.py",   True),
    ("Clean and merge",    "clean_data.py",      False),
    ("Recommendations",    "recommend.py",       False),
    ("Build final JSON",   "build_data.py",      False),
    ("Cover art",          "fetch_covers.py",    True),
]


def run_step(label: str, script: str, pass_force: bool, num: int, total: int) -> None:
    """Run one pipeline step as a subprocess. Exits on failure."""
    print(f"\n[{num}/{total}] {label}")
    print("-" * 40)

    cmd = [sys.executable, str(PIPELINE / script)]
    if pass_force:
        cmd.append("--force")

    t0     = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nFailed at step {num}: {label} (exit {result.returncode})")
        print(f"Re-run manually: python pipeline/{script}")
        sys.exit(result.returncode)

    print(f"\n  Finished in {elapsed:.1f}s")


def main() -> None:
    force = "--force" in sys.argv
    total = len(STEPS)

    print("Steines Second Spin -- Data Pipeline")
    print("Mode: force re-fetch" if force else "Mode: skip already-cached")

    pipeline_start = time.time()

    for i, (label, script, accepts_force) in enumerate(STEPS, 1):
        run_step(label, script, force and accepts_force, i, total)

    print(f"\nAll {total} steps complete in {time.time() - pipeline_start:.1f}s")


if __name__ == "__main__":
    main()
