"""
run_all.py
Runs the full pipeline in order.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "pipeline"

# (script_name, description)
PIPELINE_STEPS = [
    ("fetch_spotify.py",   "Fetching Spotify metadata"),
    ("fetch_lastfm.py",    "Enriching genres via Last.fm"),
    ("clean_data.py",      "Normalizing data"),
    ("embed_reviews.py",   "Generating review embeddings"),
    ("recommend.py",       "Calculating recommendations"),
    ("build_data.py",      "Merging final JSON"),
    ("fetch_covers.py",    "Downloading covers"),
    ("build_html.py",      "Generating HTML pages")
]

def run_script(script_path, description):
    """Runs a script and returns success status."""
    print(f"\n[STEP] {description}...")
    print(f"       Running {script_path.name}...")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            capture_output=False,
            text=True
        )
        
        duration = time.time() - start_time
        if result.returncode == 0:
            print(f"       Success! ({duration:.1f}s)")
            return True
        else:
            print(f"       [ERROR] {script_path.name} failed (code {result.returncode})")
            return False
            
    except Exception as e:
        print(f"       [ERROR] Could not run {script_path.name}: {e}")
        return False

def main():
    print("=" * 60)
    print("STEINES' SECOND SPIN — MASTER PIPELINE")
    print("=" * 60)
    
    overall_start = time.time()
    success_count = 0
    
    for script_name, description in PIPELINE_STEPS:
        script_path = PIPELINE_DIR / script_name
        
        if not script_path.exists():
            print(f"\n[CRITICAL ERROR] Script not found: {script_path}")
            sys.exit(1)
            
        if not run_script(script_path, description):
            print("\n" + "!" * 60)
            print(f"PIPELINE HALTED: Failure in {script_name}")
            print("!" * 60)
            sys.exit(1)
            
        success_count += 1

    total_duration = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE: {success_count}/{len(PIPELINE_STEPS)} steps successful.")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print("=" * 60)
    print("Your site is up to date and ready to serve.")

if __name__ == "__main__":
    main()
