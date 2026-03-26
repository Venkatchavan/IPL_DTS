"""
setup_data.py — Data bootstrap for Streamlit Cloud deployment.

On Streamlit Community Cloud, the raw CSVs are not committed to the repo.
This script:
  1. Checks if processed parquets already exist (committed to repo → skip).
  2. If not: reads KAGGLE_USERNAME and KAGGLE_KEY from st.secrets, downloads
     deliveries.csv + matches.csv via the Kaggle API, then runs the full
     pipeline sequence to regenerate all parquets.

Usage (called automatically from app/main.py before any data load):
  from setup_data import ensure_data_ready
  ensure_data_ready()
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def _parquets_present() -> bool:
    """Return True if the minimum required processed files exist."""
    required = [
        ROOT / "data" / "features" / "ball_states.parquet",
        ROOT / "data" / "features" / "player_features.parquet",
    ]
    return all(p.exists() for p in required)


def _raw_csvs_present() -> bool:
    raw = ROOT / "data" / "raw"
    return (raw / "deliveries.csv").exists() and (raw / "matches.csv").exists()


def _download_kaggle_data():
    """
    Download IPL dataset from Kaggle using credentials from st.secrets or env vars.
    Requires KAGGLE_USERNAME and KAGGLE_KEY to be set.
    """
    try:
        import streamlit as st
        username = st.secrets.get("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
        key      = st.secrets.get("KAGGLE_KEY",      os.getenv("KAGGLE_KEY",      ""))
    except Exception:
        username = os.getenv("KAGGLE_USERNAME", "")
        key      = os.getenv("KAGGLE_KEY",      "")

    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials not found. "
            "Set KAGGLE_USERNAME and KAGGLE_KEY in .streamlit/secrets.toml "
            "or as environment variables."
        )

    # Write ephemeral kaggle.json (Kaggle CLI requires this file)
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
    kaggle_json.chmod(0o600)

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading IPL dataset from Kaggle...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "chaitu20/ipl-dataset2008-2025",
            "-p", str(raw_dir),
            "--unzip",
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")
    log.info("Download complete.")


def _run_pipeline(script_name: str, extra_args: list[str] | None = None):
    args = [sys.executable, str(ROOT / "pipelines" / script_name)]
    if extra_args:
        args.extend(extra_args)
    log.info(f"Running {script_name}...")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed:\n{result.stderr[-2000:]}")
    log.info(f"  {script_name} complete.")


def ensure_data_ready():
    """
    Called at app startup. Idempotent — exits fast if data is already present.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if _parquets_present():
        log.info("Processed parquets found — skipping data setup.")
        return

    log.info("Processed data not found — running full data pipeline...")

    if not _raw_csvs_present():
        _download_kaggle_data()

    _run_pipeline("01_ingest_validate.py")
    _run_pipeline("02_state_reconstruction.py")
    _run_pipeline("03_feature_engineering.py")
    _run_pipeline("04_metrics_compute.py", ["--phase", "A"])
    _run_pipeline("05_model_training.py")
    _run_pipeline("04_metrics_compute.py", ["--phase", "B"])

    log.info("All pipelines complete. Dashboard data is ready.")


if __name__ == "__main__":
    ensure_data_ready()
