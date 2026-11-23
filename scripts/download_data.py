"""Download the IBM Telco Customer Churn dataset into data/raw/.

Usage:
    python scripts/download_data.py           # download if not already present
    python scripts/download_data.py --force   # download again and overwrite
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

from config import settings  # noqa: E402
from src.data.loader import DataLoadError, load_customer_csv  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("download_data")


def download(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        logger.info("Dataset already exists at %s (use --force to re-download).", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".part")
    logger.info("Downloading %s ...", url)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temp_path, "wb") as out:
            out.write(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc

    temp_path.replace(destination)
    logger.info("Saved to %s", destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)

    try:
        path = download(settings.DATASET_URL, settings.RAW_DATA_FILE, args.force)
        df = load_customer_csv(path)
    except (RuntimeError, DataLoadError) as exc:
        logger.error("%s", exc)
        print(
            "\nManual alternative:\n"
            f"  1. Download the CSV from {settings.DATASET_KAGGLE_URL}\n"
            f"  2. Save it as {settings.RAW_DATA_FILE}\n"
        )
        return 1

    if df.shape != (settings.EXPECTED_ROWS, settings.EXPECTED_COLUMNS):
        logger.warning(
            "File has %s rows x %s columns; the published dataset has %s x %s. "
            "Check you downloaded the right file.",
            *df.shape, settings.EXPECTED_ROWS, settings.EXPECTED_COLUMNS,
        )
    else:
        logger.info("Size check passed: %s rows x %s columns.", *df.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
