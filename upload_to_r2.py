"""
upload_to_r2.py
---------------
Upload all MLB Optimizer data assets to Cloudflare R2 object storage.
Preserves folder structure inside the bucket.

Required environment variables:
    R2_ACCOUNT_ID   — Cloudflare account ID (found in R2 dashboard)
    R2_ACCESS_KEY   — R2 API token Access Key ID
    R2_SECRET_KEY   — R2 API token Secret Access Key
    R2_BUCKET_NAME  — Name of your R2 bucket

Usage:
    python upload_to_r2.py

Optional: set R2_BASE_URL in Render env vars after uploading so the app
knows where to fetch files from (e.g. https://pub-xxx.r2.dev).
"""

import os
import sys
import mimetypes
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Config — all values come from environment variables
# ---------------------------------------------------------------------------
ACCOUNT_ID  = os.environ.get("R2_ACCOUNT_ID", "")
ACCESS_KEY  = os.environ.get("R2_ACCESS_KEY", "")
SECRET_KEY  = os.environ.get("R2_SECRET_KEY", "")
BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "")

_REQUIRED = {"R2_ACCOUNT_ID": ACCOUNT_ID, "R2_ACCESS_KEY": ACCESS_KEY,
             "R2_SECRET_KEY": SECRET_KEY,  "R2_BUCKET_NAME": BUCKET_NAME}

for _var, _val in _REQUIRED.items():
    if not _val:
        print(f"ERROR: environment variable {_var} is not set.")
        sys.exit(1)

R2_ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

# ---------------------------------------------------------------------------
# Files / directories to upload
# ---------------------------------------------------------------------------
# Project root is the parent of this script's directory
_HERE  = Path(__file__).parent.resolve()

# Each entry: (local_path, bucket_key_prefix)
# bucket_key_prefix="" means the file sits at the bucket root.
# For directories, every file inside is uploaded preserving sub-structure.
UPLOAD_TARGETS: list[tuple[Path, str]] = [
    # ── Core data files (root of data/) ─────────────────────────────────────
    (_HERE / "data" / "mlb_combined_2021_2025.csv",  "data/mlb_combined_2021_2025.csv"),
    (_HERE / "data" / "2025mlbshared.csv",            "data/2025mlbshared.csv"),
    (_HERE / "data" / "razzball.csv",                 "data/razzball.csv"),

    # ── Pre-generated analysis outputs (sit at project root) ─────────────────
    (_HERE / "efficiency_detail.csv",                 "efficiency_detail.csv"),
    (_HERE / "al_nl_ranking_table.csv",               "al_nl_ranking_table.csv"),
    (_HERE / "efficiency_scatter.png",                "efficiency_scatter.png"),
    (_HERE / "efficiency_ranking.png",                "efficiency_ranking.png"),
    (_HERE / "position_breakdown.png",                "position_breakdown.png"),

    # ── Directories (all files inside, preserving sub-paths) ─────────────────
    # These are handled specially below — see _collect_dir_files()
]

# Directories to upload recursively
UPLOAD_DIRS: list[tuple[Path, str]] = [
    (_HERE / "2026 Payroll",           "2026 Payroll"),
    (_HERE / "data" / "headshots",     "data/headshots"),
    (_HERE / "data" / "2026 Depth Chart", "data/2026 Depth Chart"),
]


def _collect_dir_files(local_dir: Path, bucket_prefix: str) -> list[tuple[Path, str]]:
    """Recursively collect (local_path, bucket_key) pairs for a directory."""
    results = []
    if not local_dir.exists():
        print(f"  SKIP (not found): {local_dir}")
        return results
    for fpath in sorted(local_dir.rglob("*")):
        if fpath.is_file():
            relative = fpath.relative_to(local_dir)
            bucket_key = f"{bucket_prefix}/{relative.as_posix()}"
            results.append((fpath, bucket_key))
    return results


def _guess_content_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def upload_file(s3, local_path: Path, bucket_key: str) -> bool:
    """Upload a single file to R2. Returns True on success."""
    if not local_path.exists():
        print(f"  SKIP (not found): {local_path}")
        return False
    content_type = _guess_content_type(local_path)
    size_kb = local_path.stat().st_size / 1024
    try:
        s3.upload_file(
            str(local_path),
            BUCKET_NAME,
            bucket_key,
            ExtraArgs={"ContentType": content_type},
        )
        print(f"  ✓  {bucket_key}  ({size_kb:.0f} KB)")
        return True
    except ClientError as e:
        print(f"  ✗  {bucket_key}  ERROR: {e}")
        return False


def main():
    print(f"Connecting to R2 bucket '{BUCKET_NAME}' ...")
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="auto",
    )

    # Verify bucket is reachable
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' found. Starting upload.\n")
    except ClientError as e:
        print(f"ERROR: Cannot access bucket '{BUCKET_NAME}': {e}")
        sys.exit(1)

    ok = err = skip = 0

    # ── Individual files ─────────────────────────────────────────────────────
    print("=== Individual files ===")
    for local_path, bucket_key in UPLOAD_TARGETS:
        if not local_path.exists():
            print(f"  SKIP (not found): {local_path.name}")
            skip += 1
            continue
        if upload_file(s3, local_path, bucket_key):
            ok += 1
        else:
            err += 1

    # ── Directories ──────────────────────────────────────────────────────────
    for local_dir, prefix in UPLOAD_DIRS:
        print(f"\n=== Directory: {local_dir.name} → {prefix}/ ===")
        pairs = _collect_dir_files(local_dir, prefix)
        if not pairs:
            print(f"  (empty or not found)")
            continue
        for local_path, bucket_key in pairs:
            if upload_file(s3, local_path, bucket_key):
                ok += 1
            else:
                err += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Upload complete: {ok} succeeded, {err} failed, {skip} skipped")
    if err > 0:
        print("WARNING: Some files failed to upload. Check errors above.")
        sys.exit(1)
    print(f"\nNext steps:")
    print(f"  1. Enable public access on the bucket (or set a custom domain).")
    print(f"  2. Set R2_BASE_URL in Render environment variables.")
    print(f"     Example: https://pub-<hash>.r2.dev")


if __name__ == "__main__":
    main()
