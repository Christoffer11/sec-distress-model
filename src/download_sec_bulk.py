"""Download an official SEC bulk API archive.

The default archive is ``submissions.zip``, which contains public filing
histories for all EDGAR filers. The ZIP is deliberately left compressed:
later scripts can scan JSON members directly without creating thousands of
small files.

Example
-------
PowerShell:
    python src/download_sec_bulk.py submissions
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests

from fetch_submissions import get_user_agent


ARCHIVES = {
    "submissions": (
        "https://www.sec.gov/Archives/edgar/daily-index/"
        "bulkdata/submissions.zip"
    ),
    "companyfacts": (
        "https://www.sec.gov/Archives/edgar/daily-index/"
        "xbrl/companyfacts.zip"
    ),
}

DEFAULT_OUTPUT_DIR = Path("data/raw/sec_bulk")
DEFAULT_TIMEOUT_SECONDS = 120
CHUNK_SIZE_BYTES = 1024 * 1024


def download_archive(
    url: str,
    destination: Path,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Stream an SEC ZIP archive to disk and replace atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0

            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE_BYTES
                ):
                    if not chunk:
                        continue

                    output.write(chunk)
                    downloaded_bytes += len(chunk)

                    if total_bytes:
                        percentage = 100 * downloaded_bytes / total_bytes
                        print(
                            "\r"
                            f"Downloaded "
                            f"{downloaded_bytes / 1024**2:,.1f} MB "
                            f"of {total_bytes / 1024**2:,.1f} MB "
                            f"({percentage:,.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            "\r"
                            f"Downloaded "
                            f"{downloaded_bytes / 1024**2:,.1f} MB",
                            end="",
                            flush=True,
                        )

        print()

    except requests.Timeout as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SEC download timed out after {timeout} seconds."
        ) from exc
    except requests.HTTPError as exc:
        temporary_path.unlink(missing_ok=True)
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )
        raise RuntimeError(
            f"SEC returned HTTP {status} while downloading {url}."
        ) from exc
    except requests.RequestException as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the SEC archive: {exc}"
        ) from exc
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise

    if not zipfile.is_zipfile(temporary_path):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded file is not a valid ZIP archive."
        )

    os.replace(temporary_path, destination)


def inspect_archive(path: Path) -> tuple[int, int]:
    """Return the number of files and total uncompressed bytes."""
    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
        ]

    return (
        len(members),
        sum(member.file_size for member in members),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an official SEC bulk API ZIP archive."
    )
    parser.add_argument(
        "archive",
        choices=sorted(ARCHIVES),
        help="Bulk archive to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which the ZIP is stored.",
    )
    parser.add_argument(
        "--user-agent",
        help=(
            "Declared identity for SEC requests. Alternatively set "
            "SEC_USER_AGENT."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Read timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing archive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        user_agent = get_user_agent(args.user_agent)
        destination = args.output_dir / f"{args.archive}.zip"

        if destination.exists() and not args.overwrite:
            print(f"Archive already exists: {destination}")
            print("Use --overwrite to download the current nightly copy.")
        else:
            print(f"Downloading: {ARCHIVES[args.archive]}")
            download_archive(
                url=ARCHIVES[args.archive],
                destination=destination,
                user_agent=user_agent,
                timeout=args.timeout,
            )
            print(f"Saved to: {destination}")

        file_count, uncompressed_bytes = inspect_archive(destination)

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    compressed_bytes = destination.stat().st_size

    print(f"ZIP members: {file_count:,}")
    print(f"Compressed size: {compressed_bytes / 1024**2:,.1f} MB")
    print(
        "Uncompressed size: "
        f"{uncompressed_bytes / 1024**2:,.1f} MB"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
