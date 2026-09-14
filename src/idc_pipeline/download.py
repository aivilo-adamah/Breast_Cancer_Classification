from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
import os
import zipfile

from .config import CANONICAL_DATASET_DIRNAME


def infer_archive_name(url: str, default_name: str = "dataset.zip") -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if "p" in query and query["p"]:
        candidate = Path(unquote(query["p"][0])).name
        if candidate:
            return candidate

    candidate = Path(unquote(parsed.path)).name
    return candidate or default_name


def download_file(
    url: str,
    destination: str | Path,
    *,
    force: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists() and not force:
        print(f"Using existing archive: {destination_path}")
        return destination_path

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, destination_path.open("wb") as handle:
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) if content_length is not None else None
        downloaded = 0
        next_report_threshold_mb = 100.0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break

            handle.write(chunk)
            downloaded += len(chunk)

            downloaded_mb = downloaded / (1024 * 1024)
            should_report = downloaded_mb >= next_report_threshold_mb

            if total_size and downloaded == total_size:
                should_report = True

            if should_report:
                if total_size:
                    total_size_mb = total_size / (1024 * 1024)
                    percent = downloaded / total_size * 100
                    print(
                        f"Downloaded {downloaded_mb:.1f} MB "
                        f"of {total_size_mb:.1f} MB ({percent:.1f}%)"
                    )
                else:
                    print(f"Downloaded {downloaded_mb:.1f} MB")

                next_report_threshold_mb = downloaded_mb + 100.0

    return destination_path


def _ensure_safe_member_path(extract_dir: Path, member_name: str) -> None:
    resolved_root = extract_dir.resolve()
    resolved_target = (extract_dir / member_name).resolve()

    # Prevent path traversal when extracting third-party zip archives.
    if resolved_target != resolved_root and os.path.commonpath(
        [str(resolved_root), str(resolved_target)]
    ) != str(resolved_root):
        raise ValueError(f"Unsafe archive member path: {member_name}")


def extract_zip(
    archive_path: str | Path,
    extract_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    archive = Path(archive_path).expanduser()
    destination = Path(extract_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    if not force:
        existing_match = list(destination.rglob(CANONICAL_DATASET_DIRNAME))
        if existing_match:
            print(f"Using existing extracted dataset: {existing_match[0]}")
            return destination

    with zipfile.ZipFile(archive, "r") as zip_handle:
        for member in zip_handle.infolist():
            _ensure_safe_member_path(destination, member.filename)

        zip_handle.extractall(destination)

    return destination


def find_canonical_dataset_dir(search_root: str | Path) -> Path:
    root = Path(search_root).expanduser()

    direct = root / CANONICAL_DATASET_DIRNAME
    if direct.exists():
        return direct

    matches = sorted(root.rglob(CANONICAL_DATASET_DIRNAME))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {CANONICAL_DATASET_DIRNAME} under {root} after extraction."
        )

    return matches[0]


def prepare_dataset_from_zip_url(
    url: str,
    working_dir: str | Path,
    *,
    archive_name: str | None = None,
    extract_dir_name: str = "dataset",
    force_download: bool = False,
    force_extract: bool = False,
) -> Path:
    workspace = Path(working_dir).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)

    final_archive_name = archive_name or infer_archive_name(url)
    archive_path = workspace / final_archive_name
    extract_dir = workspace / extract_dir_name

    download_file(url, archive_path, force=force_download)
    extract_zip(archive_path, extract_dir, force=force_extract)

    dataset_dir = find_canonical_dataset_dir(extract_dir)
    print(f"Resolved canonical dataset directory: {dataset_dir}")
    return dataset_dir


def prepare_dataset_from_cached_zip(
    archive_path: str | Path,
    extract_dir: str | Path,
    *,
    force_extract: bool = False,
) -> Path:
    archive = Path(archive_path).expanduser()
    if not archive.exists():
        raise FileNotFoundError(f"Cached dataset archive not found: {archive}")

    extract_zip(archive, extract_dir, force=force_extract)
    dataset_dir = find_canonical_dataset_dir(extract_dir)
    print(f"Resolved canonical dataset directory: {dataset_dir}")
    return dataset_dir


def prepare_dataset_from_zip_url_with_cache(
    url: str,
    archive_path: str | Path,
    extract_dir: str | Path,
    *,
    force_download: bool = False,
    force_extract: bool = False,
) -> Path:
    archive = Path(archive_path).expanduser()
    archive.parent.mkdir(parents=True, exist_ok=True)

    download_file(url, archive, force=force_download)
    return prepare_dataset_from_cached_zip(
        archive_path=archive,
        extract_dir=extract_dir,
        force_extract=force_extract,
    )
