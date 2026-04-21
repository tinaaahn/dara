"""Download or locate BGMN executable for Linux, Mac, and Windows."""
from __future__ import annotations

import os
import platform
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def _get_bgmn_root() -> Path:
    return Path(__file__).parent


def _get_bgmnwin_dir() -> Path:
    return _get_bgmn_root() / "BGMNwin"


def _has_bgmn_files(bgmn_dir: Path) -> bool:
    if not bgmn_dir.exists():
        return False

    required = ["bgmn", "teil", "eflech", "output"]
    return all((bgmn_dir / name).exists() for name in required)


def _copy_local_bgmn(source_dir: Path, target_dir: Path) -> None:
    if not _has_bgmn_files(source_dir):
        raise FileNotFoundError(
            f"Local BGMN folder found at {source_dir}, but required files are missing."
        )

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)


def _download_and_extract(url: str, destination_root: Path) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    zip_path = destination_root / "bgmnwin.zip"

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress = tqdm(total=total_size, unit="iB", unit_scale=True)

    with zip_path.open("wb") as f:
        for data in response.iter_content(block_size):
            progress.update(len(data))
            f.write(data)

    progress.close()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path=destination_root)

    zip_path.unlink(missing_ok=True)


def _set_permissions(bgmn_dir: Path) -> None:
    os_name = platform.system()
    if os_name in {"Linux", "Darwin"}:
        for exe in ["bgmn", "teil", "eflech", "output"]:
            exe_path = bgmn_dir / exe
            if exe_path.exists():
                os.system(f'chmod +x "{exe_path}"')


def download_bgmn() -> Path:
    """
    Ensure BGMN exists locally and return the BGMNwin directory.

    Priority:
    1. Existing packaged/local BGMNwin under this module
    2. Environment variable DARA_BGMN_PATH
    3. Download from upstream URL
    """
    os_name = platform.system()
    if os_name not in {"Darwin", "Linux", "Windows"}:
        raise RuntimeError(f"Unsupported OS: {os_name}")

    bgmn_root = _get_bgmn_root()
    bgmn_dir = _get_bgmnwin_dir()

    # 1. Already present inside package
    if _has_bgmn_files(bgmn_dir):
        _set_permissions(bgmn_dir)
        return bgmn_dir

    # 2. User-provided local path
    env_path = os.getenv("DARA_BGMN_PATH")
    if env_path:
        source_dir = Path(env_path).expanduser().resolve()
        _copy_local_bgmn(source_dir, bgmn_dir)
        _set_permissions(bgmn_dir)
        return bgmn_dir

    # 3. Fallback download
    url = f"https://cedergrouphub.github.io/dara/_static/bgmnwin_{os_name}.zip"
    _download_and_extract(url, bgmn_root)

    if not _has_bgmn_files(bgmn_dir):
        raise RuntimeError(
            f"BGMN download completed, but expected files were not found in {bgmn_dir}"
        )

    _set_permissions(bgmn_dir)
    return bgmn_dir


if __name__ == "__main__":
    download_bgmn()