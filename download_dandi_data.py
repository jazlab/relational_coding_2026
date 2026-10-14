"""Download data files from DANDI dataset.

Run this script to download behavior and spikesorting data from the DANDI
dataset. Only the ring-task sessions (`_RING_SESSIONS` below) are downloaded.
Flages can be used to filter the data by modality, subject, and session:
- `--modality`: Specify the modality to download. Options are 'behavior' or
  'spikesorting'. If not specified, both modalities will be downloaded.
- `--subject`: Specify the subject to filter files by. If not specified, all
  subjects will be downloaded.
- `--session`: Specify the session to filter files by. If not specified, all
  sessions will be downloaded.

Dandiset 000620 is public, so no DANDI API key is needed. If you have one, you
can provide it with the `DANDI_API_KEY` environment variable or a file named
`DANDI_API_KEY.txt` next to this script, containing only the key.

Usage examples:
```
# Download behavior and spikesorting data for all subjects and sessions
$ python3 download_dandi_data.py

# Download only behavior data for all subjects and sessions
$ python3 download_dandi_data.py --modality=behavior

# Download all spikesorting data for subject "Perle"
$ python3 download_dandi_data.py --modality=spikesorting --subject=Perle

# Download behavior and spikesorting data for session "Perle/2022-06-11"
$ python3 download_dandi_data.py --subject=Perle --session=2022-06-11
```
"""

import os
import sys
from pathlib import Path

from dandi import dandiapi

_WRITE_DIR = Path("./cache/dandi_data")
_MODALITY_TO_SUFFIX = {
    "behavior": "behavior+task",
    "spikesorting": "spikesorting",
}

# The ring-task sessions. The dandiset also holds sessions of another task, which
# nothing in this repo uses.
_RING_SESSIONS = {
    "Elgar": [
        "2022-09-06",
        "2022-09-07",
        "2022-09-09",
        "2022-09-10",
        "2022-09-11",
        "2022-09-12",
        "2022-09-13",
        "2022-09-14",
        "2022-09-15",
        "2022-09-18",
        "2022-09-19",
        "2022-09-20",
    ],
    "Perle": [
        "2022-06-07",
        "2022-06-08",
        "2022-06-09",
        "2022-06-10",
        "2022-06-11",
        "2022-06-12",
        "2022-06-13",
        "2022-06-14",
        "2022-06-15",
        "2022-06-16",
        "2022-06-19",
    ],
}
_RING_SESSION_TAGS = [
    f"sub-{subject}/sub-{subject}_ses-{session}_"
    for subject, sessions in _RING_SESSIONS.items()
    for session in sessions
]


def _load_api_key() -> str | None:
    """Optional DANDI API key, from the environment or `DANDI_API_KEY.txt`."""
    key = os.environ.get("DANDI_API_KEY")
    key_file = Path(__file__).resolve().parent / "DANDI_API_KEY.txt"
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    return key or None


def _download_dandiset(
    modality: str | None = None,
    subject: str | None = None,
    session: str | None = None,
    dandiset_id: str = "000620",
):
    """Download all files except raw ecephys files from a dandiset.

    Args:
        modality (str): None or modality to download. Must be either 'behavior'
            or 'spikesorting'. If None, downloads both modalities.
        subject (str | None): Subject to filter files by.
        session (str | None): Session to filter files by.
        dandiset_id (str): ID of the DANDI dataset to download.
    """
    api = dandiapi.DandiAPIClient(token=_load_api_key())
    dandiset = api.get_dandiset(dandiset_id, "draft")
    all_assets = [x for x in dandiset.get_assets()]

    if modality is None:
        modalities = _MODALITY_TO_SUFFIX.keys()
    else:
        modalities = [modality]

    for modality in modalities:
        write_dir = _WRITE_DIR / modality
        print(f"\nDownloading {modality} data to {write_dir}\n")

        # Filter assets by modality
        suffix = _MODALITY_TO_SUFFIX[modality]
        assets = [x for x in all_assets if x.path.endswith(f"{suffix}.nwb")]
        assets = [
            x
            for x in assets
            if any(x.path.startswith(tag) for tag in _RING_SESSION_TAGS)
        ]
        if not assets:
            print(f"No assets found for modality: {modality}")
            continue

        # Filter by subject if provided
        if subject is not None:
            assets = [
                x for x in assets if x.path.startswith(f"sub-{subject}/")
            ]
            if not assets:
                print(f"No assets found for subject: {subject}")
                continue

        # Filter by session if provided
        if session is not None:
            assets = [x for x in assets if f"_ses-{session}_" in x.path]
            if not assets:
                print(f"No assets found for session: {session}")
                continue

        # Download each asset
        for i, asset in enumerate(assets):
            print(f"    Downloading asset {i + 1}/{len(assets)}: {asset.path}")
            download_path = write_dir / asset.path
            if download_path.exists():
                print(f"    Asset already exists: {download_path}")
                continue

            # Create download directory if it does not exist
            download_path.parent.mkdir(parents=True, exist_ok=True)

            # Download asset
            asset.download(download_path)


if __name__ == "__main__":
    # Load flag arguments
    kwargs = {}
    for arg in sys.argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(
                f"Invalid argument format: {arg}. Expected format: --flag=value"
            )
        flag, value = arg[2:].split("=", 1)
        kwargs[flag] = value
    _download_dandiset(**kwargs)
