"""Download the labeled recipe file from the approved Kaggle dataset."""

import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/kaggle/recipe-ingredients-dataset"
DESTINATION = Path("data/raw/train.json")


def main():
    if DESTINATION.exists():
        print(f"Already downloaded: {DESTINATION}")
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "cosc325project"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()
    if not zipfile.is_zipfile(io.BytesIO(archive)):
        raise ValueError("Kaggle did not return a ZIP. Download the dataset manually; see README.")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name == "train.json"]
        if len(matches) != 1:
            raise ValueError(f"Expected one train.json; archive contains {zipped.namelist()}")
        # Read only the intended file instead of extracting arbitrary archive paths.
        content = zipped.read(matches[0])
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(content)
    print(f"Downloaded {DESTINATION} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
