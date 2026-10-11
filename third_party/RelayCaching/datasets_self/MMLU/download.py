import os
import requests
import tarfile
import sys, pathlib
from pathlib import Path
# Get current file path
current_file_path = Path(__file__).resolve()
# Get 'RelayCaching' root directory (assuming structure: RelayCaching/RelayCaching/utils/layer_profiler.py)
project_root = current_file_path.parents[2] 
# Add project root to sys.path so we can import as `from RelayCaching.utils...`
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from RelayCaching.utils.log import logger


def download():

    this_file_path = os.path.split(__file__)[0]
    tar_path = os.path.join(this_file_path, "data.tar")
    if not os.path.exists(tar_path):
        url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        logger.info("Downloading {}", url)
        r = requests.get(url, allow_redirects=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)
        logger.info("Saved to {}", tar_path)

    data_path = os.path.join(this_file_path, "data")
    if not os.path.exists(data_path):
        tar = tarfile.open(tar_path)
        tar.extractall(this_file_path)
        tar.close()
        logger.info("Saved to {}", data_path)


if __name__ == "__main__":
    download()
