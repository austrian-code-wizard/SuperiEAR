import json
import subprocess
from tqdm import tqdm

from superiear.utils import ensure_dir_exists


@ensure_dir_exists
def download_audio(urls, output_dir, batch_size=35):
    """
    Downloads all audio in files in batches into specified directory.

    Expects 'urls' input to be a list of dictionaries with keys 'name'
    and 'url'. 
    """

    # Download audio in batches
    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    print(f"Downloading {len(batches)} batches of size {batch_size}")

    for batch in tqdm(batches):
        processes = []
        for url in batch:
            cmd = ["youtube-dl", "-x", "--audio-format", "wav",
                   "-o" f"{output_dir}/{url['name']}", url['url']]
            p = subprocess.Popen(cmd)
            processes.append(p)

        for p in tqdm(processes):
            p.wait()


def read_urls(filename):
    with open(filename) as f:
        data = json.load(f)
    return [{
        "name": name,
        "url": url
    } for name, url in data.items()]
