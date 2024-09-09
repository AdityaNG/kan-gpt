import os
from typing import List

import requests
from tqdm import tqdm

CHUNK_SIZE = 8192


def download_webtext(
    download_path: str = "datasets/webtext",
    splits: List[str] = ["train", "test", "valid"],
    base_url: str = "https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1",  # noqa
):
    os.makedirs(download_path, exist_ok=True)
    for split in splits:
        file_path = f"{download_path}/webtext.{split}.jsonl"
        file_url = f"{base_url}/webtext.{split}.jsonl"

        # Check if file exists and get its size
        initial_pos = 0
        if os.path.exists(file_path):
            initial_pos = os.path.getsize(file_path)
            print(
                f"Resuming download of webtext.{split}.jsonl from {initial_pos} bytes"  # noqa
            )

        # Set up headers for resuming download
        headers = {"Range": f"bytes={initial_pos}-"}

        response = requests.get(file_url, stream=True, headers=headers)

        # If the server doesn't support range requests, start over
        if response.status_code == 416:
            print(
                f"Cannot resume download for webtext.{split}.jsonl. Starting from beginning."  # noqa
            )
            initial_pos = 0
            headers = {}
            response = requests.get(file_url, stream=True)

        response.raise_for_status()  # Raise HTTP errors
        total_size = (
            int(response.headers.get("content-length", 0)) + initial_pos
        )

        # Open the local file for writing in binary mode, appending if resuming
        mode = "ab" if initial_pos > 0 else "wb"
        with open(file_path, mode) as file, tqdm(
            desc=f"Downloading webtext.{split}.jsonl",
            initial=initial_pos,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                size = file.write(chunk)
                progress_bar.update(size)

        # Verify file size after download
        if os.path.getsize(file_path) != total_size:
            print(
                f"Warning: Downloaded file size does not match expected size for webtext.{split}.jsonl"  # noqa
            )


def download_tinyshakespeare(
    download_path: str = "datasets/tinyshakespeare",
    base_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare",  # noqa
):
    os.makedirs(download_path, exist_ok=True)

    response = requests.get(f"{base_url}/input.txt", stream=True)
    response.raise_for_status()  # Raise HTTP errors
    total_size = int(response.headers.get("content-length", 0))

    # Open a local file for writing in binary mode
    with open(f"{download_path}/input.txt", "wb") as file, tqdm(
        desc=f"Downloading {download_path}/input.txt",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            size = file.write(chunk)
            progress_bar.update(size)


def download_mnist():
    from .dataset import MNISTDataset

    MNISTDataset("train", "gpt2")
    MNISTDataset("test", "gpt2")


def main(args):
    if args.dataset == "webtext":
        download_webtext()
    elif args.dataset == "tinyshakespeare":
        download_tinyshakespeare()
    elif args.dataset == "mnist":
        download_mnist()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("KAN-GPT Trainer")
    parser.add_argument(
        "--dataset",
        choices=["webtext", "tinyshakespeare", "mnist"],
        default="tinyshakespeare",
    )

    args = parser.parse_args()

    main(args)
