import os
from typing import List

import requests

CHUNK_SIZE = 8192


def download_webtext(
    download_path: str = "datasets/webtext",
    splits: List[str] = ["train", "test", "valid"],
    base_url: str = "https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1",  # noqa
):
    os.makedirs(download_path, exist_ok=True)
    for split in splits:
        response = requests.get(
            f"{base_url}/webtext.{split}.jsonl", stream=True
        )
        response.raise_for_status()  # Raise HTTP errors

        # Open a local file for writing in binary mode
        with open(f"{download_path}/webtext.{split}.jsonl", "wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                file.write(chunk)


def download_tinyshakespeare(
    download_path: str = "datasets/tinyshakespeare",
    base_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare",  # noqa
):
    os.makedirs(download_path, exist_ok=True)

    response = requests.get(f"{base_url}/input.txt", stream=True)
    response.raise_for_status()  # Raise HTTP errors

    # Open a local file for writing in binary mode
    with open(f"{download_path}/input.txt", "wb") as file:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            file.write(chunk)


def main(args):
    if args.dataset == "webtext":
        download_webtext()
    elif args.dataset == "tinyshakespeare":
        download_tinyshakespeare()

    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("KAN-GPT Trainer")
    parser.add_argument(
        "--dataset",
        choices=["webtext", "tinyshakespeare"],
        default="tinyshakespeare",
    )

    args = parser.parse_args()

    main(args)
