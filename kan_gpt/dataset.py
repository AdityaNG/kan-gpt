import os
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import GPT2Tokenizer


class WebTextDataset(Dataset):
    """
    WebText Dataset
    """

    def __init__(self, split, model_type, block_size=1024, vocab_size=50257):
        assert split in {"train", "test", "valid"}

        self.split = split
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)

        self.tokenized_dataset_path = f"datasets/webtext/webtext.{split}.pkl"

        if not os.path.isfile(self.tokenized_dataset_path):
            self.tokenized_dataset = []

            self.json_path = f"datasets/webtext/webtext.{split}.jsonl"

            assert os.path.isfile(self.json_path)

            self.data = pd.read_json(path_or_buf=self.json_path, lines=True)

            tokenized_data = []
            tokenized_lengths = []

            for _, row in tqdm(
                self.data.iterrows(), desc="Tokenizing", total=len(self.data)
            ):
                text = row["text"]

                tokenized = self.tokenizer.encode(
                    text=text, add_special_tokens=False
                )
                tokenized_length = len(tokenized)

                tokenized_data.append(tokenized)
                tokenized_lengths.append(tokenized_length)

                self.tokenized_dataset += tokenized

            with open(self.tokenized_dataset_path, "wb") as f:
                pickle.dump(self.tokenized_dataset, f)

        with open(self.tokenized_dataset_path, "rb") as f:
            self.tokenized_dataset = pickle.load(f)

    def __len__(self):
        return len(self.tokenized_dataset) - 2 * self.block_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):

        x = self.tokenized_dataset[idx : idx + self.block_size]
        y = self.tokenized_dataset[
            idx + self.block_size : idx + 2 * self.block_size
        ]

        assert len(x) == self.block_size, f"Unexpected len: {len(x)}"
        assert len(y) == self.block_size, f"Unexpected len: {len(y)}"

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        # x = x.unsqueeze(0)
        # y = y.unsqueeze(0)

        return x, y


class TinyShakespeareDataset(Dataset):
    """
    Tiny Shakespeare dataset
    """

    TRAIN = 0.8
    VALID = 0.1
    TEST = 0.1

    def __init__(
        self,
        split,
        model_type,
        block_size=1024,
        vocab_size=50257,
    ):
        assert split in {"train", "test", "valid"}

        self.split = split
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)

        self.tokenized_dataset_path = (
            f"datasets/tinyshakespeare/input.{split}.pkl"
        )

        if not os.path.isfile(self.tokenized_dataset_path):
            self.tokenized_dataset = []

            self.tsp_path = "datasets/tinyshakespeare/input.txt"

            assert os.path.isfile(self.tsp_path)

            self.data = open(self.tsp_path, "r").readlines()

            # Select data based on split
            # First self.TRAIN % is for train
            # Next self.VALID % is for validation
            # Last self.TEST % is for test
            if self.split == "train":
                self.data = self.data[
                    : int((1 - (self.VALID + self.TEST)) * len(self.data))
                ]
            elif self.split == "val":
                self.data = self.data[
                    int((1 - (self.VALID + self.TEST)) * len(self.data)) : int(
                        ((1 - (self.VALID + self.TEST))) * len(self.data)
                    )
                    + int((1 - (self.TEST)) * len(self.data))
                ]
            elif self.split == "test":
                self.data = self.data[
                    int(((1 - (self.VALID + self.TEST))) * len(self.data)) :
                ]

            tokenized_data = []
            tokenized_lengths = []

            for text in tqdm(
                self.data, desc="Tokenizing", total=len(self.data)
            ):
                tokenized = self.tokenizer.encode(
                    text=text, add_special_tokens=False
                )
                tokenized_length = len(tokenized)

                tokenized_data.append(tokenized)
                tokenized_lengths.append(tokenized_length)

                self.tokenized_dataset += tokenized

            with open(self.tokenized_dataset_path, "wb") as f:
                pickle.dump(self.tokenized_dataset, f)

        with open(self.tokenized_dataset_path, "rb") as f:
            self.tokenized_dataset = pickle.load(f)

    def __len__(self):
        return len(self.tokenized_dataset) - 2 * self.block_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):

        x = self.tokenized_dataset[idx : idx + self.block_size]
        y = self.tokenized_dataset[
            idx + self.block_size : idx + 2 * self.block_size
        ]

        assert len(x) == self.block_size, f"Unexpected len: {len(x)}"
        assert len(y) == self.block_size, f"Unexpected len: {len(y)}"

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        # x = x.unsqueeze(0)
        # y = y.unsqueeze(0)

        return x, y


class MNISTDataset(Dataset):
    """
    MNIST Dataset for Transformer (GPT-style) processing
    """

    def __init__(self, split, model_type, block_size=784):  # 784 + 1 for label
        assert split in {"train", "test"}

        self.split = split
        self.block_size = block_size
        self.model_type = model_type

        # Load MNIST dataset
        dataset = datasets.MNIST(
            root="./data",
            train=(split == "train"),
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        self.data = []

        for img, label in dataset:
            # Flatten the image
            flattened_img = img.view(-1)
            # Convert to integer values (0-255)
            flattened_img = (flattened_img * 255).long()

            # Append label to the end of flattened image
            sample = torch.cat([flattened_img, torch.tensor([label])])

            # Pad with zeros to reach block_size
            if len(sample) < self.block_size:
                padding = torch.zeros(
                    self.block_size - len(sample), dtype=torch.long
                )
                sample = torch.cat([sample, padding])

            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        return 256  # 0-255 pixel values + 10 classes + 1 padding token

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        x = self.data[idx][:-1]  # Input: all but last token
        y = self.data[idx][1:]  # Target: all but first token

        return x, y
