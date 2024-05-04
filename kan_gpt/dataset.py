import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer


class WebTextDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, model_type, block_size=1024, vocab_size=50257):
        assert split in {"train", "test", "valid"}

        self.split = split
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)

        self.compiled_json_path = (
            f"datasets/webtext/webtext.{split}.compiled.jsonl"
        )

        encoding = "utf8"

        if not os.path.isfile(self.compiled_json_path):
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

            self.data["tokenized"] = tokenized_data
            self.data["tokenized_length"] = tokenized_lengths

            for _, row in tqdm(
                self.data.iterrows(),
                desc="Caching chunks",
                total=len(self.data),
            ):
                tokenized = row["tokenized"]
                tokenized_length = row["tokenized_length"]

                for index in range(1, tokenized_length - 1, 100):
                    start_index = 0
                    mid_index = start_index + index
                    last_index = tokenized_length

                    x = np.array(
                        [
                            self.vocab_size - 1,
                        ]
                        * self.block_size
                    )
                    y = np.array(
                        [
                            self.vocab_size - 1,
                        ]
                        * self.block_size
                    )

                    assert len(x) == self.block_size
                    assert len(y) == self.block_size

                    x[-(mid_index - start_index) :] = tokenized[
                        start_index:mid_index
                    ]
                    y[: (last_index - mid_index)] = tokenized[
                        mid_index:last_index
                    ]

                    assert len(x) == self.block_size
                    assert len(y) == self.block_size

                    x = x.tolist()
                    y = y.tolist()

                    data = {"x": x, "y": y}
                    data_json = json.dumps(data) + "\n"

                    data_loaded = json.loads(data_json)
                    assert (
                        len(data_loaded["x"]) == self.block_size
                    ), f"Unexpected len: {len(data_loaded['x'])}"
                    assert (
                        len(data_loaded["y"]) == self.block_size
                    ), f"Unexpected len: {len(data_loaded['y'])}"
                    with open(
                        self.compiled_json_path, "a", encoding=encoding
                    ) as cache_file:
                        cache_file.write(data_json)

        # Read from cache
        self.dataset = pd.read_json(
            path_or_buf=self.compiled_json_path, encoding=encoding, lines=True
        )

    def __len__(self):
        return len(self.dataset)

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):

        x = self.dataset["x"][idx]
        y = self.dataset["y"][idx]

        assert len(x) == self.block_size, f"Unexpected len: {len(x)}"
        assert len(y) == self.block_size, f"Unexpected len: {len(y)}"

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        # x = x.unsqueeze(0)
        # y = y.unsqueeze(0)

        return x, y
