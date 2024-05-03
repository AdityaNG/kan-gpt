# KAN-GPT

[![codecov](https://codecov.io/gh/AdityaNG/kan-gpt/branch/main/graph/badge.svg?token=kan-gpt_token_here)](https://codecov.io/gh/AdityaNG/kan-gpt)
[![CI](https://github.com/AdityaNG/kan-gpt/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/kan-gpt/actions/workflows/main.yml)

Awesome KAN-GPT created by AdityaNG

## Install it from PyPI

```bash
pip install kan_gpt
```

## Usage

```py
from kan_gpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = "gpt2"
model_config.vocab_size = 5
model_config.block_size = 10
model = GPT(model_config)

x = torch.zeros((1, 10), dtype=torch.long)
y = torch.zeros((1, 10), dtype=torch.long)

# x = x.cuda()
# y = y.cuda()
# model = model.cuda()

logits, loss = model(x, y)

print(logits.shape)
```

```bash
$ python -m kan_gpt.train
```

## Setup

```bash
# Download Repo
%cd /content
!git clone https://github.com/AdityaNG/kan-gpt
%cd kan-gpt
!git pull

# Download Dataset
!./scripts/download_webtext.sh

# Install dependencies for development
!pip install -r requirements.txt
!pip install -e .
```

## Train

Dummy script to make sure everything is working as expected
```bash
CUDA_VISIBLE_DEVICE="0" python3 -m kan_gpt.train --architecture MLP --batch_size 1 --dummy_dataset
```

## TODOs

- [x] Integrate [minGPT](https://github.com/karpathy/minGPT) and [pykan](https://github.com/KindXiaoming/pykan)
- [x] Dataset downloading script for [WebText](https://github.com/openai/gpt-2-output-dataset)
- [x] PyTorch Dataset parser for [WebText](https://github.com/openai/gpt-2-output-dataset)
- [ ] Mini training POC for KAN-GPT
  - [ ] Integrate KAN training logic from `KAN.train_kan`
- [x] Mini training POC for MLP-GPT
- [x] Train MLP-GPT on the webtext dataset as a baseline
- [ ] Auto Save checkpoints
- [ ] Auto Save checkpoints to W&B
- [ ] Script to load checkpoint in interactive mode
- [ ] Training script to PyTorch Lighting

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## References

- [minGPT](https://github.com/karpathy/minGPT)
- [pykan](https://github.com/KindXiaoming/pykan)
- [WebText](https://github.com/openai/gpt-2-output-dataset)
