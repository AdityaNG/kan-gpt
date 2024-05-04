# KAN-GPT

[![codecov](https://codecov.io/gh/AdityaNG/kan-gpt/branch/main/graph/badge.svg?token=kan-gpt_token_here)](https://codecov.io/gh/AdityaNG/kan-gpt)
[![CI](https://github.com/AdityaNG/kan-gpt/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/kan-gpt/actions/workflows/main.yml)

The PyTorch implementation of Generative Pre-trained Transformers (GPTs) using Kolmogorov-Arnold Networks (KANs) for language modeling

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

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

x = torch.zeros((1, 10), dtype=torch.long)
y = torch.zeros((1, 10), dtype=torch.long)

prompt = "Bangalore is often described as the "

prompt_encoded = tokenizer.encode(
  text=prompt, add_special_tokens=False
)

result = prompt
x = torch.tensor(prompt_encoded).unsqueeze(0)

for _ in range(50):  # sample 50 tokens
  logits, loss = model(x)
  x = torch.cat(
    (x[:, 1:-2], logits[:, -2:-1]), dim=0
  )
  result += tokenizer.decode(logits[0, -2:-1])

print(result)
```

## Setup for Development

```bash
# Download Repo
git clone https://github.com/AdityaNG/kan-gpt
cd kan-gpt
git pull

# Download Dataset
./scripts/download_webtext.sh

# Install dependencies for development
pip install -r requirements.txt
pip install -e .
```

## Train

Use the following dummy script to make sure everything is working as expected
```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICE="" python3 -m kan_gpt.train --architecture MLP --batch_size 1 --dummy_dataset --device cpu
WANDB_MODE=offline CUDA_VISIBLE_DEVICE="" python3 -m kan_gpt.train --architecture KAN --batch_size 1 --dummy_dataset --device cpu
```

Then make use of the training script
```bash
python -m kan_gpt.train
```

## TODOs

- [x] Integrate [minGPT](https://github.com/karpathy/minGPT) and [pykan](https://github.com/KindXiaoming/pykan)
- [x] Dataset downloading script for [WebText](https://github.com/openai/gpt-2-output-dataset)
- [x] PyTorch Dataset parser for [WebText](https://github.com/openai/gpt-2-output-dataset)
- [ ] Mini training POC for KAN-GPT
  - [x] Integrate KAN training logic from `KAN.train_kan`
  - [ ] Train a dummy batch
- [x] Mini training POC for MLP-GPT
- [x] Train MLP-GPT on the webtext dataset as a baseline
- [ ] Auto Save checkpoints
- [ ] Auto Save checkpoints to W&B
- [ ] Script to load checkpoint in interactive mode
- [ ] Training script to PyTorch Lighting
- [ ] Test Cases
  - [x] KAN: Forward-Backward test
  - [x] GPT: Forward-Backward test
  - [x] KAN_GPT: Forward-Backward test

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## References

- [minGPT](https://github.com/karpathy/minGPT)
- [pykan](https://github.com/KindXiaoming/pykan)
- [WebText](https://github.com/openai/gpt-2-output-dataset)
