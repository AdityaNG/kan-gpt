# KAN-GPT

[![codecov](https://codecov.io/gh/AdityaNG/kan-gpt/branch/main/graph/badge.svg?token=kan-gpt_token_here)](https://codecov.io/gh/AdityaNG/kan-gpt)
[![CI](https://github.com/AdityaNG/kan-gpt/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/kan-gpt/actions/workflows/main.yml)

Awesome KAN-GPT created by AdityaNG

## Install it from PyPI

```bash
pip install kan_gpt
```

## Train

Dummy script to make sure everything is working as expected
```bash
CUDA_VISIBLE_DEVICE="0" python3 -m kan_gpt.train --architecture MLP --batch_size 1 --dummy_dataset
```

## Usage

```py
from kan_gpt import BaseClass
from kan_gpt import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m kan_gpt
#or
$ kan_gpt
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## References

- [minGPT](https://github.com/karpathy/minGPT)
- [pykan](https://github.com/KindXiaoming/pykan)
- [WebText](https://github.com/openai/gpt-2-output-dataset)
