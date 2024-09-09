#!/bin/bash

# Download Dataset
python3 -m kan_gpt.download_dataset --dataset tinyshakespeare
python3 -m kan_gpt.download_dataset --dataset mnist
python3 -m kan_gpt.download_dataset --dataset webtext

# Train
python3 -m kan_gpt.train --dataset mnist --architecture MLP
python3 -m kan_gpt.train --dataset mnist --architecture KAN

python3 -m kan_gpt.train --dataset tinyshakespeare --architecture MLP
python3 -m kan_gpt.train --dataset tinyshakespeare --architecture KAN

python3 -m kan_gpt.train --dataset webtext --architecture MLP
python3 -m kan_gpt.train --dataset webtext --architecture KAN
