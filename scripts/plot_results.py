import wandb
import numpy as np
import matplotlib.pyplot as plt

api = wandb.Api()

mlp_run = api.run("KAN-GPT/rk3dmrwh")  # axi1qzwv
kan_run = api.run("KAN-GPT/m6msyzlz")  # eusdq4te

keys = [
    "train_loss",
    "train_perplexity",
    "train_f1",
    "train_precision",
    "train_recall",
    "train_cross_entropy",
    "test_loss",
    "test_perplexity",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_cross_entropy",
]

mlp_metrics = mlp_run.history(keys=keys)
kan_metrics = kan_run.history(keys=keys)

print("MLP")
print(mlp_metrics)
print("="*20)

print("KAN")
print(kan_metrics)
print("="*20)

metrics = [
    "Loss",
    "Perplexity",
    "F1",
    "Precision",
    "Recall",
    "Cross Entropy",
]

for metric in metrics:
    id = metric.lower().replace(" ", "_").replace("-", "_")
    # Plot the test and train losses for the two models

    kan_data = kan_metrics[[f"test_{id}", f"train_{id}"]]
    mlp_data = mlp_metrics[[f"test_{id}", f"train_{id}"]]

    kan_data = kan_data.dropna()
    mlp_data = mlp_data.dropna()

    print("MLP")
    print(mlp_data)
    print("="*20)

    print("KAN")
    print(kan_data)
    print("="*20)

    plt.plot(kan_data[f'test_{id}'].astype(np.float16), label='KAN Test', linestyle="--")
    plt.plot(kan_data[f'train_{id}'].astype(np.float16), label='KAN Train')
    plt.plot(mlp_data[f'test_{id}'].astype(np.float16), label='MLP Test', linestyle="--")
    plt.plot(mlp_data[f'train_{id}'].astype(np.float16), label='MLP Train')

    # Add a legend and show the plot

    plt.xlabel('Steps')
    plt.ylabel(metric)

    plt.title(f"{metric} curves: KAN-GPT and MLP-GPT")

    # Grid

    plt.grid(True)

    plt.legend()
    plt.draw()

    # Save to media/results_loss.png

    plt.savefig(f'media/results_{id}.png')

    plt.show()
    plt.cla()