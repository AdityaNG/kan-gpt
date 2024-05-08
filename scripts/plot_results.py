import wandb

api = wandb.Api()

mlp_run = api.run("KAN-GPT/axi1qzwv")
kan_run = api.run("KAN-GPT/eusdq4te")

keys = [
    'train_loss', 'test_loss'
]

mlp_metrics = mlp_run.history(keys=keys)
kan_metrics = kan_run.history(keys=keys)

print("MLP")
print(mlp_metrics)
print("="*20)

print("KAN")
print(kan_metrics)
print("="*20)

# Plot the test and train losses for the two models

import matplotlib.pyplot as plt

plt.plot(kan_metrics['test_loss'], label='KAN Test', linestyle="--")
plt.plot(kan_metrics['train_loss'], label='KAN Train')

plt.plot(mlp_metrics['test_loss'], label='MLP Test', linestyle="--")
plt.plot(mlp_metrics['train_loss'], label='MLP Train')

# Add a legend and show the plot

plt.xlabel('Steps')
plt.ylabel('Loss')

plt.title("Training Curves of KAN-GPT and MLP-GPT")

# Grid

plt.grid(True)

plt.legend()
plt.show()

