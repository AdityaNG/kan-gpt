import wandb
from kan_gpt.train import main


def wandb_sweep():
    run = wandb.init(resume="allow", anonymous="must")

    class Args:
        model_type = wandb.config.model_type
        dummy_dataset = wandb.config.dummy_dataset
        learning_rate = wandb.config.learning_rate
        max_iters = wandb.config.max_iters
        num_workers = wandb.config.num_workers
        batch_size = wandb.config.batch_size
        dataset = wandb.config.dataset
        architecture = wandb.config.architecture
        device = wandb.config.device

    run_args = Args()

    main(args=run_args, run=run)


def sweep(args):
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "test_loss"},
        "parameters": {
            "model_type": {"values": args.model_type},
            "batch_size": {"values": args.batch_size},
            "dummy_dataset": {"values": args.dummy_dataset},
            "learning_rate": {"values": args.learning_rate},
            "max_iters": {"values": args.max_iters},
            "num_workers": {"values": args.num_workers},
            "dataset": {"values": args.dataset},
            "architecture": {"values": args.architecture},
            "device": {"values": args.device},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="KAN-GPT")
    print("sweep_id (generated)", sweep_id)

    wandb.agent(sweep_id, function=wandb_sweep)


if __name__ == "__main__":

    class SweepArgs:
        model_type = ["gpt-mini", "gpt-nano", "gpt2"]
        dummy_dataset = [
            False,
        ]
        learning_rate = [5e-3, 5e-4, 5e-5, 5e-6]
        max_iters = [
            1000,
        ]
        num_workers = [
            0,
        ]
        batch_size = [1, 2, 4, 8, 12, 16]
        dataset = [
            "tinyshakespeare",
        ]
        architecture = ["MLP", "KAN"]
        device = [
            "auto",
        ]

    sweep(SweepArgs())
