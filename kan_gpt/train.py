import torch
from torch.utils.data.dataloader import DataLoader

import wandb
from kan_gpt.dataset import WebTextDataset
from kan_gpt.mingpt.model import GPT as MLP_GPT
from kan_gpt.mingpt.trainer import Trainer
from kan_gpt.model import GPT as KAN_GPT


def eval_split(
    trainer, split, max_batches, batch_size, model, train_dataset, test_dataset
):
    dataset = {"train": train_dataset, "test": test_dataset}[split]
    results = []

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, drop_last=False
    )
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)

        logits, loss = model(x, y)

        results.append(loss)

        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s loss: %.2f" % (split, rt.mean()))
    return rt.mean()


def main(args):
    config = {
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "dummy_dataset": args.dummy_dataset,
        "learning_rate": args.learning_rate,
        "max_iters": args.max_iters,
        "num_workers": args.num_workers,
        "architecture": args.architecture,
        "device": args.device,
    }

    wandb.init(project="KAN-GPT", config=config)

    model_type = args.model_type

    # print an example instance of the dataset
    if args.dummy_dataset:
        train_dataset = WebTextDataset("test", "gpt2")
    else:
        train_dataset = WebTextDataset("train", "gpt2")

    test_dataset = WebTextDataset("test", "gpt2")

    if args.architecture == "KAN":
        GPT = KAN_GPT
    else:
        GPT = MLP_GPT

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    wandb.watch(model)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = float(
        args.learning_rate
    )  # the model we're using is so small that we can go a bit faster
    train_config.max_iters = int(args.max_iters)
    train_config.num_workers = int(args.num_workers)
    train_config.batch_size = int(args.batch_size)
    train_config.device = args.device
    trainer = Trainer(train_config, model, train_dataset)

    def batch_end_callback(trainer):
        # TODO: Add W&B Hooks
        if trainer.iter_num % 100 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; "
                f"iter {trainer.iter_num}: "
                f"train loss {trainer.loss.item():.5f}"
            )

            print("=" * 20)
            print("EVAL")
            print("=" * 20)

            model.eval()
            with torch.no_grad():
                train_score = eval_split(
                    trainer,
                    "train",
                    max_batches=5,
                    batch_size=int(args.batch_size),
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                )
                test_score = eval_split(
                    trainer,
                    "test",
                    max_batches=5,
                    batch_size=int(args.batch_size),
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                )

            model.train()
            print("train_score: ", train_score)
            print("test_score: ", test_score)

            wandb.log(
                {
                    "train_loss": train_score,
                    "test_loss": test_score,
                }
            )

            print("=" * 20)

    trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("KAN-GPT Trainer")
    parser.add_argument("--model_type", default="gpt-mini")
    parser.add_argument("--dummy_dataset", action="store_true")
    parser.add_argument("--learning_rate", default=5e-3)
    parser.add_argument("--max_iters", default=2000)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--batch_size", default=64)

    parser.add_argument(
        "--architecture", choices=["MLP", "KAN"], default="KAN"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    args = parser.parse_args()

    args.learning_rate = float(args.learning_rate)
    args.max_iters = int(args.max_iters)
    args.num_workers = int(args.num_workers)
    args.batch_size = int(args.batch_size)

    main(args)
