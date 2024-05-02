import torch
from torch.utils.data.dataloader import DataLoader

from kan_gpt.dataset import WebTextDataset
from kan_gpt.mingpt.model import GPT as MLP_GPT
from kan_gpt.mingpt.trainer import Trainer
from kan_gpt.mingpt.utils import set_seed
from kan_gpt.model import GPT as KAN_GPT

set_seed(3407)


def eval_split(
    trainer, split, max_batches, model, train_dataset, test_dataset
):
    dataset = {"train": train_dataset, "test": test_dataset}[split]
    n = train_dataset.length  # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(
        dataset, batch_size=100, num_workers=0, drop_last=False
    )
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(
            inp, n, do_sample=False
        )  # using greedy argmax, not sampling
        sol_candidate = cat[:, n:]  # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (
            (sol == sol_candidate).all(1).cpu()
        )  # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if (
                not correct[i] and mistakes_printed_already < 3
            ):  # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print(
                    "GPT claims that %s sorted is %s but gt is %s"
                    % (
                        inp[i].tolist(),
                        sol_candidate[i].tolist(),
                        sol[i].tolist(),
                    )
                )
        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print(
        "%s final score: %d/%d = %.2f%% correct"
        % (split, rt.sum(), len(results), 100 * rt.mean())
    )
    return rt.sum()


def main(args):
    model_type = args.model_type

    # print an example instance of the dataset
    if args.dummy_dataset:
        train_dataset = WebTextDataset("test", model_type)
    else:
        train_dataset = WebTextDataset("train", model_type)

    test_dataset = WebTextDataset("test", model_type)

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

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = (
        5e-4  # the model we're using is so small that we can go a bit faster
    )
    train_config.max_iters = 2000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)

    def batch_end_callback(trainer):
        # TODO: Add W&B Hooks
        if trainer.iter_num % 100 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

    trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()

    print("=" * 20)
    print("EVAL")
    print("=" * 20)

    model.eval()
    with torch.no_grad():
        train_score = eval_split(
            trainer,
            "train",
            max_batches=50,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        test_score = eval_split(
            trainer,
            "test",
            max_batches=50,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

    print("train_score: ", train_score)
    print("test_score: ", test_score)

    print("=" * 20)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("KAN-GPT Trainer")
    parser.add_argument("--model_type", default="gpt2")
    parser.add_argument("--dummy_dataset", action="store_true")
    parser.add_argument(
        "--architecture", choices=["MLP", "KAN"], default="KAN"
    )

    args = parser.parse_args()

    main(args)
