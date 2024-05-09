from tempfile import TemporaryDirectory
import torch
from kan_gpt.kan.KAN import KAN
from kan_gpt.kan.utils import create_dataset


def test_forward():
    with torch.no_grad():
        model = KAN(width=[2, 5, 2])
        x = torch.zeros((1, 1, 2), dtype=torch.float32)

        y = model.forward(x)

        assert y.shape == (1, 1, 2), f"Shape mismatch: {y.shape}"


def test_backward():
    model = KAN(width=[2, 5, 2])
    x = torch.zeros((1, 1, 2), dtype=torch.float32)

    # Make sure grads exist
    requires_grad_set = set()
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_set.add(param)
    assert len(requires_grad_set) > 0, "requires_grad is not set"

    y = model.forward(x)

    assert y.shape == (1, 1, 2), f"Shape mismatch: {y.shape}"

    loss = y.mean()
    loss.backward()

    # Make sure grads exist
    grad_set = set()
    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            grad_set.add(param)
    assert len(grad_set) > 0, f"Tensor.grad missing"


def test_forward_batched():
    with torch.no_grad():
        model = KAN(width=[2, 5, 2])
        x = torch.zeros((2, 1, 2), dtype=torch.float32)

        y = model.forward(x)

        assert y.shape == (2, 1, 2), f"Shape mismatch: {y.shape}"


def test_backward_batched():
    model = KAN(width=[2, 5, 2])
    x = torch.zeros((2, 1, 2), dtype=torch.float32)

    # Make sure grads exist
    requires_grad_set = set()
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_set.add(param)
    assert len(requires_grad_set) > 0, "requires_grad is not set"

    y = model.forward(x)

    assert y.shape == (2, 1, 2), f"Shape mismatch: {y.shape}"

    loss = y.mean()
    loss.backward()

    # Make sure grads exist
    grad_set = set()
    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            grad_set.add(param)
    assert len(grad_set) > 0, f"Tensor.grad missing"


def test_plot():
    model = KAN(width=[2, 3, 2, 1])
    x = torch.normal(0, 1, size=(100, 1, 2))
    model(x)
    beta = 100
    with TemporaryDirectory() as folder:
        model.plot(beta=beta, folder=folder)


def test_train():
    f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
    dataset = create_dataset(f, n_var=2)
    dataset["train_input"].shape, dataset["train_label"].shape

    model = KAN(width=[2, 1], grid=5, k=3, seed=0)
    model.train_kan(dataset, opt="LBFGS", steps=1, lamb=0.1)
    model.plot()
    model.prune()
    with TemporaryDirectory() as folder:
        model.plot(mask=True, folder=folder)
