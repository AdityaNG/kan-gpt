import torch
from kan_gpt.kan.KAN import KAN


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
