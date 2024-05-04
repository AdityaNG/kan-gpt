import torch
from kan_gpt.mingpt.model import GPT as MLP_GPT

VOCAB_SIZE = 8
BLOCK_SIZE = 16
MODEL_TYPE = "gpt-pico"


def get_gpt_model() -> MLP_GPT:
    model_config = MLP_GPT.get_default_config()
    model_config.model_type = MODEL_TYPE
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = BLOCK_SIZE
    model = MLP_GPT(model_config)
    return model


def test_forward():
    with torch.no_grad():
        model = get_gpt_model()
        x = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)

        y, loss = model.forward(x)

        assert y.shape == (
            1,
            BLOCK_SIZE,
            VOCAB_SIZE,
        ), f"Shape mismatch: {y.shape}"


def test_backward():
    model = get_gpt_model()
    x = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)

    # Make sure grads exist
    requires_grad_set = set()
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_set.add(param)
    assert len(requires_grad_set) > 0, "requires_grad is not set"

    y, loss = model.forward(x)

    assert y.shape == (1, BLOCK_SIZE, VOCAB_SIZE), f"Shape mismatch: {y.shape}"

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
        model = get_gpt_model()
        x = torch.zeros((2, BLOCK_SIZE), dtype=torch.long)

        y, loss = model.forward(x)

        assert y.shape == (
            2,
            BLOCK_SIZE,
            VOCAB_SIZE,
        ), f"Shape mismatch: {y.shape}"


def test_backward_batched():
    model = get_gpt_model()
    x = torch.zeros((2, BLOCK_SIZE), dtype=torch.long)

    # Make sure grads exist
    requires_grad_set = set()
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_set.add(param)
    assert len(requires_grad_set) > 0, "requires_grad is not set"

    y, loss = model.forward(x)

    assert y.shape == (2, BLOCK_SIZE, VOCAB_SIZE), f"Shape mismatch: {y.shape}"

    loss = y.mean()
    loss.backward()

    # Make sure grads exist
    grad_set = set()
    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            grad_set.add(param)
    assert len(grad_set) > 0, f"Tensor.grad missing"
