import os
import torch
import pytest
from unittest import mock

VOCAB_SIZE = 8
BLOCK_SIZE = 16
MODEL_TYPE = "gpt-pico"


@mock.patch.dict(
    os.environ, {"KAN_IMPLEMENTATION": "EFFICIENT_KAN"}, clear=True
)
def get_gpt_model_efficient():
    from kan_gpt.model import GPT as KAN_GPT

    model_config = KAN_GPT.get_default_config()
    model_config.model_type = MODEL_TYPE
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = BLOCK_SIZE
    model = KAN_GPT(model_config)

    del KAN_GPT

    return model


@mock.patch.dict(
    os.environ, {"KAN_IMPLEMENTATION": "ORIGINAL_KAN"}, clear=True
)
def get_gpt_model_original():
    from kan_gpt.model import GPT as KAN_GPT

    model_config = KAN_GPT.get_default_config()
    model_config.model_type = MODEL_TYPE
    model_config.vocab_size = VOCAB_SIZE
    model_config.block_size = BLOCK_SIZE
    model = KAN_GPT(model_config)

    del KAN_GPT

    return model


@pytest.fixture
def model(request):
    return request.param()


@pytest.mark.parametrize(
    "model", (get_gpt_model_efficient, get_gpt_model_original), indirect=True
)
def test_forward(model):
    with torch.no_grad():

        x = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)

        y, loss = model.forward(x)

        assert y.shape == (
            1,
            BLOCK_SIZE,
            VOCAB_SIZE,
        ), f"Shape mismatch: {y.shape}"


@pytest.mark.parametrize(
    "model", (get_gpt_model_efficient, get_gpt_model_original), indirect=True
)
def test_backward(model):
    model = model
    x = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)
    y_gt = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)

    # Make sure grads exist
    requires_grad_set = set()
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_set.add(param)
    assert len(requires_grad_set) > 0, "requires_grad is not set"

    y, loss = model.forward(x, y_gt)

    assert y.shape == (1, BLOCK_SIZE, VOCAB_SIZE), f"Shape mismatch: {y.shape}"

    loss.backward()

    # Make sure grads exist
    grad_set = set()
    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            grad_set.add(param)
    assert len(grad_set) > 0, f"Tensor.grad missing"


@pytest.mark.parametrize(
    "model", (get_gpt_model_efficient, get_gpt_model_original), indirect=True
)
def test_forward_batched(model):
    with torch.no_grad():

        x = torch.zeros((2, BLOCK_SIZE), dtype=torch.long)

        y, loss = model.forward(x)

        assert y.shape == (
            2,
            BLOCK_SIZE,
            VOCAB_SIZE,
        ), f"Shape mismatch: {y.shape}"


@pytest.mark.parametrize(
    "model", (get_gpt_model_efficient, get_gpt_model_original), indirect=True
)
def test_backward_batched(model):
    model = model
    x = torch.zeros((2, BLOCK_SIZE), dtype=torch.long)
    y_gt = torch.zeros((2, BLOCK_SIZE), dtype=torch.long)

    # Make sure grads exist
    requires_grad_set = set()
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_set.add(param)
    assert len(requires_grad_set) > 0, "requires_grad is not set"

    y, loss = model.forward(x, y_gt)

    assert y.shape == (2, BLOCK_SIZE, VOCAB_SIZE), f"Shape mismatch: {y.shape}"

    loss.backward()

    # Make sure grads exist
    grad_set = set()
    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            grad_set.add(param)
    assert len(grad_set) > 0, f"Tensor.grad missing"
