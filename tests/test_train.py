import os
import torch
from kan_gpt.mingpt.model import GPT as MLP_GPT
from kan_gpt.train import save_model

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


def test_save_model():
    with torch.no_grad():
        model = get_gpt_model()
        save_path = save_model(model, None)

        assert os.path.isfile(save_path), f"Model not saved at {save_path}"

        model_loaded = get_gpt_model()
        model_loaded.load_state_dict(torch.load(save_path))

        # Assert that the loaded model is identical to the original one
        for name, param in model.named_parameters():
            assert torch.equal(param, model_loaded.state_dict()[name]), (
                f"Model not saved correctly at {save_path}, parameter "
                f"{name} does not match original model"
            )
