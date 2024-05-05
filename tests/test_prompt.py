from kan_gpt.prompt import main

VOCAB_SIZE = 8
BLOCK_SIZE = 16
MODEL_TYPE = "gpt-pico"


def test_train():
    class Args:
        model_type = MODEL_TYPE
        model_path = None
        max_tokens = 3
        prompt = "Bangalore is often described as the "
        architecture = "KAN"
        device = "cpu"

    args = Args()
    main(args)
