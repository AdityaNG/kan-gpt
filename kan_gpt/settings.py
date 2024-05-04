import os


class TrainSettings:

    WEIGHTS_PATH: str = os.getenv("WEIGHTS_PATH", default="weights")


class KANSettings:

    KAN_IMPLEMENTATION: str = os.getenv(
        "KAN_IMPLEMENTATION", default="EFFICIENT_KAN"
    )

    def __init__(self) -> None:
        assert self.KAN_IMPLEMENTATION in ("EFFICIENT_KAN", "ORIGINAL_KAN")


class Settings:

    kan: KANSettings = KANSettings()
    train: TrainSettings = TrainSettings()


settings = Settings()
