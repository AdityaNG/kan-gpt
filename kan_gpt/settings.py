import os


class KANSettings:

    KAN_IMPLEMENTATION: str = os.getenv(
        "KAN_IMPLEMENTATION", default="EFFICIENT_KAN"
    )

    def __init__(self) -> None:
        assert self.KAN_IMPLEMENTATION in ("EFFICIENT_KAN", "ORIGINAL_KAN")


class Settings:

    kan: KANSettings = KANSettings()


settings = Settings()
