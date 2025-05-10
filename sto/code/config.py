from dataclasses import dataclass
import tomllib


@dataclass
class Config:
    OPTIMIZER: str
    BATCH_SIZE: int
    LR: float
    EPOCHS: int
    LAMBDA: float
    NUM_FEATURES: int
    TRAIN_FILE: str
    TEST_FILE: str

    @staticmethod
    def from_file(filename: str) -> "Config":
        with open(filename, "rb") as f:
            config = tomllib.load(f)
        general = config["general"]
        optimizer = config["optimizer"]
        data = config["data"]
        config = {
            "OPTIMIZER": optimizer["name"].lower(),
            "BATCH_SIZE": optimizer["batch_size"],
            "LR": optimizer["lr"],
            "EPOCHS": general["epochs"],
            "LAMBDA": general["lambda"],
            "NUM_FEATURES": general["num_features"],
            "TRAIN_FILE": data["train_file"],
            "TEST_FILE": data["test_file"],
        }
        return Config(**config)
