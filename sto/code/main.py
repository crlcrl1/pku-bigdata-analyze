import argparse

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import Config
from data import SVMDataset, TensorDataloader
from model import SimpleModel
from optimizer import Adagrad, Adam


def train(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y = torch.load(config.TRAIN_FILE)
    test_x, test_y = torch.load(config.TEST_FILE)

    train_dataset = SVMDataset(train_x, train_y, device)
    test_dataset = SVMDataset(test_x, test_y, device)
    train_loader = TensorDataloader(train_dataset, batch_size=config.BATCH_SIZE)
    test_loader = TensorDataloader(test_dataset, batch_size=len(test_dataset))

    model = SimpleModel(config.NUM_FEATURES).to(device)
    model.train()

    match config.OPTIMIZER:
        case "adagrad":
            optimizer = Adagrad(model.parameters(), lr=config.LR)
        case "adam":
            optimizer = Adam(model.parameters(), lr=config.LR)
        case _:
            raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")

    train_errors = []
    test_errors = []
    for _ in tqdm(range(config.EPOCHS)):
        train_error = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            error = (1 - torch.tanh(outputs * batch_y)).mean()
            train_error += error.item() * batch_y.size(0)
            loss = error + config.LAMBDA * torch.norm(model.w, p=2) ** 2
            loss.backward()
            optimizer.step()
        train_errors.append(train_error / len(train_loader.dataset))

        with torch.no_grad():
            test_error = 0
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                error = (1 - torch.tanh(outputs * batch_y)).mean()
                test_error += error.item() * batch_y.size(0)
            test_errors.append(test_error / len(test_loader.dataset))

    return model, train_errors, test_errors


def validate(model: SimpleModel, config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = torch.load(config.TEST_FILE)

    test_dataset = SVMDataset(x, y, device)
    test_loader = TensorDataloader(test_dataset, batch_size=len(test_dataset))

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predicted = torch.sign(outputs)
            total += batch_y.size(0)
            correct += ((predicted * batch_y) > 0).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train and validate a model.")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    model, train_errors, test_errors = train(config)
    validate(model, config)

    plt.plot(train_errors, label="Train Error")
    plt.plot(test_errors, label="Test Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Train and Test Error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
