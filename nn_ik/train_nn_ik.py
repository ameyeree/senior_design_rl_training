import argparse
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _destandardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


class IKMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=(256, 256, 256), output_dim=3):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    data_path: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    rng = np.random.default_rng(seed)

    data = np.load(data_path)
    positions = data["positions"].astype(np.float32)
    thetas = data["thetas"].astype(np.float32)

    # Normalize inputs and outputs
    pos_mean = positions.mean(axis=0)
    pos_std = positions.std(axis=0) + 1e-8
    theta_mean = thetas.mean(axis=0)
    theta_std = thetas.std(axis=0) + 1e-8

    positions_n = _standardize(positions, pos_mean, pos_std)
    thetas_n = _standardize(thetas, theta_mean, theta_std)

    dataset = TensorDataset(
        torch.from_numpy(positions_n),
        torch.from_numpy(thetas_n),
    )

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = IKMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])

        best_val = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)

            train_loss /= max(1, n_train)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item() * x.size(0)

            val_loss /= max(1, n_val)
            elapsed = time.time() - t0

            writer.writerow([epoch, train_loss, val_loss, elapsed])
            f.flush()

            print(
                f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                f"val={val_loss:.6f} | {elapsed:.2f}s"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item() * x.size(0)

    test_loss /= max(1, n_test)
    print(f"Test MSE: {test_loss:.6f}")

    model_path = os.path.join(out_dir, "ik_mlp.pt")
    torch.save(model.state_dict(), model_path)

    scaler_path = os.path.join(out_dir, "ik_scaler.npz")
    np.savez(
        scaler_path,
        pos_mean=pos_mean,
        pos_std=pos_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NN IK model.")
    parser.add_argument("--data", type=str, default=os.path.join("nn_ik", "data", "fk_dataset.npz"))
    parser.add_argument("--out", type=str, default=os.path.join("nn_ik", "models"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
