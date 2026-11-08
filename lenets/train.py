'''Train LeNet-5 (baseline) and/or LeNetWide32 (primary).

Federated (default): 10 clients, 10 FedAvg rounds, 1 local epoch.
Centralized: --central, 10 epochs on the full train set.

Examples:
  python lenets/train.py --model wide          # FedAvg LeNetWide32 (99.3% recipe)
  python lenets/train.py --model lenet5        # FedAvg LeNet-5 baseline
  python lenets/train.py --model both          # both, same protocol
  python lenets/train.py --model wide --central
'''

import argparse

import torch

from common import (
    BATCH_SIZE,
    NUM_CLIENTS,
    ROUNDS,
    SEED,
    central_train,
    deskew_and_center,
    device,
    fedavg_train,
    load_mnist_packed,
    predict,
    report,
    split_clients,
)
from models import LeNet5, LeNetWide32

# Winner recipe for Wide32; LeNet-5 uses the same opt for a fair baseline.
LR = 0.05
WIDE_SEED = 70
LENET5_SEED = 67


def translate(imgs, dx, dy):
    if dx == 0 and dy == 0:
        return imgs
    n, c, h, w = imgs.shape
    out = imgs.new_zeros(n, c, h, w)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = imgs[:, :, src_y0:src_y1, src_x0:src_x1]
    return out


def random_shift_batch(xb, max_shift=1, generator=None):
    n = xb.shape[0]
    dx = torch.randint(-max_shift, max_shift + 1, (n,), generator=generator)
    dy = torch.randint(-max_shift, max_shift + 1, (n,), generator=generator)
    out = xb.new_zeros(xb.shape)
    for sx in range(-max_shift, max_shift + 1):
        for sy in range(-max_shift, max_shift + 1):
            mask = (dx == sx) & (dy == sy)
            if mask.any():
                out[mask] = translate(xb[mask], sx, sy)
    return out


def train_augment(xb, generator=None):
    return random_shift_batch(xb, generator=generator)


def parse_args():
    p = argparse.ArgumentParser(description="LeNet-5 / LeNetWide32 MNIST trainer")
    p.add_argument(
        "--model",
        choices=("wide", "lenet5", "both"),
        default="wide",
        help="Which architecture(s) to train (default: wide)",
    )
    p.add_argument(
        "--central",
        action="store_true",
        help="Centralized 10-epoch training instead of FedAvg",
    )
    return p.parse_args()


def run_one(name, cls, seed, nesterov, img_train, y_train, img_val, y_val, img_test, y_test, client_idx, central):
    augment = train_augment if name == "LeNetWide32" else None
    tag = f"{name} {'central' if central else 'FedAvg'}"
    print(f"\n===== {tag} seed={seed} nesterov={nesterov} =====")
    if central:
        model = central_train(
            cls, img_train, y_train, img_val, y_val, tag, LR,
            augment=augment, seed=seed, nesterov=nesterov,
        )
    else:
        model = fedavg_train(
            cls, img_train, y_train, client_idx, img_val, y_val, tag, LR,
            augment=augment, seed=seed, nesterov=nesterov,
        )
    report(f"{name} Validation", y_val, predict(model, img_val))
    report(f"{name} Test", y_test, predict(model, img_test))
    return model


if __name__ == "__main__":
    args = parse_args()
    mode = "central" if args.central else "federated"
    print(f"Device: {device}")
    print(
        f"{mode}, {NUM_CLIENTS} clients, {ROUNDS} rounds/epochs, "
        f"lr={LR}, batch={BATCH_SIZE}, model={args.model}"
    )

    images, labels, train_idx, val_idx = load_mnist_packed()
    images = deskew_and_center(images)
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[len(train_idx) + len(val_idx):]
    img_train = images[train_idx]
    img_val = images[val_idx]
    img_test = images[len(train_idx) + len(val_idx):]

    client_idx = None
    if not args.central:
        client_idx = split_clients(y_train, NUM_CLIENTS, SEED)
        for i, idx in enumerate(client_idx):
            print(f"  client {i}: {len(idx)} examples")

    jobs = []
    if args.model in ("lenet5", "both"):
        jobs.append(("LeNet5", LeNet5, LENET5_SEED, True))
    if args.model in ("wide", "both"):
        jobs.append(("LeNetWide32", LeNetWide32, WIDE_SEED, True))

    for name, cls, seed, nesterov in jobs:
        run_one(
            name, cls, seed, nesterov,
            img_train, y_train, img_val, y_val, img_test, y_test,
            client_idx, args.central,
        )
