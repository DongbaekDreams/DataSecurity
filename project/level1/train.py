'''Level 1: Centralized MNIST (course loader via shared.dataloader).

  python level1/train.py
  python level1/train.py --model lenet5
  python level1/train.py --test-only --checkpoint level1/checkpoints/best_model.pth
'''

from pathlib import Path
import argparse
import sys

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.dataloader import load_central
from shared.models import make_model
from shared.utils import (
    LR,
    NESTEROV,
    ROUNDS,
    evaluate,
    get_device,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
)

OUT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CKPT = OUT_DIR / "best_model.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Level 1 centralized MNIST")
    p.add_argument("--model", choices=("wide", "lenet5"), default="wide")
    p.add_argument("--epochs", type=int, default=ROUNDS)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--test-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    print(f"Level 1 central | model={args.model} epochs={args.epochs} lr={LR} nesterov={NESTEROV} seed={args.seed}")

    train_loader, test_loader = load_central()
    model = make_model(args.model).to(device)

    if args.test_only:
        load_checkpoint(model, args.checkpoint)
        loss, acc = evaluate(model, test_loader, device)
        print(f"Test loss={loss:.4f} acc={100 * acc:.2f}%")
        return

    opt = make_optimizer(model)
    best_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss, acc = evaluate(model, test_loader, device)
        print(f"{epoch} loss={loss:.4f} acc={100 * acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(
                model, args.checkpoint,
                meta={"level": 1, "model": args.model, "seed": args.seed, "acc": best_acc},
            )

    load_checkpoint(model, args.checkpoint)
    loss, acc = evaluate(model, test_loader, device)
    print(f"Best test acc={100 * acc:.2f}%  ({args.checkpoint})")


if __name__ == "__main__":
    main()
