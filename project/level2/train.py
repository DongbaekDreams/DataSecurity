'''Level 2: FedAvg MNIST — same shared federated loader as Level 3.

  python level2/train.py
  python level2/train.py --model lenet5
  python level2/train.py --test-only --checkpoint level2/checkpoints/federated_model.pth
'''

from pathlib import Path
import argparse
import copy
import sys

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.dataloader import load_federated
from shared.models import make_model
from shared.utils import (
    LR,
    LOCAL_EPOCHS,
    NESTEROV,
    NUM_CLIENTS,
    ROUNDS,
    client_update,
    evaluate,
    fedavg,
    get_device,
    load_checkpoint,
    save_checkpoint,
)

OUT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CKPT = OUT_DIR / "federated_model.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Level 2 federated MNIST")
    p.add_argument("--model", choices=("wide", "lenet5"), default="wide")
    p.add_argument("--clients", type=int, default=NUM_CLIENTS)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--test-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    print(
        f"Level 2 FedAvg | model={args.model} clients={args.clients} "
        f"rounds={args.rounds} local_epochs={args.local_epochs} "
        f"lr={LR} nesterov={NESTEROV} seed={args.seed}"
    )

    client_datasets, client_loaders, test_loader = load_federated(n_clients=args.clients)
    global_model = make_model(args.model).to(device)

    if args.test_only:
        load_checkpoint(global_model, args.checkpoint)
        loss, acc = evaluate(global_model, test_loader, device)
        print(f"Test loss={loss:.4f} acc={100 * acc:.2f}%")
        return

    for round_id in range(args.rounds):
        states, weights = [], []
        for client_id, loader in enumerate(tqdm(client_loaders, desc=f"round {round_id + 1}/{args.rounds}")):
            client = copy.deepcopy(global_model)
            state = client_update(
                client, loader, device=device, epochs=args.local_epochs, lr=LR, nesterov=NESTEROV,
            )
            states.append(state)
            weights.append(len(client_datasets[client_id]))
        global_model.load_state_dict(fedavg(states, weights))
        loss, acc = evaluate(global_model, test_loader, device)
        print(f"{round_id} loss={loss:.4f} acc={100 * acc:.2f}%")

    save_checkpoint(
        global_model, args.checkpoint,
        meta={"level": 2, "model": args.model, "seed": args.seed, "acc": acc},
    )
    print(f"Final test acc={100 * acc:.2f}%")


if __name__ == "__main__":
    main()
