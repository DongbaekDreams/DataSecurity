'''Level 3: Robust FL + attack detection (same shared federated loader as Level 2).

Malicious logic only in malicious_client.py for Demo Day swap-out.

  python level3/train.py
  python level3/train.py --attack sign_flip --detect cosine_l2 --aggregate trimmed_mean
  python level3/train.py --no-defense
  python level3/train.py --test-only --checkpoint level3/checkpoints/robust_model.pth
'''

from pathlib import Path
import argparse
import copy
import sys

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import level3.malicious_client as malicious_client
from level3.defense import robust_aggregate
from level3.malicious_client import generate_malicious_update
from shared.dataloader import load_federated
from shared.models import make_model
from shared.utils import (
    LR,
    LOCAL_EPOCHS,
    NESTEROV,
    NUM_CLIENTS,
    ROUNDS,
    client_update,
    clone_state,
    evaluate,
    fedavg,
    get_device,
    load_checkpoint,
    save_checkpoint,
)

OUT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CKPT = OUT_DIR / "robust_model.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Level 3 robust federated MNIST")
    p.add_argument("--model", choices=("wide", "lenet5"), default="wide")
    p.add_argument("--clients", type=int, default=NUM_CLIENTS)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--malicious-id", type=int, default=5)
    p.add_argument("--attack-round", type=int, default=2, help="1-based; must be >= 2")
    p.add_argument(
        "--attack",
        choices=("sign_flip", "gaussian", "scale", "label_flip"),
        default="sign_flip",
    )
    p.add_argument(
        "--detect",
        choices=("cosine_l2", "cosine", "l2", "multi_krum"),
        default="cosine_l2",
    )
    p.add_argument(
        "--aggregate",
        choices=("trimmed_mean", "fedavg", "multi_krum"),
        default="trimmed_mean",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--no-defense", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.malicious_id < args.clients):
        raise SystemExit(f"--malicious-id must be in [0, {args.clients})")
    if args.attack_round < 2:
        raise SystemExit("--attack-round must be >= 2")

    malicious_client.ATTACK_TYPE = args.attack
    device = get_device()
    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    print(
        f"Level 3 robust FL | model={args.model} clients={args.clients} "
        f"rounds={args.rounds} local_epochs={args.local_epochs} "
        f"malicious_id={args.malicious_id} attack_round={args.attack_round} "
        f"attack={args.attack} detect={args.detect} aggregate={args.aggregate} "
        f"defense={not args.no_defense} lr={LR} nesterov={NESTEROV} seed={args.seed}"
    )

    client_datasets, client_loaders, test_loader = load_federated(n_clients=args.clients)
    for i, ds in enumerate(client_datasets):
        tag = "MALICIOUS" if i == args.malicious_id else "benign"
        print(f"  role client {i}: {len(ds)} [{tag}]")

    global_model = make_model(args.model).to(device)

    if args.test_only:
        load_checkpoint(global_model, args.checkpoint)
        loss, acc = evaluate(global_model, test_loader, device)
        print(f"Test loss={loss:.4f} acc={100 * acc:.2f}%")
        return

    detections = []
    for round_id in range(args.rounds):
        round_num = round_id + 1
        attacking = round_num >= args.attack_round
        global_snapshot = clone_state(global_model.state_dict())
        states, weights = [], []

        for client_id, loader in enumerate(tqdm(client_loaders, desc=f"round {round_num}/{args.rounds}")):
            client = copy.deepcopy(global_model)
            is_mal = client_id == args.malicious_id and attacking
            if is_mal:
                # === MALICIOUS CLIENT LOGIC HERE ===
                state = generate_malicious_update(
                    clone_state(global_snapshot),
                    client_datasets[client_id],
                    device,
                )
            else:
                state = client_update(
                    client, loader, device=device,
                    epochs=args.local_epochs, lr=LR, nesterov=NESTEROV,
                )
            states.append({k: v.detach().cpu() for k, v in state.items()})
            weights.append(len(client_datasets[client_id]))

        if args.no_defense:
            agg = fedavg(states, weights)
            details = {"flagged": [], "kept": list(range(args.clients))}
        else:
            agg, details = robust_aggregate(
                global_snapshot, states, weights=weights, max_flag=1,
                detect=args.detect, aggregate=args.aggregate,
            )

        global_model.load_state_dict(agg)
        loss, acc = evaluate(global_model, test_loader, device)
        flagged = details.get("flagged", [])
        detections.append(flagged)
        caught = args.malicious_id in flagged
        print(
            f"{round_id} loss={loss:.4f} acc={100 * acc:.2f}%  flagged={flagged}  "
            f"attacking={attacking}  caught_malicious={caught}"
        )

    caught_rounds = [
        i + 1 for i, flagged in enumerate(detections)
        if (i + 1) >= args.attack_round and args.malicious_id in flagged
    ]
    print(
        f"Detection: malicious client {args.malicious_id} caught in rounds {caught_rounds or 'NONE'} "
        f"(attack started round {args.attack_round})"
    )
    save_checkpoint(
        global_model, args.checkpoint,
        meta={
            "level": 3,
            "model": args.model,
            "seed": args.seed,
            "acc": acc,
            "attack": args.attack,
            "detect": args.detect,
            "aggregate": args.aggregate,
            "malicious_id": args.malicious_id,
            "attack_round": args.attack_round,
            "caught_rounds": caught_rounds,
        },
    )
    print(f"Final test acc={100 * acc:.2f}%")


if __name__ == "__main__":
    main()
