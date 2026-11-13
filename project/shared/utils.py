'''Shared training helpers for Levels 1–3.'''

from pathlib import Path
import copy

import torch
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.05
MOMENTUM = 0.9
NESTEROV = True
NUM_CLIENTS = 10
ROUNDS = 10
LOCAL_EPOCHS = 1


def get_device():
    return device


def make_optimizer(model, lr=LR, nesterov=NESTEROV):
    return optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, nesterov=nesterov)


def train_one_epoch(model, loader, optimizer, device=device):
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def evaluate(model, loader, device=device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def save_checkpoint(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    print(f"Saved model to {path}")


def load_checkpoint(model, path, map_location=None):
    path = Path(path)
    blob = torch.load(path, map_location=map_location or device, weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state)
    print(f"Loaded model from {path}")
    return blob.get("meta", {}) if isinstance(blob, dict) else {}


def clone_state(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def state_dict_to_vector(state_dict):
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in state_dict.values()])


def fedavg(states, weights=None):
    if weights is None:
        weights = [1.0] * len(states)
    total = float(sum(weights))
    avg = {}
    for key in states[0]:
        acc = None
        for state, w in zip(states, weights):
            piece = state[key].float() * w
            acc = piece if acc is None else acc + piece
        avg[key] = (acc / total).to(dtype=states[0][key].dtype)
    return avg


def client_update(model, train_loader, device=device, epochs=LOCAL_EPOCHS, lr=LR, nesterov=NESTEROV):
    opt = make_optimizer(model, lr=lr, nesterov=nesterov)
    for _ in range(epochs):
        train_one_epoch(model, train_loader, opt, device=device)
    return clone_state(model.state_dict())
