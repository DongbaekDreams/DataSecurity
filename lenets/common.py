'''Shared MNIST load, deskew/center, FedAvg, and centralized training helpers.'''

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os

from torch import optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.auto import tqdm

LENETS_DIR = Path(__file__).resolve().parent
ROOT = LENETS_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = LENETS_DIR / "out"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 67
NUM_CLIENTS = 10
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
CLIENT_WORKERS = min(NUM_CLIENTS, os.cpu_count() or 1)


def loss_batch(model, loss_func, xb, yb, opt=None):
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def dataset_to_tensors(ds):
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    images, labels = [], []
    for xb, yb in loader:
        images.append(xb)
        labels.append(yb)
    return torch.cat(images), torch.cat(labels)


def load_mnist_packed():
    '''Return images, labels packed as [train | val | test], plus index ranges.'''
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_ds, validation_ds = random_split(
        training_data, [0.7, 0.3], generator=torch.Generator().manual_seed(SEED),
    )
    train_images, train_labels = dataset_to_tensors(train_ds)
    val_images, val_labels = dataset_to_tensors(validation_ds)
    test_images, test_labels = dataset_to_tensors(test_ds)
    images = torch.cat([train_images, val_images, test_images])
    labels = torch.cat([train_labels, val_labels, test_labels])
    n_train, n_val = len(train_labels), len(val_labels)
    return images, labels, torch.arange(n_train), torch.arange(n_train, n_train + n_val)


@torch.no_grad()
def _deskew_center_batch(imgs):
    '''Shear-correct slant (mu11/mu02), then put ink center of mass at the canvas center.'''
    n, _, h, w = imgs.shape
    ink = imgs.clamp(min=0)
    mass = ink.sum(dim=(2, 3)).clamp(min=1e-6).reshape(n)
    y = torch.arange(h, dtype=imgs.dtype, device=imgs.device).view(1, 1, h, 1)
    x = torch.arange(w, dtype=imgs.dtype, device=imgs.device).view(1, 1, 1, w)
    cy = (ink * y).sum(dim=(2, 3)).reshape(n) / mass
    cx = (ink * x).sum(dim=(2, 3)).reshape(n) / mass
    y0 = y - cy.view(n, 1, 1, 1)
    x0 = x - cx.view(n, 1, 1, 1)
    mu02 = (ink * y0 * y0).sum(dim=(2, 3)).reshape(n) / mass
    mu11 = (ink * x0 * y0).sum(dim=(2, 3)).reshape(n) / mass
    skew = (mu11 / mu02.clamp(min=1e-4)).clamp(-1.0, 1.0)

    theta = imgs.new_zeros(n, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 1] = skew * (h - 1) / (w - 1)
    grid = F.affine_grid(theta, imgs.size(), align_corners=True)
    deskewed = F.grid_sample(imgs, grid, align_corners=True, padding_mode="zeros")

    ink = deskewed.clamp(min=0)
    mass = ink.sum(dim=(2, 3)).clamp(min=1e-6).reshape(n)
    cy = (ink * y).sum(dim=(2, 3)).reshape(n) / mass
    cx = (ink * x).sum(dim=(2, 3)).reshape(n) / mass
    theta = imgs.new_zeros(n, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * ((w - 1) / 2 - cx) / (w - 1)
    theta[:, 1, 2] = -2 * ((h - 1) / 2 - cy) / (h - 1)
    grid = F.affine_grid(theta, imgs.size(), align_corners=True)
    return F.grid_sample(deskewed, grid, align_corners=True, padding_mode="zeros")


def deskew_and_center(images, batch_size=1024):
    chunks = []
    for start in tqdm(range(0, len(images), batch_size), desc="Deskew + center"):
        chunks.append(_deskew_center_batch(images[start:start + batch_size]))
    return torch.cat(chunks)


def split_clients(y_train, n_clients, seed):
    g = torch.Generator().manual_seed(seed)
    parts = [[] for _ in range(n_clients)]
    for digit in range(10):
        idx = (y_train == digit).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        for client_id, shard in enumerate(idx.chunk(n_clients)):
            if len(shard):
                parts[client_id].append(shard)
    return [torch.cat(p) for p in parts]


def fedavg(states, weights):
    total = float(sum(weights))
    avg = {}
    for key in states[0]:
        acc = None
        for state, weight in zip(states, weights):
            piece = state[key].float() * weight
            acc = piece if acc is None else acc + piece
        avg[key] = acc / total
    return avg


def local_train(init_state, make_model, X, y, lr, augment=None, nesterov=False, rng_seed=None):
    client = make_model().to(device)
    client.load_state_dict(init_state)
    opt = optim.SGD(client.parameters(), lr=lr, momentum=0.9, nesterov=nesterov)
    g = torch.Generator()
    if rng_seed is not None:
        g.manual_seed(int(rng_seed) % (2**31 - 1))
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=min(BATCH_SIZE, len(X)),
        shuffle=True,
        generator=g,
    )
    client.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb, generator=g)
            loss_batch(client, F.cross_entropy, xb, yb, opt)
    return client.state_dict(), len(X)


@torch.no_grad()
def eval_loss_acc(model, loader):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return loss_sum / n, correct / n


def central_train(make_model, X_train, y_train, X_val, y_val, desc, lr, augment=None, seed=None, nesterov=False):
    run_seed = SEED if seed is None else seed
    torch.manual_seed(run_seed)
    model = make_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{desc}: {n_params:,} parameters, lr={lr}, nesterov={nesterov}, central {ROUNDS} epochs")
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=nesterov)
    g = torch.Generator()
    g.manual_seed(run_seed)
    train_dl = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=min(BATCH_SIZE, len(X_train)),
        shuffle=True,
        generator=g,
    )
    valid_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    for epoch in range(ROUNDS):
        model.train()
        for xb, yb in tqdm(train_dl, desc=f"{desc} epoch {epoch + 1}/{ROUNDS}"):
            if augment is not None:
                xb = augment(xb, generator=g)
            loss_batch(model, F.cross_entropy, xb, yb, opt)
        val_loss, val_acc = eval_loss_acc(model, valid_dl)
        print(f"{epoch} loss={val_loss:.4f} acc={100 * val_acc:.1f}%")
    return model


def fedavg_train(make_model, X_train, y_train, client_idx, X_val, y_val, desc, lr, augment=None, seed=None, nesterov=False):
    run_seed = SEED if seed is None else seed
    torch.manual_seed(run_seed)
    global_model = make_model().to(device)
    valid_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    n_params = sum(p.numel() for p in global_model.parameters())
    workers = CLIENT_WORKERS
    print(f"{desc}: {n_params:,} parameters, lr={lr}, nesterov={nesterov}, client_threads={workers}")
    prev_threads = torch.get_num_threads()
    if workers > 1:
        torch.set_num_threads(1)
    try:
        for round_id in range(ROUNDS):
            snapshot = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
            states = [None] * len(client_idx)
            weights = [None] * len(client_idx)

            def run_client(client_id, idx):
                return local_train(
                    snapshot, make_model, X_train[idx], y_train[idx], lr, augment,
                    nesterov=nesterov,
                    rng_seed=run_seed * 1_000_003 + round_id * 1_009 + client_id,
                )

            if workers <= 1:
                for client_id, idx in enumerate(tqdm(client_idx, desc=f"{desc} round {round_id + 1}/{ROUNDS}")):
                    states[client_id], weights[client_id] = run_client(client_id, idx)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = {
                        pool.submit(run_client, client_id, idx): client_id
                        for client_id, idx in enumerate(client_idx)
                    }
                    for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{desc} round {round_id + 1}/{ROUNDS}"):
                        client_id = futs[fut]
                        states[client_id], weights[client_id] = fut.result()
            global_model.load_state_dict(fedavg(states, weights))
            val_loss, val_acc = eval_loss_acc(global_model, valid_dl)
            print(f"{round_id} loss={val_loss:.4f} acc={100 * val_acc:.1f}%")
    finally:
        torch.set_num_threads(prev_threads)
    return global_model


@torch.no_grad()
def predict(model, X):
    model.eval()
    chunks = []
    for start in range(0, len(X), BATCH_SIZE):
        xb = X[start:start + BATCH_SIZE].to(device, non_blocking=True)
        chunks.append(model(xb).argmax(1).cpu())
    return torch.cat(chunks)


def report(name, y, pred):
    acc = (pred == y).float().mean().item()
    print(f"{name} Results:")
    print(f"Accuracy: {(100 * acc):.1f}%")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y.tolist(), pred.tolist(), labels=range(10), ax=ax)
    ax.set_title(name)
    path = OUT_DIR / ("".join(ch if ch.isalnum() else "_" for ch in name).strip("_") + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {path}")
    return acc
