# CSC8370 — Robust Federated Learning (Level 3)

Everything for Levels 1–3 lives in this folder.

```
project/
  shared/           # ONE dataloader, models, train helpers
  level1/train.py   # centralized
  level2/train.py   # clean FedAvg
  level3/           # robust FL + malicious_client.py + defense
```

## Setup

```bash
cd project
pip install -r requirements.txt
```

MNIST → `./data/mnist` (course path). Run all commands from **`project/`**.

## Shared pieces

- **Data:** `shared/dataloader.py` — course `Normalize((0.5,), (0.5,))`, same MNIST for all levels; `load_central()` for L1, `load_federated()` for L2 **and** L3
- **Models:** LeNetWide32 (primary), LeNet5 (baseline)
- **Opt:** SGD lr=0.05, momentum 0.9, Nesterov; seed configurable

## Level 1 — Centralized

```bash
python level1/train.py
python level1/train.py --model lenet5
```

## Level 2 — Federated (10 clients, 10 rounds, 1 local epoch)

```bash
python level2/train.py
```

## Level 3 — Robust FL + detection

Attacks (proposal §3): `sign_flip`, `gaussian`, `scale`, `label_flip`  
Detection (§4.1): `cosine_l2`, `cosine`, `l2`, `multi_krum`  
Aggregation (§4.2): `trimmed_mean`, `fedavg`, `multi_krum`

Malicious logic is **only** in `level3/malicious_client.py` (`# === MALICIOUS CLIENT LOGIC HERE ===`).

```bash
python level3/train.py
python level3/train.py --attack gaussian --detect l2 --aggregate trimmed_mean
python level3/train.py --no-defense
```
