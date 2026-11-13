### Level 3 Code Structure Requirement (For Demo Day Testing)
'''Simulated malicious updates covering proposal attack types.

Demo Day: TA replaces this file. Keep generate_malicious_update() signature unchanged.
Set ATTACK_TYPE for our experiments: sign_flip | gaussian | scale | label_flip
'''

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

ATTACK_TYPE = "sign_flip"
ATTACK_SCALE = 10.0


def generate_malicious_update(client_model_state_dict, local_dataset, device):
    """
    Args:
        client_model_state_dict (dict): A copy of the current global model's state_dict().
        local_dataset (Dataset): The local dataset assigned to this client.
        device (torch.device): CPU or CUDA device.
    Returns:
        malicious_state_dict (dict): The modified model parameters (state_dict) to be uploaded to the server.
    """
    # === STUDENTS: IMPLEMENT YOUR SIMULATED ATTACK BELOW ===
    # === MALICIOUS CLIENT LOGIC HERE ===
    attack = ATTACK_TYPE.lower()
    global_state = {k: v.detach().clone() for k, v in client_model_state_dict.items()}

    if attack == "gaussian":
        out = {}
        for key, param in global_state.items():
            p = param.to(device).float()
            scale = p.std().clamp(min=1e-3) * ATTACK_SCALE
            out[key] = (p + torch.randn_like(p) * scale).detach().cpu()
        return out

    if attack == "scale":
        return {k: (v.to(device).float() * ATTACK_SCALE).detach().cpu() for k, v in global_state.items()}

    # sign_flip / label_flip: one local epoch on the real model, then poison
    from shared.models import make_model

    model = make_model("wide").to(device)
    model.load_state_dict(global_state)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True)
    loader = DataLoader(local_dataset, batch_size=50, shuffle=True)
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if attack == "label_flip":
            yb = 9 - yb
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
    trained = model.state_dict()

    if attack == "label_flip":
        return {k: v.detach().cpu() for k, v in trained.items()}

    if attack == "sign_flip":
        # reverse the update direction: 2*global - trained
        return {
            k: (2 * global_state[k].to(device) - trained[k].to(device)).detach().cpu()
            for k in global_state
        }

    raise ValueError(f"Unknown ATTACK_TYPE={ATTACK_TYPE!r}")
