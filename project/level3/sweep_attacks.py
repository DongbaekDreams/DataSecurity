'''Run Level 3 against every proposal attack and print a summary table.

  python level3/sweep_attacks.py
'''

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
ATTACKS = ("sign_flip", "gaussian", "scale", "label_flip")


def run_attack(attack):
    ckpt = Path(__file__).resolve().parent / "checkpoints" / f"robust_{attack}.pth"
    cmd = [
        sys.executable, "level3/train.py",
        "--attack", attack,
        "--detect", "combined",
        "--aggregate", "trimmed_mean",
        "--seed", "67",
        "--checkpoint", str(ckpt),
    ]
    print(f"\n===== {attack} =====", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out)
    if proc.returncode != 0:
        return attack, None, None, f"FAILED({proc.returncode})"
    acc_m = re.findall(r"Final test acc=([0-9.]+)%", out)
    det_m = re.findall(r"caught in rounds (\[[^\]]+\]|NONE)", out)
    acc = float(acc_m[-1]) if acc_m else None
    caught = det_m[-1] if det_m else "?"
    return attack, acc, caught, "ok"


def main():
    rows = []
    for attack in ATTACKS:
        rows.append(run_attack(attack))

    print("\n===== Attack sweep summary (detect=combined, aggregate=trimmed_mean) =====")
    print(f"{'attack':<12} {'test_acc':>10} {'caught_rounds':<28} status")
    for attack, acc, caught, status in rows:
        acc_s = f"{acc:.2f}%" if acc is not None else "—"
        print(f"{attack:<12} {acc_s:>10} {str(caught):<28} {status}")


if __name__ == "__main__":
    main()
