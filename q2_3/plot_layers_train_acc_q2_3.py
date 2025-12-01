import os
import re
import matplotlib.pyplot as plt
from typing import Dict, List

RESULTS_FILE = "table_results_q2_3.txt"
OUTPUT_FIG = "q2_3_train_acc_vs_layers.png"


def parse_q2_3_results(path: str) -> List[dict]:
    """Parse the Q2.3 results table and return list of dict rows.

    Expected columns per data row:
    Hidden  Layers  LR  Dropout  L2  Train Acc  Val Acc  Test Acc
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if (not line or line.startswith("=") or line.startswith("-")
                    or line.startswith("Hidden") or line.startswith("SUMMARY")):
                continue
            # Lines should start with an integer hidden size
            if not re.match(r"^\d+\s+", line):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 8:
                # Not enough columns; skip
                continue
            try:
                hidden = int(parts[0])
                layers = int(parts[1])
                lr = float(parts[2])
                dropout = float(parts[3])
                l2 = float(parts[4])
                train_acc = float(parts[5])
                val_acc = float(parts[6])
                test_acc = float(parts[7])
            except ValueError:
                # Malformed numeric value; skip
                continue
            rows.append({
                "hidden": hidden,
                "layers": layers,
                "lr": lr,
                "dropout": dropout,
                "l2": l2,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
            })
    return rows


essential_layers = [1, 3, 5, 7, 9]

def train_acc_by_layers(rows: List[dict], hidden_filter: int = 32) -> Dict[int, float]:
    """Return mapping from layers -> training accuracy for a given hidden size (default 32).
    If multiple rows exist per layer, take the maximum training accuracy found.
    """
    best: Dict[int, float] = {}
    for r in rows:
        if r.get("hidden") != hidden_filter:
            continue
        L = r["layers"]
        acc = r["train_acc"]
        if L not in best or acc > best[L]:
            best[L] = acc
    # Keep only the essential L in ascending order if present
    ordered = {L: best[L] for L in sorted(best.keys())}
    return ordered


def plot_train_vs_layers(layer_acc: Dict[int, float], hidden_size: int = 32) -> str:
    layers = list(layer_acc.keys())
    accs = [layer_acc[L] for L in layers]

    plt.figure(figsize=(6, 4))
    plt.plot(layers, accs, marker='o')
    plt.title(f'Training Accuracy vs Number of Layers (hidden={hidden_size})')
    plt.xlabel('Number of Hidden Layers (L)')
    plt.ylabel('Training Accuracy')
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    # Annotate points
    for x, y in zip(layers, accs):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    return OUTPUT_FIG


def main():
    rows = parse_q2_3_results(RESULTS_FILE)
    layer_acc = train_acc_by_layers(rows, hidden_filter=32)
    if not layer_acc:
        raise RuntimeError("No rows found for hidden size 32 in the results file.")
    out = plot_train_vs_layers(layer_acc, hidden_size=32)
    print("Training accuracy per number of layers (hidden=32):")
    for L, acc in layer_acc.items():
        print(f"L={L}: {acc:.4f}")
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
