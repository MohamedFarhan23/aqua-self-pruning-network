"""Does each criterion actually predict the loss increase from removing a connection?

    uv run python scripts/criterion_check.py

Trains the dense network for a few epochs while collecting the pruner's running
statistics (EMA of g and of the per-example Fisher). Then, for a random sample
of weights, it zeroes each one individually and measures the true change in
training loss on a fixed subset. Each criterion's prediction is compared with
that ground truth twice:

* with the running (EMA) statistics the pruner actually uses, and
* with statistics computed at the current weights: on the ground-truth subset
  itself ("exact", in-sample) and on disjoint calibration batches of 1k-16k.

The gap between the two separates estimation noise from approximation error.

Ground truth is measured twice: on the training subset the exact statistics
come from (in-sample: what the Taylor expansion literally approximates), and
on the held-out validation split (what pruning should actually preserve).

Reported per criterion:
* Spearman rank correlation between predicted and true dL;
* regret: the true dL summed over the criterion's bottom half of the sample,
  minus the same sum for the true bottom half. This is the damage done by
  pruning the wrong half, ignoring interactions. 0 is optimal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aqua.data import load_dataset
from aqua.engine import Tensor, cross_entropy, no_grad
from aqua.nn import MLP
from aqua.optim import Adam
from aqua.prune import CRITERIA, CubicSchedule, Pruner
from aqua.prune.pruner import batch_stats
from aqua.train import train

ROOT = Path(__file__).resolve().parents[1]


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def regret(pred: np.ndarray, true: np.ndarray, frac: float = 0.5) -> float:
    k = int(len(true) * frac)
    return float(true[np.argsort(pred, kind="stable")[:k]].sum() - np.sort(true)[:k].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-layer", type=int, default=600)
    parser.add_argument("--subset", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_dataset("mnist", seed=args.seed)
    model = MLP([data.num_features, 300, 100, data.num_classes], seed=args.seed, dtype=np.float64)
    steps = args.epochs * int(np.ceil(len(data.train) / 128))
    # A schedule that never fires: the pruner only collects statistics.
    pruner = Pruner(model, CubicSchedule(0.5, start_step=10**9, end_step=10**9), criterion="taylor2")
    train(model, Adam(model.parameters(), lr=1e-3), data, args.epochs, seed=args.seed, on_step=pruner.on_step)
    ema_stats = pruner.stats

    x, y = data.train.x[:args.subset].astype(np.float64), data.train.y[:args.subset]
    xv, yv = data.val.x.astype(np.float64), data.val.y

    def losses() -> np.ndarray:
        with no_grad():
            return np.array([cross_entropy(model(Tensor(x)), y).item(),
                             cross_entropy(model(Tensor(xv)), yv).item()])

    # Statistics at the current weights: on the ground-truth subset itself
    # ("exact", in-sample) and on disjoint calibration batches of several sizes.
    variants = {"exact": batch_stats(model, x, y)}
    pool_x, pool_y = data.train.x[args.subset:].astype(np.float64), data.train.y[args.subset:]
    for size in (1024, 4096, 16384):
        variants[f"calib{size // 1024}k"] = batch_stats(model, pool_x[:size], pool_y[:size])

    base = losses()
    rng = np.random.default_rng(args.seed)
    rows = []  # (layer, position)
    true_dl = []  # (train-subset dL, validation dL)
    for li, p in enumerate(model.prunable()):
        for idx in rng.choice(p.size, size=min(args.samples_per_layer, p.size), replace=False):
            pos = np.unravel_index(idx, p.shape)
            saved = p.data[pos]
            p.data[pos] = 0.0
            rows.append((li, pos))
            true_dl.append(losses() - base)
            p.data[pos] = saved
    true_dl = np.array(true_dl)

    report = {"config": vars(args), "base_loss": {"train_subset": base[0], "val": base[1]},
              "num_weights": len(rows), "criteria": {}}
    for j, split in enumerate(("train_subset", "val")):
        print(f"[{split}] base loss {base[j]:.4f}; true dL median {np.median(true_dl[:, j]):.2e}, "
              f"{np.mean(true_dl[:, j] < 0):.1%} negative")
    print(f"{'criterion':<10} {'stats':<9} {'rho train':>10} {'rho val':>8} {'regret train':>13} {'regret val':>11}")
    for name, fn in CRITERIA.items():
        for label, stats in (("ema", ema_stats), *variants.items()):
            if name == "magnitude" and label != "ema":
                continue
            scores = [fn(p.data, stats[li]) for li, p in enumerate(model.prunable())]
            pred = np.array([scores[li][pos] for li, pos in rows])
            entry = {}
            for j, split in enumerate(("train_subset", "val")):
                entry[split] = {"spearman": spearman(pred, true_dl[:, j]), "regret": regret(pred, true_dl[:, j])}
            report["criteria"][f"{name}/{label}"] = entry
            t, v = entry["train_subset"], entry["val"]
            print(f"{name:<10} {label:<9} {t['spearman']:>10.3f} {v['spearman']:>8.3f} "
                  f"{t['regret']:>13.2e} {v['regret']:>11.2e}")

    out = ROOT / "results" / "part3" / "criterion_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
