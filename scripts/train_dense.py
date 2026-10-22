"""Part 2: train a dense MLP with Adam and record its learning curve.

    uv run python scripts/train_dense.py            # MNIST, LeNet-300-100, 20 epochs
    uv run python scripts/train_dense.py --dataset fashion_mnist
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from aqua.data import load_dataset
from aqua.nn import MLP
from aqua.optim import Adam
from aqua.plotting import SERIES, plt, style
from aqua.train import constant, cosine, evaluate, train

ROOT = Path(__file__).resolve().parents[1]


def plot(history: list[dict], initial: dict, out: Path, title: str) -> None:
    style()
    epochs = [0] + [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, key, label in ((ax_loss, "loss", "Cross-entropy"), (ax_acc, "acc", "Accuracy")):
        train_vals = [initial[f"train_{key}"]] + [h[f"train_{key}"] for h in history]
        val_vals = [initial[f"val_{key}"]] + [h[f"val_{key}"] for h in history]
        ax.plot(epochs[1:], train_vals[1:], color=SERIES[0], label="train (running)")
        # The untrained network (epoch 0) is shown on the loss panel only: its
        # chance-level accuracy would squash the accuracy axis.
        first = 0 if key == "loss" else 1
        ax.plot(epochs[first:], val_vals[first:], color=SERIES[1], label="validation")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label, loc="left")
        ax.legend()
    ax_loss.set_yscale("log")
    ax_acc.set_ylim(0.9 if history[-1]["val_acc"] > 0.95 else 0.7, 1.0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color="#52514e")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "spirals"])
    parser.add_argument("--hidden", type=int, nargs="+", default=[300, 100])
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh", "gelu"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or ROOT / "results" / "part2" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset, seed=args.seed)
    sizes = [data.num_features, *args.hidden, data.num_classes]
    model = MLP(sizes, activation=args.activation, seed=args.seed)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Epoch 0: the untrained network. With unit-order logit variance s^2 the
    # expected loss is about ln(C) + s^2 (C - 1) / (2C): uninformed, not confidently wrong.
    train_loss0, train_acc0 = evaluate(model, data.train)
    val_loss0, val_acc0 = evaluate(model, data.val)
    initial = {"train_loss": train_loss0, "train_acc": train_acc0, "val_loss": val_loss0, "val_acc": val_acc0,
               "uninformed_loss": math.log(data.num_classes)}
    print(f"epoch  0  val loss {val_loss0:.4f} (ln {data.num_classes} = {math.log(data.num_classes):.4f})")

    def log(s):
        print(f"epoch {s.epoch:2d}  train loss {s.train_loss:.4f} acc {s.train_acc:.4f}  "
              f"val loss {s.val_loss:.4f} acc {s.val_acc:.4f}  max|g| {s.max_grad_norm:.2f}  {s.seconds:.1f}s")

    steps = args.epochs * math.ceil(len(data.train) / args.batch_size)
    schedule = cosine(args.lr, steps) if args.schedule == "cosine" else constant(args.lr)
    history = [s.as_dict() for s in train(model, optimizer, data, args.epochs, args.batch_size,
                                          seed=args.seed, lr_schedule=schedule, on_epoch=log)]
    test_loss, test_acc = evaluate(model, data.test)
    print(f"test loss {test_loss:.4f} acc {test_acc:.4f}")

    result = {
        "config": {**{k: v for k, v in vars(args).items() if k != "out"}, "sizes": sizes,
                   "num_parameters": model.num_parameters()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "initial": initial,
        "history": history,
        "test": {"loss": test_loss, "acc": test_acc},
    }
    (out / "history.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(history, initial, out / "learning_curve.png",
         f"{args.dataset} · MLP {'-'.join(map(str, sizes))} · Adam lr={args.lr} ({args.schedule}) · seed {args.seed} · "
         f"test acc {test_acc:.2%}")
    print(f"wrote {out}/history.json and learning_curve.png")


if __name__ == "__main__":
    main()
