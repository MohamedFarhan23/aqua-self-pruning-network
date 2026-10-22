"""Part 3: train a network that prunes itself during training to a hard sparsity budget.

    uv run python scripts/train_prune.py                          # K-FAC OBS + regrowth, 90% sparsity
    uv run python scripts/train_prune.py --criterion magnitude --no-regrow
    uv run python scripts/train_prune.py --sparsity 0.95 --seed 1

The network trains dense for ``--prune-start`` epochs, then a cubic schedule
ramps sparsity from 0 to the target by ``--prune-end`` epochs (re-scoring every
``--interval`` steps), then it fine-tunes with the mask frozen. The learning
rate follows one cosine decay over the whole run.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from aqua.data import load_dataset
from aqua.nn import MLP
from aqua.optim import Adam
from aqua.plotting import SERIES, TEXT_SECONDARY, plt, style
from aqua.prune import CRITERIA, CubicSchedule, Pruner
from aqua.train import cosine, evaluate, train

ROOT = Path(__file__).resolve().parents[1]


def run_name(args) -> str:
    regrow = "_regrow" if args.regrow else ""
    ema = "_ema" if args.stats == "ema" and args.criterion != "magnitude" else ""
    return f"{args.criterion}{regrow}{ema}_s{round(args.sparsity * 100):02d}_seed{args.seed}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "spirals"])
    parser.add_argument("--hidden", type=int, nargs="+", default=[300, 100])
    parser.add_argument("--criterion", default="obs", choices=sorted(CRITERIA))
    parser.add_argument("--obs-damping", type=float, default=None, help="override OBS damping (default in criteria.py)")
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--regrow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regrow-fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--prune-start", type=float, default=1.0, help="epoch at which pruning starts")
    parser.add_argument("--prune-end", type=float, default=12.0, help="epoch at which the target is reached")
    parser.add_argument("--interval", type=int, default=100, help="optimizer steps between pruning events")
    parser.add_argument("--stats", default="batch", choices=["batch", "ema"],
                        help="gradient statistics: fresh calibration batch per event, or running EMA")
    parser.add_argument("--calibration-size", type=int, default=4096)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def plot(result: dict, out: Path) -> None:
    style()
    history, events = result["history"], result["events"]
    steps_per_epoch = result["config"]["steps_per_epoch"]
    fig, (ax_acc, ax_sp) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)

    epochs = [h["epoch"] for h in history]
    ax_acc.plot(epochs, [h["val_acc"] for h in history], color=SERIES[0], label="validation")
    ax_acc.plot(epochs, [h["train_acc"] for h in history], color=SERIES[1], label="train (running)")
    ax_acc.set_ylim(0.9, 1.0)
    ax_acc.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_acc.set_title("Accuracy", loc="left")
    ax_acc.legend(loc="lower right")

    ev_epochs = [e["step"] / steps_per_epoch for e in events]
    ax_sp.plot(ev_epochs, [e["target_sparsity"] for e in events], color=SERIES[0], label="cubic target")
    ax_sp.plot(ev_epochs, [e["sparsity"] for e in events], color=SERIES[1], linestyle="none", marker="o",
               markersize=3, label="achieved")
    revived = [e["revived"] for e in events]
    if any(revived):
        twin = ax_sp.twinx()
        twin.bar(ev_epochs, revived, width=0.08, color=SERIES[2], alpha=0.6, label="revived per event")
        twin.set_ylabel("revived connections", color=TEXT_SECONDARY)
        twin.spines["right"].set_visible(True)
        twin.grid(False)
        twin.legend(loc="center right")
    ax_sp.set_xlim(0, result["config"]["epochs"])
    ax_sp.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_sp.set_title("Sparsity of weight matrices", loc="left")
    ax_sp.legend(loc="lower right")
    for ax in (ax_acc, ax_sp):
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    cfg, final = result["config"], result["final"]
    fig.suptitle(f"{cfg['dataset']} · {cfg['criterion']}{' + regrowth' if cfg['regrow'] else ''} · "
                 f"sparsity {final['sparsity']:.1%} · test acc {final['test_acc']:.2%} · seed {cfg['seed']}",
                 x=0.01, ha="left", fontsize=10, color=TEXT_SECONDARY)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(args) -> dict:
    data = load_dataset(args.dataset, seed=args.seed)
    sizes = [data.num_features, *args.hidden, data.num_classes]
    model = MLP(sizes, seed=args.seed)
    optimizer = Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = math.ceil(len(data.train) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    schedule = CubicSchedule(args.sparsity, start_step=max(1, round(args.prune_start * steps_per_epoch)),
                             end_step=round(args.prune_end * steps_per_epoch), interval=args.interval)
    calibration = (data.train.x, data.train.y) if args.stats == "batch" else None
    kwargs = {"damping": args.obs_damping} if args.criterion == "obs" and args.obs_damping is not None else {}
    pruner = Pruner(model, schedule, criterion=args.criterion, criterion_kwargs=kwargs, ema=args.ema, regrow=args.regrow,
                    regrow_fraction=args.regrow_fraction, calibration=calibration,
                    calibration_size=args.calibration_size, seed=args.seed)

    def log(s):
        if not args.quiet:
            print(f"epoch {s.epoch:2d}  sparsity {pruner.sparsity:6.2%}  train loss {s.train_loss:.4f}  "
                  f"val loss {s.val_loss:.4f} acc {s.val_acc:.4f}  max|g| {s.max_grad_norm:.2f}  {s.seconds:.1f}s")

    history = train(model, optimizer, data, args.epochs, args.batch_size, seed=args.seed,
                    lr_schedule=cosine(args.lr, total_steps), on_step=pruner.on_step, on_epoch=log)
    test_loss, test_acc = evaluate(model, data.test)

    # Honest accounting: count exact zeros in the weight matrices themselves.
    weights = model.prunable()
    zeros = sum(int(np.count_nonzero(p.data == 0.0)) for p in weights)
    total_params = model.num_parameters()
    final = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "val_acc": history[-1].val_acc,
        "sparsity": pruner.sparsity,
        "active_weights": pruner.num_active,
        "total_weights": pruner.total,
        "exact_zero_weights": zeros,
        "active_parameters_incl_bias": total_params - (pruner.total - pruner.num_active),
        "total_parameters_incl_bias": total_params,
        "layer_density": pruner.layer_density(),
        "total_revived": sum(e.revived for e in pruner.events),
    }
    config = {**{k: v for k, v in vars(args).items() if k not in ("out", "quiet")}, "sizes": sizes,
              "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
              "prune_start_step": schedule.start_step, "prune_end_step": schedule.end_step}
    return {
        "config": config,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "history": [h.as_dict() for h in history],
        "events": [vars(e) for e in pruner.events],
        "final": final,
    }


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = args.out or ROOT / "results" / "part3" / args.dataset / run_name(args)
    out.mkdir(parents=True, exist_ok=True)
    result = run(args)
    (out / "history.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, out / "pruning.png")
    f = result["final"]
    print(f"{run_name(args)}: sparsity {f['sparsity']:.2%} ({f['active_weights']}/{f['total_weights']} weights active, "
          f"{f['exact_zero_weights']} exact zeros)  test acc {f['test_acc']:.4f}  -> {out}")
    return result


if __name__ == "__main__":
    main()
