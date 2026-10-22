"""Dataset loading: MNIST / Fashion-MNIST from the raw IDX files, plus a synthetic task.

No third-party loaders (scikit-learn, torchvision, ...) are used: the IDX
format is a short big-endian header followed by raw uint8 data, and every
download is verified against a pinned SHA-256 so results reproduce exactly.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_SOURCES = {
    "mnist": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        {
            "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
            "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
            "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
            "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        },
    ),
    "fashion_mnist": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        {
            "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
            "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
            "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
            "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
        },
    ),
}


@dataclass
class Split:
    x: np.ndarray  # (N, D) float32
    y: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Dataset:
    name: str
    train: Split
    val: Split
    test: Split
    num_classes: int

    @property
    def num_features(self) -> int:
        return self.train.x.shape[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, filename: str) -> Path:
    base_url, hashes = _SOURCES[name]
    path = DATA_DIR / name / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(base_url + filename, tmp)
        tmp.rename(path)
    digest = _sha256(path)
    if digest != hashes[filename]:
        raise RuntimeError(f"checksum mismatch for {path}: {digest}; delete it and retry")
    return path


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    zero, dtype_code, ndim = raw[0] << 8 | raw[1], raw[2], raw[3]
    if zero != 0 or dtype_code != 0x08:  # 0x08 = unsigned byte
        raise ValueError(f"{path} is not a uint8 IDX file")
    dims = np.frombuffer(raw, dtype=">u4", count=ndim, offset=4).astype(int)
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_image_dataset(name: str = "mnist", val_size: int = 5000, seed: int = 0) -> Dataset:
    """MNIST-style dataset as flattened float32 vectors.

    Pixels are scaled to [0, 1] and then standardised with the *training*
    split's global mean/std, so inputs have roughly zero mean and unit variance
    (the assumption behind the initialisation in ``aqua.nn``). The validation
    split is a fixed random subset of the official training set; the official
    test set is held out and only used for final reporting.
    """
    if name not in _SOURCES:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_SOURCES)}")
    x_train = _read_idx(_fetch(name, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx(_fetch(name, "train-labels-idx1-ubyte.gz"))
    x_test = _read_idx(_fetch(name, "t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx(_fetch(name, "t10k-labels-idx1-ubyte.gz"))

    x_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0

    perm = np.random.default_rng(seed).permutation(len(x_train))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    mean, std = x_train[train_idx].mean(), x_train[train_idx].std()

    def split(x, y):
        return Split(((x - mean) / std).astype(np.float32), y.astype(np.int64))

    return Dataset(
        name=name,
        train=split(x_train[train_idx], y_train[train_idx]),
        val=split(x_train[val_idx], y_train[val_idx]),
        test=split(x_test, y_test),
        num_classes=10,
    )


def make_spirals(n_per_class: int = 300, num_classes: int = 3, noise: float = 0.2, seed: int = 0) -> Dataset:
    """Interleaved 2-D spirals: a small, non-linearly separable task for fast tests."""
    rng = np.random.default_rng(seed)

    def sample(n):
        xs, ys = [], []
        for c in range(num_classes):
            r = np.linspace(0.05, 1.0, n)
            theta = np.linspace(0, 6, n) + c * 2 * np.pi / num_classes + rng.normal(0, noise, n)
            xs.append(np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1))
            ys.append(np.full(n, c))
        x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)
        perm = rng.permutation(len(y))
        return Split(x[perm] * 2.0, y[perm])

    return Dataset("spirals", sample(n_per_class), sample(n_per_class // 3), sample(n_per_class // 3), num_classes)


def load_dataset(name: str, seed: int = 0) -> Dataset:
    if name == "spirals":
        return make_spirals(seed=seed)
    return load_image_dataset(name, seed=seed)
