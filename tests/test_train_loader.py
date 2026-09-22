"""Guard against the persistent-worker bug that froze the training data.

DataLoader workers each hold a pickled copy of the dataset made when they
start. With `persistent_workers=True` they never see `train_ds.set_epoch()`,
so every epoch regenerated the IDENTICAL mixtures - verified 22 Sep 2026 with
a batch checksum that was the same at epochs 0, 1 and 2, and different each
epoch once workers were re-created. Every run before that date trained on one
fixed set of `epoch_size` mixtures.

The failure is silent (training runs, loss falls, validation plateaus), so it
is pinned here at the source level. Parsed with `ast`, so it needs no PyTorch.
"""
from __future__ import annotations

import ast
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "src" / "train.py"


def _dataloader_calls():
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DataLoader":
            yield node


def test_train_loader_does_not_use_persistent_workers():
    calls = [c for c in _dataloader_calls()
             if c.args and getattr(c.args[0], "id", None) == "train_ds"]
    assert len(calls) == 1, "expected exactly one DataLoader(train_ds, ...) in src/train.py"
    kw = {k.arg: k.value for k in calls[0].keywords if k.arg}
    assert "persistent_workers" in kw, (
        "train DataLoader must set persistent_workers=False explicitly - "
        "persistent workers never see set_epoch(), freezing the training data")
    v = kw["persistent_workers"]
    assert isinstance(v, ast.Constant) and v.value is False, (
        "train DataLoader must use persistent_workers=False")


def test_shared_kwargs_do_not_reintroduce_persistence():
    """`common` is splatted into both loaders; it must not carry the flag."""
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "common" for t in node.targets)
                and isinstance(node.value, ast.Call)):
            keys = {k.arg for k in node.value.keywords}
            assert "persistent_workers" not in keys
