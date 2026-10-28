"""Train the LSTM + head on frozen MobileNetV3-Small features. Splits by SOURCE VIDEO (no leakage)."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import CnnLstm, clip_to_tensor, save_checkpoint  # noqa: E402


class ClipDataset(Dataset):
    def __init__(self, root, df, train=False):
        self.root, self.df, self.train = Path(root), df.reset_index(drop=True), train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = clip_to_tensor(np.load(self.root / r.clip_path))
        if self.train and random.random() < 0.5:
            x = x.flip(-1)  # horizontal flip augmentation
        return x, torch.tensor(float(r.label))


def split_sources(df, val_frac, test_frac, seed):
    """Stratified split of source videos (videos containing accident clips vs. not)."""
    rng = random.Random(seed)
    has_pos = df.groupby("source_video").label.max()
    train, val, test = set(), set(), set()
    for group in (sorted(has_pos[has_pos == 1].index), sorted(has_pos[has_pos == 0].index)):
        rng.shuffle(group)
        nt, nv = round(len(group) * test_frac), round(len(group) * val_frac)
        test |= set(group[:nt]); val |= set(group[nt:nt + nv]); train |= set(group[nt + nv:])
    return train, val, test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu()); ys.append(y)
    y, p = torch.cat(ys).numpy().astype(int), (torch.cat(ps).numpy() >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
            "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y, p, labels=[0, 1]).tolist(),
            "n_clips": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="clips.csv from prepare_clips.py")
    ap.add_argument("--out", default="models/accident_cnn_lstm.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    root = Path(a.clips).parent
    df = pd.read_csv(a.clips)
    tr_s, va_s, te_s = split_sources(df, a.val_frac, a.test_frac, a.seed)
    parts = {n: df[df.source_video.isin(s)] for n, s in (("train", tr_s), ("val", va_s), ("test", te_s))}
    for n, d in parts.items():
        print(f"{n}: {d.source_video.nunique()} videos, {len(d)} clips, {int(d.label.sum())} accident clips")
        if d.empty or d.label.nunique() < 2:
            print(f"WARNING: '{n}' split is empty or has a single class; metrics will be unreliable.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mk = lambda d, tr: DataLoader(ClipDataset(root, d, tr), batch_size=a.batch, shuffle=tr, num_workers=2)
    loaders = {n: mk(d, n == "train") for n, d in parts.items()}

    model = CnnLstm(pretrained=True).to(device)
    pos = max(int(parts["train"].label.sum()), 1)
    neg = max(len(parts["train"]) - pos, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    best_f1, best_val = -1.0, {}
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            total += loss.item() * len(y)
        val = evaluate(model, loaders["val"], device)
        print(f"epoch {ep}: loss={total / len(parts['train']):.4f} val_f1={val['f1']:.3f} val_acc={val['accuracy']:.3f}")
        if val["f1"] > best_f1:
            best_f1, best_val = val["f1"], val
            save_checkpoint(model, Path(a.out), {"val": val})

    best = torch.load(a.out, map_location=device)
    model.load_state_dict(best["state_dict"])
    test = evaluate(model, loaders["test"], device)
    best["metrics"] = json.loads(json.dumps({"val": best_val, "test": test}))
    torch.save(best, a.out)
    print("TEST METRICS:", json.dumps(test, indent=2))
    print(f"Checkpoint saved: {a.out}")


if __name__ == "__main__":
    main()
