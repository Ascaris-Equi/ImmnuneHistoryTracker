# train_two_tower.py
import argparse, os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PairDataset
from common.aa import AATokenizer
from common.utils import set_seed, ensure_dir
from models.two_tower import TwoTower
import torch.optim as optim

def train(args):
    set_seed(args.seed)
    tok = AATokenizer()
    model = TwoTower(vocab_size=tok.vocab_size, d_model=args.d_model).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    train_ds = PairDataset(args.csv, split="train", add_bos_eos=False)
    val_ds = PairDataset(args.csv, split="val", add_bos_eos=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=val_ds.collate)

    best = 1e9
    ensure_dir(os.path.dirname(args.out))
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}"):
            for k in batch:
                batch[k] = batch[k].to(args.device)
            loss, _ = model(batch, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / max(1, len(train_dl))
        # val with InfoNCE on matched pairs
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for batch in val_dl:
                for k in batch:
                    batch[k] = batch[k].to(args.device)
                loss, _ = model(batch, temperature=args.temperature)
                vloss += loss.item()
            vavg = vloss / max(1, len(val_dl))
        print(f"Epoch {epoch} train {avg:.4f} val {vavg:.4f}")
        if vavg < best:
            best = vavg
            torch.save(model.state_dict(), args.out)
            print(f"Saved best to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train(args)