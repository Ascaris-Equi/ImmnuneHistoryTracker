# train_cross.py
import argparse, os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PairDataset
from common.aa import AATokenizer
from common.utils import set_seed, ensure_dir
from models.cross_encoder import CrossAttnGPT
import torch.optim as optim
import torch.nn.functional as F

def train(args):
    set_seed(args.seed)
    tok = AATokenizer()
    model = CrossAttnGPT(vocab_size=tok.vocab_size, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, dim_ff=args.dim_ff, dropout=args.dropout).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    train_ds = PairDataset(args.csv, split="train", add_bos_eos=True, max_len_pep=args.max_len)
    val_ds = PairDataset(args.csv, split="val", add_bos_eos=True, max_len_pep=args.max_len)
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
            out = model.lm_forward(batch)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / max(1, len(train_dl))
        # val: perplexity proxy
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for batch in val_dl:
                for k in batch:
                    batch[k] = batch[k].to(args.device)
                out = model.lm_forward(batch)
                vloss += out["loss"].item()
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
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train(args)