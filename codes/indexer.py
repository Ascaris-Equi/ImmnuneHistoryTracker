# indexer.py
import argparse, os, json
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
from tqdm import tqdm
from common.utils import read_fasta, sliding_window_peptides, ensure_dir
from common.aa import AATokenizer
from common.trie import Trie
from models.two_tower import TwoTower

def build_library(fasta_dir: str, lengths: List[int]) -> Tuple[List[str], Dict[str, List[int]], Trie]:
    all_peps: List[str] = []
    pep_to_idx: Dict[str, int] = {}
    pathogen_to_indices: Dict[str, List[int]] = {}
    trie = Trie()
    for fname in os.listdir(fasta_dir):
        if not fname.lower().endswith((".fa", ".fasta", ".faa")):
            continue
        pathogen = os.path.splitext(fname)[0]
        seqs = read_fasta(os.path.join(fasta_dir, fname))
        peps_set: Set[str] = set()
        for sid, seq in seqs.items():
            peps_set |= sliding_window_peptides(seq, lengths)
        indices = []
        for p in peps_set:
            if p not in pep_to_idx:
                pep_to_idx[p] = len(all_peps)
                all_peps.append(p)
                trie.insert(p)
            indices.append(pep_to_idx[p])
        pathogen_to_indices[pathogen] = sorted(list(set(indices)))
    return all_peps, pathogen_to_indices, trie

def embed_peptides(peptides: List[str], model_ckpt: str, d_model: int, device: str, batch_size: int = 512) -> np.ndarray:
    tok = AATokenizer()
    model = TwoTower(vocab_size=tok.vocab_size, d_model=d_model).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()
    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(peptides), batch_size), desc="Embedding peptides"):
            batch = peptides[i:i+batch_size]
            ids = [tok.encode_seq(p) for p in batch]
            lens = [len(x) for x in ids]
            maxl = max(lens)
            pad = tok.pad_id
            ids_t = torch.full((len(batch), maxl), pad, dtype=torch.long, device=device)
            attn = torch.zeros_like(ids_t, dtype=torch.bool)
            for r, s in enumerate(ids):
                ids_t[r, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                attn[r, :len(s)] = True
            _, p = model.enc_pep(ids_t, attn)
            p = model.proj_p(p)
            p = torch.nn.functional.normalize(p, dim=-1)
            outs.append(p.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta_dir", required=True)
    ap.add_argument("--lengths", type=str, default="9,10,11,12,13,14,15")
    ap.add_argument("--two_tower_ckpt", required=True)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    ensure_dir(args.out_dir)

    peps, pathogen_to_indices, trie = build_library(args.fasta_dir, lengths)
    np.save(os.path.join(args.out_dir, "peptides.npy"), np.array(peps, dtype=object))
    with open(os.path.join(args.out_dir, "pathogen_to_indices.json"), "w") as f:
        json.dump({k: v for k, v in pathogen_to_indices.items()}, f)
    # save trie as list of peptides (rebuild at load time)
    with open(os.path.join(args.out_dir, "trie_peptides.txt"), "w") as f:
        for p in peps:
            f.write(p + "\n")
    # embeddings
    emb = embed_peptides(peps, args.two_tower_ckpt, args.d_model, args.device)
    np.save(os.path.join(args.out_dir, "embeddings.npy"), emb.astype(np.float32))
    print(f"Library built with {len(peps)} peptides")

if __name__ == "__main__":
    main()