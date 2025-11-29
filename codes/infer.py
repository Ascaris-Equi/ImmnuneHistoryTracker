# infer.py
import argparse, os, json
from typing import List, Dict
import numpy as np
import torch
from common.aa import AATokenizer
from common.trie import Trie
from common.utils import chunked_topk
from models.two_tower import TwoTower
from models.cross_encoder import CrossAttnGPT

def load_library(lib_dir: str):
    peptides = np.load(os.path.join(lib_dir, "peptides.npy"), allow_pickle=True).tolist()
    emb = np.load(os.path.join(lib_dir, "embeddings.npy"))
    with open(os.path.join(lib_dir, "pathogen_to_indices.json")) as f:
        pathogen_to_indices = json.load(f)
    trie = Trie()
    with open(os.path.join(lib_dir, "trie_peptides.txt")) as f:
        for line in f:
            trie.insert(line.strip())
    return peptides, emb, pathogen_to_indices, trie

@torch.no_grad()
def retrieve_candidates(cdr3: str, mhc: str, two_tower_ckpt: str, d_model: int, lib_emb: np.ndarray, topM: int, device: str):
    tok = AATokenizer()
    model = TwoTower(vocab_size=tok.vocab_size, d_model=d_model).to(device)
    model.load_state_dict(torch.load(two_tower_ckpt, map_location=device))
    model.eval()
    # encode query
    ids_c = torch.tensor([tok.encode_seq(cdr3)], dtype=torch.long, device=device)
    attn_c = torch.ones_like(ids_c, dtype=torch.bool)
    ids_m = torch.tensor([tok.encode_seq(mhc)], dtype=torch.long, device=device)
    attn_m = torch.ones_like(ids_m, dtype=torch.bool)
    q = model.encode_query(ids_c, attn_c, ids_m, attn_m)  # (1,D)
    q = q.detach().cpu().numpy()  # (1,D)
    scores = (q @ lib_emb.T)  # cosine if embeddings are normalized
    idx, scr = chunked_topk(scores, k=topM, chunk=100000)
    return idx[0], scr[0]

@torch.no_grad()
def rerank(cdr3: str, mhc: str, peptides: List[str], cross_ckpt: str, d_model: int, device: str) -> np.ndarray:
    tok = AATokenizer()
    model = CrossAttnGPT(vocab_size=tok.vocab_size, d_model=d_model).to(device)
    model.load_state_dict(torch.load(cross_ckpt, map_location=device))
    model.eval()
    # batch
    c_ids = torch.tensor([tok.encode_seq(cdr3)], dtype=torch.long, device=device)
    c_attn = torch.ones_like(c_ids, dtype=torch.bool)
    m_ids = torch.tensor([tok.encode_seq(mhc)], dtype=torch.long, device=device)
    m_attn = torch.ones_like(m_ids, dtype=torch.bool)
    # prepare pep with BOS/EOS
    pep_ids = []
    pep_attn = []
    for p in peptides:
        ids = tok.encode_seq(p, add_bos=True, add_eos=True)
        pep_ids.append(ids)
    maxl = max(len(x) for x in pep_ids)
    pad = tok.pad_id
    P = len(peptides)
    pep_t = torch.full((P, maxl), pad, dtype=torch.long, device=device)
    pep_a = torch.zeros_like(pep_t, dtype=torch.bool)
    for i, s in enumerate(pep_ids):
        pep_t[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        pep_a[i, :len(s)] = True
    # expand cdr3/mhc to P
    c_ids = c_ids.expand(P, -1)
    c_attn = c_attn.expand(P, -1)
    m_ids = m_ids.expand(P, -1)
    m_attn = m_attn.expand(P, -1)
    scores = model.seq_log_prob(c_ids, c_attn, m_ids, m_attn, pep_t, pep_a).detach().cpu().numpy()
    return scores

def constrained_generate(cdr3: str, mhc: str, cross_ckpt: str, d_model: int, trie: Trie,
                         num_samples: int, min_len: int, max_len: int, top_p: float, top_k: int, device: str) -> List[str]:
    tok = AATokenizer()
    model = CrossAttnGPT(vocab_size=tok.vocab_size, d_model=d_model).to(device)
    model.load_state_dict(torch.load(cross_ckpt, map_location=device))
    model.eval()
    c_ids = torch.tensor([tok.encode_seq(cdr3)], dtype=torch.long, device=device)
    c_attn = torch.ones_like(c_ids, dtype=torch.bool)
    m_ids = torch.tensor([tok.encode_seq(mhc)], dtype=torch.long, device=device)
    m_attn = torch.ones_like(m_ids, dtype=torch.bool)
    seqs = model.generate(c_ids, c_attn, m_ids, m_attn, min_len=min_len, max_len=max_len,
                          num_samples=num_samples, top_p=top_p, top_k=top_k, constrained_trie=trie)
    outs = []
    for ids in seqs:
        outs.append(tok.decode_seq(ids, skip_special=True))
    # unique
    outs = list(dict.fromkeys(outs))
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib_dir", required=True)
    ap.add_argument("--two_tower_ckpt", required=True)
    ap.add_argument("--cross_ckpt", required=True)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--cdr3", required=True)
    ap.add_argument("--mhc", required=True)
    ap.add_argument("--topM", type=int, default=1000)
    ap.add_argument("--rerank_top", type=int, default=50)
    ap.add_argument("--gen_num", type=int, default=50)
    ap.add_argument("--min_len", type=int, default=9)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    peptides, emb, path2idx, trie = load_library(args.lib_dir)
    idx, scr = retrieve_candidates(args.cdr3, args.mhc, args.two_tower_ckpt, args.d_model, emb, args.topM, args.device)
    cand_peps = [peptides[i] for i in idx]
    # re-rank
    scores = rerank(args.cdr3, args.mhc, cand_peps, args.cross_ckpt, args.d_model, args.device)
    order = np.argsort(-scores)[:args.rerank_top]
    print("Top reranked candidates:")
    for r in order:
        print(f"{cand_peps[r]}\tLL={scores[r]:.4f}")
    # constrained generation
    gens = constrained_generate(args.cdr3, args.mhc, args.cross_ckpt, args.d_model, trie, args.gen_num, args.min_len, args.max_len, args.top_p, args.top_k, args.device)
    print("\nConstrained generation (unique):")
    for g in gens[:50]:
        print(g)

if __name__ == "__main__":
    main()