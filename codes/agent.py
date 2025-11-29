# agent.py
import argparse, os, json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from common.aa import AATokenizer
from common.trie import Trie
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
    # map peptide idx -> list of pathogens
    idx_to_pathogens: Dict[int, List[str]] = {}
    for pat, idxs in pathogen_to_indices.items():
        for i in idxs:
            idx_to_pathogens.setdefault(int(i), []).append(pat)
    return peptides, emb, pathogen_to_indices, idx_to_pathogens, trie

@torch.no_grad()
def get_candidates_for_tcr(cdr3: str, mhc: str, lib_emb: np.ndarray, two_tower: TwoTower, cross: CrossAttnGPT,
                           peptides: List[str], topM: int = 2000, rerank_top: int = 50, device: str = "cpu") -> List[Tuple[str, float]]:
    tok = AATokenizer()
    # encode query
    ids_c = torch.tensor([tok.encode_seq(cdr3)], dtype=torch.long, device=device)
    attn_c = torch.ones_like(ids_c, dtype=torch.bool)
    ids_m = torch.tensor([tok.encode_seq(mhc)], dtype=torch.long, device=device)
    attn_m = torch.ones_like(ids_m, dtype=torch.bool)
    q = two_tower.encode_query(ids_c, attn_c, ids_m, attn_m)  # (1,D)
    q = q.detach().cpu().numpy()
    scores = (q @ lib_emb.T)
    # topM
    Ni = lib_emb.shape[0]
    k = min(topM, Ni)
    idx = np.argpartition(-scores[0], kth=k-1)[:k]
    idx = idx[np.argsort(-scores[0, idx])]
    cand_peps = [peptides[i] for i in idx]
    # rerank with cross-encoder avg log prob
    # batch scoring
    tok = AATokenizer()
    c_ids = torch.tensor([tok.encode_seq(cdr3)], dtype=torch.long, device=device)
    c_attn = torch.ones_like(c_ids, dtype=torch.bool)
    m_ids = torch.tensor([tok.encode_seq(mhc)], dtype=torch.long, device=device)
    m_attn = torch.ones_like(m_ids, dtype=torch.bool)
    pep_ids = [tok.encode_seq(p, add_bos=True, add_eos=True) for p in cand_peps]
    maxl = max(len(x) for x in pep_ids)
    P = len(pep_ids)
    pad = tok.pad_id
    pep_t = torch.full((P, maxl), pad, dtype=torch.long, device=device)
    pep_a = torch.zeros_like(pep_t, dtype=torch.bool)
    for i, s in enumerate(pep_ids):
        pep_t[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        pep_a[i, :len(s)] = True
    c_ids = c_ids.expand(P, -1)
    c_attn = c_attn.expand(P, -1)
    m_ids = m_ids.expand(P, -1)
    m_attn = m_attn.expand(P, -1)
    ll = cross.seq_log_prob(c_ids, c_attn, m_ids, m_attn, pep_t, pep_a).detach().cpu().numpy()
    order = np.argsort(-ll)[:rerank_top]
    return [(cand_peps[i], float(ll[i])) for i in order]

def apply_calibration(ll: float, calib: Dict) -> float:
    # Platt scaling: sigmoid(a * ll + b)
    a = calib.get("coef", 1.0)
    b = calib.get("intercept", 0.0)
    x = a * ll + b
    return 1.0 / (1.0 + np.exp(-x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib_dir", required=True)
    ap.add_argument("--repertoire_csv", required=True, help="columns: CDR3,MHC,count")
    ap.add_argument("--two_tower_ckpt", required=True)
    ap.add_argument("--cross_ckpt", required=True)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--topM", type=int, default=2000)
    ap.add_argument("--rerank_top", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5, help="scaling for evidence aggregation")
    ap.add_argument("--calib_json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    peptides, emb, path2idx, idx2path, trie = load_library(args.lib_dir)
    tok = AATokenizer()
    two_tower = TwoTower(vocab_size=tok.vocab_size, d_model=args.d_model).to(args.device)
    two_tower.load_state_dict(torch.load(args.two_tower_ckpt, map_location=args.device))
    two_tower.eval()
    cross = CrossAttnGPT(vocab_size=tok.vocab_size, d_model=args.d_model).to(args.device)
    cross.load_state_dict(torch.load(args.cross_ckpt, map_location=args.device))
    cross.eval()
    calib = None
    if args.calib_json and os.path.exists(args.calib_json):
        with open(args.calib_json) as f:
            calib = json.load(f)

    df = pd.read_csv(args.repertoire_csv)
    # normalize counts to frequencies
    if "count" in df.columns:
        df["w"] = df["count"] / df["count"].sum()
    else:
        df["w"] = 1.0 / len(df)
    pathogen_scores: Dict[str, float] = {}
    pathogen_evidence: Dict[str, List[Dict]] = {}
    for _, row in df.iterrows():
        cdr3 = row["CDR3"]
        mhc = row["MHC"]
        w = float(row["w"])
        cands = get_candidates_for_tcr(cdr3, mhc, emb, two_tower, cross, peptides, topM=args.topM, rerank_top=args.rerank_top, device=args.device)
        for pep, ll in cands:
            p_idx = peptides.index(pep)  # for simplicity; for speed, build dict outside in real runs
            pathogens = idx2path.get(p_idx, [])
            # map ll -> calibrated probability
            prob = apply_calibration(ll, calib) if calib is not None else float(1 / (1 + np.exp(-ll)))
            for pat in pathogens:
                # aggregate: s_j = 1 - prod(1 - alpha * w_i * p_i)
                contrib = args.alpha * w * prob
                pathogen_scores.setdefault(pat, 0.0)
                # convert to log(1 - s) space to avoid underflow
                if "log1m" not in pathogen_evidence:
                    pathogen_evidence["log1m"] = {}
                pathogen_evidence["log1m"].setdefault(pat, 0.0)
                pathogen_evidence.setdefault(pat, [])
                pathogen_evidence[pat].append({"cdr3": cdr3, "mhc": mhc, "peptide": pep, "ll": ll, "prob": prob, "weight": w})
                # We'll aggregate after loop for clarity
    # finalize aggregation
    final_scores: Dict[str, float] = {}
    for pat, evs in pathogen_evidence.items():
        if pat == "log1m": 
            continue
        one_minus = 1.0
        for e in evs:
            contrib = args.alpha * e["weight"] * e["prob"]
            contrib = min(max(contrib, 0.0), 1.0)
            one_minus *= (1.0 - contrib)
        final_scores[pat] = 1.0 - one_minus

    out = {
        "scores": sorted([{ "pathogen": k, "prob": float(v)} for k, v in final_scores.items()], key=lambda x: -x["prob"]),
        "evidence": pathogen_evidence
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()