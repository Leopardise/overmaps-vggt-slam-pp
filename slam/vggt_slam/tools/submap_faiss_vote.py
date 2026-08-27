#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
from pathlib import Path
from collections import defaultdict

def _load_owners_intkeys(path):
    """Load tile_owners.json and normalize keys to int."""
    if not os.path.isfile(path):
        raise SystemExit(f"missing tile_owners.json: {path} (run build_tile_owners.py)")
    raw = json.load(open(path, "r")).get("tile_owners", {})
    owners_int = {}
    for k, v in raw.items():
        try:
            owners_int[int(k)] = list(v)
        except Exception:
            # skip malformed keys
            continue
    return owners_int

def _load_matches(path):
    """Load matches_topk.json and return list of {top_ids, scores} entries."""
    if not os.path.isfile(path):
        raise SystemExit(f"missing matches_topk.json: {path}")
    M = json.load(open(path, "r"))
    matches = M.get("matches", [])
    if not isinstance(matches, list):
        matches = []
    return matches

def main():
    ap = argparse.ArgumentParser("Aggregate FAISS chip→tile matches into submap scores")
    ap.add_argument("--root", required=True, help="e.g., outputs/run")
    ap.add_argument("--submap", required=True, help="registering submap, e.g., sm_00001")
    ap.add_argument("--matches", default="", help="defaults to <root>/submaps/<sm>/matches_topk.json")
    ap.add_argument("--tile-owners", default="", help="defaults to <root>/submap_index/tile_owners.json")
    ap.add_argument("--exclude-self", action="store_true", help="ignore tiles owned by the same submap")
    ap.add_argument("--weighted", action="store_true", help="use FAISS score as weight; otherwise +1 vote per hit")
    args = ap.parse_args()

    root = args.root
    sm   = args.submap
    matches_p = args.matches or os.path.join(root, "submaps", sm, "matches_topk.json")
    owners_p  = args.tile_owners or os.path.join(root, "submap_index", "tile_owners.json")

    # Load inputs
    owners = _load_owners_intkeys(owners_p)               # <-- normalize to int keys
    matches = _load_matches(matches_p)

    vote = defaultdict(float)   # submap_id -> aggregated score
    per_tile = defaultdict(float)

    for m in matches:
        top_ids = m.get("top_ids", []) or []
        scores  = m.get("scores", [])  or []
        # zip safely; ignore any extra tails
        for tid_raw, sc_raw in zip(top_ids, scores):
            try:
                tid = int(tid_raw)
            except Exception:
                continue
            try:
                sc = float(sc_raw)
            except Exception:
                sc = 0.0

            w = sc if args.weighted else 1.0
            claimers = owners.get(tid, [])                 # <-- int lookup
            if not claimers:
                continue
            for sm_owner in claimers:
                if args.exclude_self and sm_owner == sm:
                    continue
                vote[sm_owner] += w
            per_tile[tid] += w

    ranked = sorted(vote.items(), key=lambda kv: kv[1], reverse=True)

    outd = os.path.join(root, "submaps", sm)
    Path(outd).mkdir(parents=True, exist_ok=True)
    outp = os.path.join(outd, "faiss_votes_by_submap.json")
    with open(outp, "w") as f:
        json.dump({
            "registering_submap": sm,
            "exclude_self": bool(args.exclude_self),
            "weighted": bool(args.weighted),
            "scores": {k: vote[k] for k in vote},          # dict-of-scores for robustness
            "ranking": ranked,                             # also keep list for older readers
            "num_chips": len(matches)
        }, f, indent=2)

    best = ranked[0][0] if ranked else None
    if best is None:
        print("[vote] no gallery submap received votes — check owners key types and matches content.")
    print(f"[vote] best gallery submap: {best}  (out → {outp})")

if __name__ == "__main__":
    main()
