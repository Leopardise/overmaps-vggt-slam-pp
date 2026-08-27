#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update (or create) a global CSV of AnyLoc-derived loop votes.

For each query submap:
- Rank-weight top-K patch matches per chip (weights K..1).
- Aggregate to submap scores via tile_owners.json (exclude self).
- Keep the top-N submaps overall (N is --topn).
- Insert or overwrite a single row in $ROOT/anyloc_io/loop_votes.csv

Default CSV schema (for --topn=N):
query_submap, match1_submap, match1_score, ..., matchN_submap, matchN_score, updated_at_iso

Assumptions:
- $ROOT/submap_index/tile_owners.json exists:
   {"tile_owners": {"300": ["sm_00002","sm_00006"], ...}}
- AnyLoc matches.csv has columns: query_path, db_path, score

Usage:
python vggt_slam/tools/update_loop_votes_csv.py \
  --root "$ROOT" \
  --submap sm_00001 \
  --matches "$IO/matches.csv" \
  --per-chip-topk 5 \
  --topn 7
"""
import os, re, csv, json, argparse, datetime
from collections import defaultdict
from typing import Dict, List, Tuple

def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _tile_id_from_path(path: str):
    base = os.path.basename(path)
    m = re.search(r'tile_(\d+)\.(?:png|jpg|jpeg|npy)$', base, re.IGNORECASE)
    return int(m.group(1)) if m else None

def _load_tile_owners(root: str) -> Dict[int, List[str]]:
    j = os.path.join(root, "submap_index", "tile_owners.json")
    with open(j, "r") as f:
        data = json.load(f)
    own = data["tile_owners"]
    return {int(k): v for k, v in own.items()}  # tid -> [sm_xxxxx, ...]

def _read_matches(matches_csv: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns groups: query_path -> [(db_path, score), ...]
    """
    groups = defaultdict(list)
    with open(matches_csv, "r") as f:
        rd = csv.DictReader(f)
        if not {"query_path", "db_path", "score"}.issubset(set(rd.fieldnames or [])):
            raise SystemExit("matches.csv must contain columns: query_path, db_path, score")
        for row in rd:
            try:
                sc = float(row["score"])
            except:
                sc = 0.0
            groups[row["query_path"]].append((row["db_path"], sc))
    return groups

def _rank_weight_votes(groups, owners, self_submap: str, per_chip_topk: int,
                       topn: int) -> List[Tuple[str, float]]:
    """
    Rank-weight per-chip: topK entries get weights K..1.
    Aggregate to submap scores via tile owners, excluding self. Return top-N pairs.
    """
    submap_scores = defaultdict(float)

    for qp, lst in groups.items():
        lst.sort(key=lambda x: -x[1])
        top = lst[:per_chip_topk]
        K = len(top)
        for r, (dbp, _) in enumerate(top):
            w = float(K - r)  # K..1
            tid = _tile_id_from_path(dbp)
            if tid is None:
                continue
            for sm in owners.get(tid, []):
                if sm != self_submap:
                    submap_scores[sm] += w

    ranked = sorted(submap_scores.items(), key=lambda kv: -kv[1])[:max(0, topn)]
    # pad to exactly topn entries (with empty sid/0 score) for stable CSV shape
    while len(ranked) < topn:
        ranked.append(("", 0.0))
    return ranked  # [ (sm_id, score), ... length=topn ]

def _read_existing(csv_path: str) -> Dict[str, dict]:
    """
    Read existing CSV into dict keyed by query_submap.
    We store entire row dicts; writer will handle missing columns gracefully.
    """
    rows = {}
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            q = r.get("query_submap")
            if q:
                rows[q] = r
    return rows

def _dynamic_fieldnames(topn: int) -> List[str]:
    cols = ["query_submap"]
    for i in range(1, topn+1):
        cols += [f"match{i}_submap", f"match{i}_score"]
    cols += ["updated_at_iso"]
    return cols

def _write_all(csv_path: str, rows_dict: dict, topn: int):
    _ensure_dir(csv_path)
    fieldnames = _dynamic_fieldnames(topn)
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        # Write rows in key order for stability
        for k in sorted(rows_dict.keys()):
            # Ensure all needed columns exist; missing become blanks
            base = {fn: "" for fn in fieldnames}
            base.update(rows_dict[k])
            wr.writerow(base)

def main():
    ap = argparse.ArgumentParser("Update loop_votes.csv (top-N configurable)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--submap", required=True)            # e.g. sm_00001
    ap.add_argument("--matches", required=True)           # AnyLoc matches.csv
    ap.add_argument("--per-chip-topk", type=int, default=5)
    ap.add_argument("--topn", type=int, default=7, help="number of gallery submaps to write")
    ap.add_argument("--out", default="", help="output CSV; default $ROOT/anyloc_io/loop_votes.csv")
    args = ap.parse_args()

    out_csv = args.out or os.path.join(args.root, "anyloc_io", "loop_votes.csv")

    owners = _load_tile_owners(args.root)
    groups = _read_matches(args.matches)
    ranked = _rank_weight_votes(groups, owners, args.submap, args.per_chip_topk, args.topn)

    rows = _read_existing(out_csv)

    now = datetime.datetime.utcnow().isoformat() + "Z"
    # Build row for this submap with dynamic columns
    row = {"query_submap": args.submap, "updated_at_iso": now}
    for i, (sid, sc) in enumerate(ranked, start=1):
        row[f"match{i}_submap"] = sid
        # Keep fixed precision for numeric stability
        row[f"match{i}_score"]  = f"{float(sc):.6f}" if sid else ""

    rows[args.submap] = row
    _write_all(out_csv, rows, args.topn)

    # Compact console summary
    shown = [f"{sid}({sc:.3f})" for sid, sc in ranked if sid]
    print(f"[ok] updated {out_csv} for {args.submap}: top-{args.topn} = {', '.join(shown) if shown else '—'}")

if __name__ == "__main__":
    main()
