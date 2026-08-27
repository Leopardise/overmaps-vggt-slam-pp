#!/usr/bin/env python3
import os, json, argparse, glob
from collections import defaultdict

def load_tile_owners(root):
    owners = defaultdict(list)  # tid -> [sm_id,...]
    for sm_dir in sorted(glob.glob(os.path.join(root, "submaps", "sm_*"))):
        sm_id = os.path.basename(sm_dir)
        tj = os.path.join(sm_dir, "tiles.json")
        if not os.path.isfile(tj): continue
        data = json.load(open(tj, "r"))
        for tid in data.get("overlapped_tile_ids", []):
            owners[int(tid)].append(sm_id)
    return owners

def main():
    ap = argparse.ArgumentParser("Aggregate chip→tile matches into per-submap votes")
    ap.add_argument("--root", required=True, help="outputs/run")
    ap.add_argument("--submap", required=True, help="e.g. sm_00001")
    ap.add_argument("--matches-json", default=None,
                    help="defaults to submaps/<SM>/matches_topk.json")
    ap.add_argument("--exclude-self", action="store_true")
    ap.add_argument("--weighted", action="store_true",
                    help="sum cosine scores; else count hits (1 per hit)")
    ap.add_argument("--out", default=None,
                    help="defaults to submaps/<SM>/submap_votes.json")
    args = ap.parse_args()

    sm_dir = os.path.join(args.root, "submaps", args.submap)
    if not os.path.isdir(sm_dir):
        raise SystemExit(f"submap dir not found: {sm_dir}")

    matches_p = args.matches_json or os.path.join(sm_dir, "matches_topk.json")
    if not os.path.isfile(matches_p):
        raise SystemExit(f"matches_topk.json not found: {matches_p}  (run submap_chip_embed_match.py first)")

    matches = json.load(open(matches_p, "r"))
    # expected structure: {"submap": "sm_00001", "topk": K, "matches": [{"chip_id":..,"top_ids":[...],"scores":[...]}]}
    owners = load_tile_owners(args.root)

    score_per_submap = defaultdict(float)

    reg_sm = args.submap
    for rec in matches.get("matches", []):
        tids = rec["top_ids"]
        scs  = rec.get("scores", [1.0]*len(tids))
        for tid, s in zip(tids, scs):
            sm_owners = owners.get(int(tid), [])
            if args.exclude_self and reg_sm in sm_owners:
                continue
            v = float(s) if args.weighted else 1.0
            for other in sm_owners:
                if args.exclude_self and other == reg_sm:
                    continue
                score_per_submap[other] += v

    ranked = sorted(
        [{"submap_id": k, "score": float(v)} for k, v in score_per_submap.items()],
        key=lambda x: -x["score"]
    )

    out_p = args.out or os.path.join(sm_dir, "submap_votes.json")
    with open(out_p, "w") as f:
        json.dump({
            "registering_submap": reg_sm,
            "exclude_self": bool(args.exclude_self),
            "weighted": bool(args.weighted),
            "ranked_submaps": ranked
        }, f, indent=2)

    print(f"[ok] wrote {out_p}")
    if ranked[:5]:
        print("[top5]", ", ".join(f"{r['submap_id']}:{r['score']:.1f}" for r in ranked[:5]))

if __name__ == "__main__":
    main()
