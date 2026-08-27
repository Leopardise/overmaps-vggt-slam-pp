#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, glob, numbers
from collections import defaultdict

# ---------- helpers ----------
def load_tile_owners(root):
    owners = defaultdict(list)
    for sm_dir in sorted(glob.glob(os.path.join(root, "submaps", "sm_*"))):
        sm_id = os.path.basename(sm_dir)
        tj = os.path.join(sm_dir, "tiles.json")
        if not os.path.isfile(tj):
            continue
        data = json.load(open(tj, "r"))
        for tid in data.get("overlapped_tile_ids", []):
            owners[int(tid)].append(sm_id)
    return owners

def pad_ids(ids, nx, ny, pad=1, neighbors=4):
    out = set(ids); frontier = set(ids)
    if pad <= 0:
        return sorted(out)
    if neighbors == 8:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    for _ in range(pad):
        newf = set()
        for tid in frontier:
            y, x = divmod(tid, nx)
            for dy, dx in nbrs:
                yy, xx = y+dy, x+dx
                if 0 <= yy < ny and 0 <= xx < nx:
                    nid = yy*nx + xx
                    if nid not in out:
                        out.add(nid); newf.add(nid)
        frontier = newf
        if not frontier:
            break
    return sorted(out)

# ---------- schema-agnostic vote parsing ----------
def _coerce_sm_id(x):
    s = str(x)
    if s.startswith("sm_"):
        return s
    try:
        return f"sm_{int(s):05d}"
    except Exception:
        return None

def _score_like(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool)

def _collect_score_maps(obj, out):
    """Recursively collect dicts that look like {'sm_*': numeric, ...}."""
    if isinstance(obj, dict):
        # does this dict look like a score map?
        keys = list(obj.keys())
        if keys and all(isinstance(k, str) for k in keys):
            sm_pairs = []
            for k, v in obj.items():
                sid = _coerce_sm_id(k)
                if sid is not None and _score_like(v):
                    sm_pairs.append((sid, float(v)))
            if len(sm_pairs) >= 1:
                out.append(dict(sm_pairs))
        # recurse into values
        for v in obj.values():
            _collect_score_maps(v, out)
    elif isinstance(obj, list):
        for it in obj:
            _collect_score_maps(it, out)

def _collect_list_of_pairs(obj):
    """Find lists of {'submap_id':..,'score':..} or [ [sid,score], ... ]."""
    rows = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                sid = it.get("submap_id") or it.get("submap") or it.get("id")
                sc  = it.get("score") or it.get("votes") or it.get("value") or it.get("weight")
                sid = _coerce_sm_id(sid)
                if sid and _score_like(sc):
                    rows.append((sid, float(sc)))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                sid = _coerce_sm_id(it[0]); sc = it[1]
                if sid and _score_like(sc):
                    rows.append((sid, float(sc)))
    return rows

def parse_votes(votes_json_path, self_submap):
    v = json.load(open(votes_json_path, "r"))

    # 0) Explicit common keys
    if isinstance(v, dict) and "ranked_submaps" in v and isinstance(v["ranked_submaps"], list):
        rows = _collect_list_of_pairs(v["ranked_submaps"])
        if rows:
            pass
        else:
            # try dict entries inside ranked_submaps
            rows = []
            for it in v["ranked_submaps"]:
                if isinstance(it, dict):
                    sid = it.get("submap_id") or it.get("submap") or it.get("id")
                    sc  = it.get("score") or it.get("votes") or it.get("value") or it.get("weight")
                    sid = _coerce_sm_id(sid)
                    if sid and _score_like(sc):
                        rows.append((sid, float(sc)))
        if rows:
            pass
        else:
            raise KeyError("Empty 'ranked_submaps' payload.")
    else:
        # 1) look for dict score maps anywhere
        maps = []
        _collect_score_maps(v, maps)
        maps.sort(key=lambda d: len(d), reverse=True)
        rows = list(maps[0].items()) if maps else []

        # 2) or a list of pairs/dicts anywhere
        if not rows:
            cand = []
            def recurse(o):
                nonlocal cand
                if isinstance(o, list):
                    rows_here = _collect_list_of_pairs(o)
                    if len(rows_here) > len(cand):
                        cand = rows_here
                    for it in o:
                        recurse(it)
                elif isinstance(o, dict):
                    for val in o.values():
                        recurse(val)
            recurse(v)
            rows = cand

        # 3) or a single 'best' field → make a minimal ranking
        if not rows and isinstance(v, dict):
            best = v.get("best") or v.get("best_submap") or v.get("best_gallery_submap")
            sid = _coerce_sm_id(best)
            if sid:
                rows = [(sid, 1.0)]

    if not rows:
        raise KeyError("Could not find any submap scores in votes JSON.")
    # dedup & normalize, drop self, sort
    agg = {}
    for sid, sc in rows:
        if sid == self_submap:
            continue
        agg[sid] = max(sc, agg.get(sid, sc))
    ranked = sorted([{"submap_id": k, "score": v} for k, v in agg.items()],
                    key=lambda r: r["score"], reverse=True)
    print(f"[votes] Parsed {len(ranked)} entries. Top-5:",
          [(r['submap_id'], round(r['score'], 3)) for r in ranked[:5]])
    return ranked

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Build retrieval region (top-N submaps + padding) and write database/queries lists")
    ap.add_argument("--root", required=True)
    ap.add_argument("--submap", required=True)
    ap.add_argument("--votes-json", required=True)
    ap.add_argument("--topn", type=int, default=2)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--neighbors", type=int, choices=[4, 8], default=4)
    ap.add_argument("--out-dir", default="anyloc_io")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.root, "index.json"), "r"))
    nx, ny = int(meta["nx"]), int(meta["ny"])
    tiles_dir = os.path.join(args.root, "tiles")

    ranked = parse_votes(args.votes_json, self_submap=args.submap)
    chosen = [r["submap_id"] for r in ranked[:args.topn]]
    if not chosen:
        raise SystemExit("No top submaps found (after excluding self).")

    owners = load_tile_owners(args.root)
    owned = {tid for tid, sms in owners.items() if any(s in chosen for s in sms)}
    region = pad_ids(sorted(owned), nx, ny, pad=args.pad, neighbors=args.neighbors)

    out_root = os.path.join(args.root, args.out_dir, args.submap)
    os.makedirs(out_root, exist_ok=True)

    # database tiles
    db_txt = os.path.join(out_root, "database.txt")
    with open(db_txt, "w") as f:
        for tid in region:
            p = os.path.join(tiles_dir, f"tile_{tid:05d}.png")
            if os.path.isfile(p):
                f.write(p + "\n")

    # query chips
    chips_dir = os.path.join(args.root, "submaps", args.submap, "chips")
    q_txt = os.path.join(out_root, "queries.txt")
    chips = sorted(glob.glob(os.path.join(chips_dir, "*.png")))
    with open(q_txt, "w") as f:
        for chip in chips:
            f.write(chip + "\n")

    # region bookkeeping
    with open(os.path.join(out_root, "region.json"), "w") as f:
        json.dump({
            "submap": args.submap,
            "top_submaps": chosen,
            "region_tile_ids": region,
            "neighbors": args.neighbors,
            "pad": args.pad
        }, f, indent=2)

    print(f"[ok] database.txt  → {db_txt}")
    print(f"[ok] queries.txt   → {q_txt}")
    print(f"[ok] region.json   → {os.path.join(out_root, 'region.json')}")
    print(f"[info] region tiles: {len(region)} from top {args.topn} submaps (+pad={args.pad}, Nbrs={args.neighbors})")

if __name__ == "__main__":
    main()
