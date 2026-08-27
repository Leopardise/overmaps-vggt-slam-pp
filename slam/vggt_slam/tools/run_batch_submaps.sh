#!/usr/bin/env bash
# Batch runner: chip→FAISS vote→AnyLoc region→VPR→(optional) loop-votes update→viz
# Usage:
#   vggt_slam/tools/run_batch_submaps.sh ROOT START_ID END_ID [options]
# Example:
#   vggt_slam/tools/run_batch_submaps.sh outputs/05 0 24 \
#     --dino facebook/dinov2-base --faiss hnsw \
#     --match-topk 20 --vote-topn 10 --vote-pad 0 \
#     --vpr-clusters 64 --vpr-topk 50 --max-edge 1024 \
#     --force --update-loop-votes --per-chip-topk 7 --topn-loop 7

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 ROOT START_ID END_ID [--dino MODEL] [--faiss hnsw|flatip] [--match-topk N] [--vote-topn N] [--vote-pad N] [--vpr-clusters C] [--vpr-topk K] [--max-edge M] [--force] [--update-loop-votes] [--per-chip-topk K] [--topn-loop N]"
  exit 2
fi

ROOT="$1"; shift
START_ID="$1"; shift
END_ID="$1"; shift

# ---- defaults / flags (all supported by your current scripts) ----
DINO_MODEL="facebook/dinov2-base"
FAISS_KIND="hnsw"
MATCH_TOPK=20
VOTE_TOPN=10
VOTE_PAD=0
VPR_CLUSTERS=64
VPR_TOPK=50
MAX_EDGE=1024
FORCE=0
UPDATE_LOOP_VOTES=0
PER_CHIP_TOPK=7
TOPN_LOOP=7

while (( "$#" )); do
  case "$1" in
    --dino)             DINO_MODEL="$2"; shift 2;;
    --faiss)            FAISS_KIND="$2"; shift 2;;
    --match-topk)       MATCH_TOPK="$2"; shift 2;;
    --vote-topn)        VOTE_TOPN="$2"; shift 2;;
    --vote-pad)         VOTE_PAD="$2"; shift 2;;
    --vpr-clusters)     VPR_CLUSTERS="$2"; shift 2;;
    --vpr-topk)         VPR_TOPK="$2"; shift 2;;
    --max-edge)         MAX_EDGE="$2"; shift 2;;
    --force)            FORCE=1; shift 1;;
    --update-loop-votes) UPDATE_LOOP_VOTES=1; shift 1;;
    --per-chip-topk)    PER_CHIP_TOPK="$2"; shift 2;;
    --topn-loop)        TOPN_LOOP="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 2;;
  esac
done

log() { printf '%s\n' "$*"; }
run() { log "[run] $*"; eval "$@"; }
need_file() { [[ -f "$1" ]] || { echo "[err] missing $1"; exit 3; }; }
nonempty_file() { [[ -s "$1" ]]; }

ensure_anyloc_env() {
  if [[ -d "AnyLoc" ]]; then
    export PYTHONPATH="$PWD/AnyLoc:$PWD/AnyLoc/demo:${PYTHONPATH:-}"
  fi
  python - >/dev/null 2>&1 <<'PY'
try:
    import einops, fast_pytorch_kmeans  # noqa
    ok = True
except Exception:
    ok = False
import sys; sys.exit(0 if ok else 1)
PY
  if [[ $? -ne 0 ]]; then
    pip install -q einops fast_pytorch_kmeans || true
  fi
}

validate_votes_json() {
  # exit 0 (true) if ≥1 scored entry (any supported schema), else 1
  local json="$1"
  [[ -f "$json" ]] || return 1
  python - "$json" <<'PY'
import json, sys, numbers
p = sys.argv[1]
try:
    v = json.load(open(p, "r"))
except Exception:
    sys.exit(1)

def score_like(x): return isinstance(x, numbers.Real) and not isinstance(x, bool)
ok = False
if isinstance(v, dict):
    # (A) flat dict: {"sm_00012": 7.5, ...}
    if any(isinstance(k, str) and k.startswith("sm_") and score_like(v.get(k)) for k in v.keys()):
        ok = True
    # (B) {"scores": {...}}
    if isinstance(v.get("scores"), dict) and any(score_like(s) for s in v["scores"].values()):
        ok = True
    # (C) {"ranked_submaps":[...]} or {"ranking":[...]} (list of [sid,score] or dict rows)
    for key in ("ranked_submaps", "ranking"):
        rs = v.get(key)
        if isinstance(rs, list) and len(rs) > 0:
            ok = True
            break
sys.exit(0 if ok else 1)
PY
}

# ---- preflight: must exist ----
need_file "$ROOT/index.json"

OWNERS_JSON="$ROOT/submap_index/tile_owners.json"
if [[ ! -f "$OWNERS_JSON" ]]; then
  log "[prep] tile_owners.json missing → building…"
  run "python vggt_slam/tools/build_tile_owners.py --root \"$ROOT\""
fi

GE_DIR="$ROOT/global_embeddings"
FAISS_IDX="$GE_DIR/faiss_${FAISS_KIND}.index"
if [[ ! -f "$FAISS_IDX" ]]; then
  log "[prep] FAISS index missing → building ($FAISS_KIND)…"
  run "python vggt_slam/tools/faiss_global_index.py --root \"$ROOT\" --index \"$FAISS_KIND\""
fi

ensure_anyloc_env

RETR_HAD=()
RETR_NONE=()

for (( sid=START_ID; sid<=END_ID; sid++ )); do
  SUB=$(printf "sm_%05d" "$sid")
  SM_DIR="$ROOT/submaps/$SUB"
  IO="$ROOT/anyloc_io/$SUB"
  mkdir -p "$IO"

  echo ""
  echo "==================== Submap $SUB ===================="

  MATCH_JSON="$SM_DIR/matches_topk.json"
  COVIS_PNG="$SM_DIR/covis_heatmap.png"
  VOTES_JSON="$SM_DIR/faiss_votes_by_submap.json"
  DB_TXT="$IO/database.txt"
  Q_TXT="$IO/queries.txt"
  REGION_JSON="$IO/region.json"
  MATCHES_CSV="$IO/matches.csv"
  CENTERS_NPY="$IO/c_centers.npy"
  GRID_PNG="$IO/patch_matches_grid.png"

  if [[ $FORCE -eq 1 ]]; then
    rm -f "$MATCH_JSON" "$COVIS_PNG" "$VOTES_JSON" \
          "$DB_TXT" "$Q_TXT" "$REGION_JSON" \
          "$MATCHES_CSV" "$CENTERS_NPY" "$GRID_PNG"
  fi

  # 1) chip→tile matching (uses your updated embed script)
  if [[ ! -f "$MATCH_JSON" ]]; then
    run "python vggt_slam/tools/submap_chip_embed_match.py \
      --root \"$ROOT\" \
      --submap \"$SUB\" \
      --faiss-index \"$FAISS_KIND\" \
      --topk $MATCH_TOPK \
      --dino-model \"$DINO_MODEL\" \
      --mode resize --max-edge $MAX_EDGE \
      --save-chips --exclude-self --vis"
  else
    log "[skip] matches exist → $MATCH_JSON"
  fi

  # 2) submap vote (ONLY flags your script supports)
  if [[ ! -f "$VOTES_JSON" ]]; then
    run "python vggt_slam/tools/submap_faiss_vote.py \
      --root \"$ROOT\" --submap \"$SUB\" \
      --exclude-self --weighted"
  else
    log "[skip] votes exist → $VOTES_JSON"
  fi

  # 2a) votes preflight: skip region/VPR if empty
  if ! validate_votes_json "$VOTES_JSON"; then
    log "[warn] $SUB: no valid FAISS votes — skipping retrieval (no region)."
    RETR_NONE+=("$SUB")
    echo "[ok] Submap $SUB done."
    continue
  fi

  # 3) AnyLoc retrieval region (topN + pad)
  if [[ ! -f "$DB_TXT" || ! -f "$Q_TXT" ]]; then
    run "python vggt_slam/tools/build_retrieval_region.py \
      --root \"$ROOT\" --submap \"$SUB\" \
      --votes-json \"$VOTES_JSON\" \
      --topn $VOTE_TOPN --pad $VOTE_PAD"
  else
    log "[skip] AnyLoc region exists → $DB_TXT & $Q_TXT"
  fi

  # Ensure region files have content
  if ! nonempty_file "$DB_TXT" || ! nonempty_file "$Q_TXT"; then
    log "[warn] $SUB: empty retrieval region — skipping VPR."
    RETR_NONE+=("$SUB")
    echo "[ok] Submap $SUB done."
    continue
  fi

  # 4) AnyLoc VPR
  if [[ ! -f "$MATCHES_CSV" ]]; then
    LOAD_CENTERS_FLAG=""
    [[ -f "$CENTERS_NPY" ]] && LOAD_CENTERS_FLAG="--load-centers \"$CENTERS_NPY\""
    run "python vggt_slam/tools/run_vpr_retrieval.py \
      --queries \"$Q_TXT\" \
      --database \"$DB_TXT\" \
      --out \"$MATCHES_CSV\" \
      --backbone dinov2_vitb14 \
      --mode resize --max-edge $MAX_EDGE \
      --clusters $VPR_CLUSTERS \
      --topk $VPR_TOPK \
      --save-centers \"$CENTERS_NPY\" $LOAD_CENTERS_FLAG"
  else
    log "[skip] VPR matches exist → $MATCHES_CSV"
  fi

  # 5) optional: update loop_votes.csv
  if [[ $UPDATE_LOOP_VOTES -eq 1 ]]; then
    if nonempty_file "$MATCHES_CSV"; then
      run "python vggt_slam/tools/update_loop_votes_csv.py \
        --root \"$ROOT\" \
        --submap \"$SUB\" \
        --matches \"$MATCHES_CSV\" \
        --per-chip-topk $PER_CHIP_TOPK \
        --topn $TOPN_LOOP"
    else
      log "[warn] $SUB: matches.csv empty — skipping loop_votes update."
    fi
  fi

  # 6) quick viz of AnyLoc patch matches
  if [[ ! -f "$GRID_PNG" ]]; then
    run "python vggt_slam/tools/vis_patch_matches.py \
      --root \"$ROOT\" \
      --io-dir \"$IO\" \
      --matches \"$MATCHES_CSV\" \
      --topk 10 --max-queries 24"
  else
    log "[skip] patch matches viz exists → $GRID_PNG"
  fi

  RETR_HAD+=("$SUB")
  echo "[ok] Submap $SUB done."
done

echo ""
echo "========== Retrieval Summary =========="
if (( ${#RETR_HAD[@]} )); then
  echo "[had retrieval] ${RETR_HAD[*]}"
else
  echo "[had retrieval] none"
fi
if (( ${#RETR_NONE[@]} )); then
  echo "[no retrieval]  ${RETR_NONE[*]}"
else
  echo "[no retrieval]  none"
fi
echo "[done] Batch complete for $(printf 'sm_%05d' "$START_ID") … $(printf 'sm_%05d' "$END_ID")"
