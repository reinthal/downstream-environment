#!/usr/bin/env bash
# Copy the fibs-basharena volume's new files from one Modal workspace (profile) to
# another through /data (this host has no cross-workspace access inside Modal).
# Only paths missing in the destination are transferred; existing files are kept.
#
#   sync_volumes.sh <src_profile> <dst_profile> [prefix ...]
#   e.g. sync_volumes.sh cadenza reinthal fibs/main steps/main/honest steps/main/attack probes/main
set -euo pipefail
SRC=$1; DST=$2; shift 2
PREFIXES=("$@")
[ ${#PREFIXES[@]} -gt 0 ] || PREFIXES=(fibs/main steps/main/attack steps/main/honest probes/main probes/pilot)
VOL=fibs-basharena
STAGE=/data/fibs-sync/$SRC-to-$DST
mkdir -p "$STAGE"
M() { nix shell nixpkgs#uv --command "$@"; }

for p in "${PREFIXES[@]}"; do
  src_list=$(MODAL_PROFILE=$SRC M uvx modal volume ls $VOL "$p" --json 2>/dev/null | /home/kog/downstream-environment/.venv-api/bin/python -c 'import sys,json; print("\n".join(e["filename"] + ("/" if e.get("type") == "dir" else "") for e in json.load(sys.stdin)))' 2>/dev/null || true)
  dst_list=$(MODAL_PROFILE=$DST M uvx modal volume ls $VOL "$p" --json 2>/dev/null | /home/kog/downstream-environment/.venv-api/bin/python -c 'import sys,json; print("\n".join(e["filename"] + ("/" if e.get("type") == "dir" else "") for e in json.load(sys.stdin)))' 2>/dev/null || true)
  missing=$(comm -23 <(echo "$src_list" | sort) <(echo "$dst_list" | sort) | grep -v '^$' || true)
  n=$(echo "$missing" | grep -c . || true)
  echo "== $p: $n files to sync"
  [ "$n" -gt 0 ] || continue
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    rel=${f#/}
    if [[ "$rel" == */ ]]; then            # a directory entry (e.g. a probe folder): copied whole
      rel=${rel%/}
      mkdir -p "$STAGE/$(dirname "$rel")"
      MODAL_PROFILE=$SRC M uvx modal volume get $VOL "$rel/" "$STAGE/$(dirname "$rel")/" --force >/dev/null
      MODAL_PROFILE=$DST M uvx modal volume put $VOL "$STAGE/$rel" "$rel" --force >/dev/null
      rm -rf "$STAGE/$rel"
    else
      mkdir -p "$STAGE/$(dirname "$rel")"
      MODAL_PROFILE=$SRC M uvx modal volume get $VOL "$rel" "$STAGE/$rel" --force >/dev/null
      MODAL_PROFILE=$DST M uvx modal volume put $VOL "$STAGE/$rel" "$rel" --force >/dev/null
      rm -f "$STAGE/$rel"
    fi
    echo "  synced $rel"
  done <<< "$missing"
done
echo SYNC_DONE
