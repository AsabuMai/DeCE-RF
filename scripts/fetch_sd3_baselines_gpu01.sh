#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:40:00
#SBATCH -J fetch-sd3-bl
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/fetch_sd3_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/fetch_sd3_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
REG="$BASE/sd3_baseline_source_registry.csv"

cd "$PROJECT"
mkdir -p "$SRC" "$BASE/logs"

host="$(hostname -f)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to fetch outside a100-01" >&2
  exit 2
fi

fetch_repo() {
  local name="$1"
  local url="$2"
  local path="$SRC/$name"

  echo "==== fetch $name"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  cd "$path"
  git fetch --all --tags --prune
  git checkout main || git checkout master
  local commit
  commit="$(git rev-parse HEAD)"
  local status
  status="$(git status --short | wc -l | tr -d ' ')"
  cd "$PROJECT"
  python3 - "$REG" "$name" "$url" "$path" "$commit" "$status" "official_git_repo" <<'PY'
import csv
import sys
from pathlib import Path

registry = Path(sys.argv[1])
name, url, path, commit, dirty_count, note = sys.argv[2:]
fields = ["name", "url", "path", "commit", "dirty_count", "note"]
rows = []
if registry.exists() and registry.stat().st_size:
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
rows = [row for row in rows if row.get("name") != name]
rows.append({
    "name": name,
    "url": url,
    "path": path,
    "commit": commit,
    "dirty_count": dirty_count,
    "note": note,
})
with registry.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
  echo "$name commit=$commit dirty=$status"
}

fetch_repo "FIA-Edit-SD3" "https://github.com/kk42yy/FIA-Edit.git"
fetch_repo "DRFS" "https://github.com/gaspardbd/DeltaRectifiedFlowSampling.git"
fetch_repo "Exploring-MM-DiT" "https://github.com/SNU-VGILab/exploring-mmdit.git"

echo "==== alias DVRF-SD3 -> DRFS"
rm -f "$SRC/DVRF-SD3"
ln -sfn "$SRC/DRFS" "$SRC/DVRF-SD3"

echo "==== create qiki-local-blending-SD3 scaffold"
mkdir -p "$SRC/qiki-local-blending-SD3"
cat > "$SRC/qiki-local-blending-SD3/README.md" <<'EOF'
# qiki-local-blending-SD3

Local project scaffold for the SD3 q_i/k_i projection replacement plus local
blending baseline described by "Exploring Multimodal Diffusion Transformers for
Enhanced Prompt-based Image Editing".

There is no separate public repository discovered under this exact name. The
official Exploring-MM-DiT repository currently contains paper assets/README and
states that code is coming soon, so this directory is reserved for the local
adapter implementation.
EOF

python3 - "$REG" "DVRF-SD3" "alias:https://github.com/gaspardbd/DeltaRectifiedFlowSampling" "$SRC/DVRF-SD3" "$(cd "$SRC/DRFS" && git rev-parse HEAD)" "0" "alias_to_DRFS_official_repo_redirect" <<'PY'
import csv
import sys
from pathlib import Path
registry = Path(sys.argv[1])
name, url, path, commit, dirty_count, note = sys.argv[2:]
fields = ["name", "url", "path", "commit", "dirty_count", "note"]
rows = []
if registry.exists() and registry.stat().st_size:
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
rows = [row for row in rows if row.get("name") != name]
rows.append(dict(name=name, url=url, path=path, commit=commit, dirty_count=dirty_count, note=note))
with registry.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

python3 - "$REG" "qiki-local-blending-SD3" "local_scaffold:no_public_repo_found" "$SRC/qiki-local-blending-SD3" "local-20260606" "0" "local_adapter_scaffold_for_qi_ki_local_blending" <<'PY'
import csv
import sys
from pathlib import Path
registry = Path(sys.argv[1])
name, url, path, commit, dirty_count, note = sys.argv[2:]
fields = ["name", "url", "path", "commit", "dirty_count", "note"]
rows = []
if registry.exists() and registry.stat().st_size:
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
rows = [row for row in rows if row.get("name") != name]
rows.append(dict(name=name, url=url, path=path, commit=commit, dirty_count=dirty_count, note=note))
with registry.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

cat "$REG"
echo fetch_sd3_baselines_complete
