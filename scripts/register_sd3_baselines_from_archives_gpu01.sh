#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:20:00
#SBATCH -J reg-sd3-bl
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/register_sd3_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/register_sd3_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
INCOMING="$BASE/incoming"
REG="$BASE/sd3_baseline_source_registry.csv"

cd "$PROJECT"
mkdir -p "$SRC" "$BASE/logs"

host="$(hostname -f)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to register outside a100-01" >&2
  exit 2
fi

unpack_zip() {
  local name="$1"
  local zip="$INCOMING/$name.zip"
  local path="$SRC/$name"
  echo "==== unpack $name"
  rm -rf "$path"
  python3 - "$zip" "$SRC" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not zip_path.exists():
    raise SystemExit(f"missing archive: {zip_path}")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dst)
PY
  cd "$path"
  local commit
  commit="$(git rev-parse HEAD)"
  local dirty
  dirty="$(git status --short | wc -l | tr -d ' ')"
  cd "$PROJECT"
  python3 - "$REG" "$name" "local_archive_from_official_github" "$path" "$commit" "$dirty" "uploaded_archive_unpacked_on_a100" <<'PY'
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
  echo "$name commit=$commit dirty=$dirty"
}

unpack_zip FIA-Edit-SD3
unpack_zip Exploring-MM-DiT

echo "==== prepare DRFS canonical source"
if [[ ! -d "$SRC/DRFS/.git" ]]; then
  if [[ -d "$SRC/DeltaRectifiedFlowSampling/.git" ]]; then
    cp -a "$SRC/DeltaRectifiedFlowSampling" "$SRC/DRFS"
  else
    echo "Missing existing DeltaRectifiedFlowSampling source for DRFS" >&2
    exit 3
  fi
fi

rm -f "$SRC/DVRF-SD3"
ln -sfn "$SRC/DRFS" "$SRC/DVRF-SD3"

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

python3 - "$REG" "$SRC" <<'PY'
import csv
from pathlib import Path
import subprocess
import sys

reg = Path("/cluster/users/grad/2025/25t8103/project/_baselines/sd3_baseline_source_registry.csv")
src = Path(sys.argv[1])
fields = ["name", "url", "path", "commit", "dirty_count", "note"]
rows = []
if reg.exists() and reg.stat().st_size:
    with reg.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
rows = [r for r in rows if r.get("name") not in {"DRFS", "DVRF-SD3", "qiki-local-blending-SD3"}]

def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

drfs = src / "DRFS"
head = git_head(drfs)
rows.extend([
    dict(name="DRFS", url="https://github.com/gaspardbd/DeltaRectifiedFlowSampling.git", path=str(drfs), commit=head, dirty_count="0", note="official_DRFS_source_prepared"),
    dict(name="DVRF-SD3", url="alias:https://github.com/gaspardbd/DeltaVelocityRectifiedFlow -> DeltaRectifiedFlowSampling", path=str(src / "DVRF-SD3"), commit=head, dirty_count="0", note="official_DVRF_url_redirects_to_DRFS_repo"),
    dict(name="qiki-local-blending-SD3", url="local_scaffold:no_public_repo_found", path=str(src / "qiki-local-blending-SD3"), commit="local-20260606", dirty_count="0", note="local_adapter_scaffold_for_qi_ki_local_blending"),
])
with reg.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

cat "$REG"
echo register_sd3_baselines_complete
