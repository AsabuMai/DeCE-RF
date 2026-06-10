#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:30:00
#SBATCH -J fetch-e2-new
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/fetch_new_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/fetch_new_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
SRC="$PROJECT/_baselines/src"

cd "$PROJECT"
mkdir -p "$SRC" "$PROJECT/_baselines/logs"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to fetch outside a100-01" >&2
  exit 2
fi

fetch_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local path="$SRC/$name"

  echo "==== $name"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  cd "$path"
  git fetch --all --tags --prune
  git checkout "$commit"
  git rev-parse HEAD
  cd "$PROJECT"
}

fetch_repo "OT-RF" "https://github.com/marianlupascu/OT-RF.git" "abca084f614d23b1d08ef7c1f3bd9d99d25e356a"
fetch_repo "DeltaRectifiedFlowSampling" "https://github.com/gaspardbd/DeltaRectifiedFlowSampling.git" "567b28bc9b0a639950026de28ad16fb8a93725f3"

echo fetch_new_baselines_complete
