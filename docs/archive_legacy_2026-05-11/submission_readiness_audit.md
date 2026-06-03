# Submission Readiness Audit

Date: 2026-05-07

## Verdict

The project is ready for a workshop-style research prototype presentation, but
it is not yet ready for a full conference submission. The central method idea is
clear; the limiting factors are experimental coverage, formal metrics,
baseline parity, and reproducibility packaging.

## Current Strengths

- The main claim is focused: RF/ODE image editing benefits from decoupling
  source preservation and target editing dynamics.
- The implementation follows the stable formulation:

```text
xdot_t = v_src + u_rec + u_edit
```

- The repository contains runnable ablation scripts for base-only, direct
  target, anchor-only, decoupled reconstruction, and local sunglasses edits.
- Most current output folders include `result.png`, `stats.json`,
  `metadata.json`, and `command.txt`.
- The project records negative results and failure modes instead of only
  keeping polished samples.

## Blocking Gaps Before Submission

### 1. Systematic experiments

The current evidence is still dominated by a small set of case studies. Build a
fixed matrix across at least three edit categories:

```text
local accessory insertion
color / attribute editing
semantic or object replacement
```

Each category should run the same comparison set:

```text
base only
direct target
anchor only
decoupled reconstruction
full method
```

### 2. Paper-grade metrics

The existing `stats.json` files mainly log internal process quantities. Add a
separate evaluation pass for edit success and preservation:

```text
CLIP or VLM edit alignment
LPIPS / DINO / SSIM source preservation
mask-outside L1 or LPIPS drift
runtime and peak memory
```

### 3. Baseline parity

External methods such as RF-Solver, FireFlow, ReFlex, FlowEdit, and h-Edit
should be treated as baselines only when they run under comparable inputs,
resolution, prompts, and seeds. If a method cannot run because of memory or
framework mismatch, record that as a limitation rather than using it as a main
comparison.

### 4. Reproducibility package

Add an environment file before submission:

```text
environment.yml or requirements.txt
```

Also keep a single command for each table:

```text
scripts/run_main_table.sh
scripts/run_ablation_table.sh
scripts/make_paper_figures.sh
```

### 5. Manuscript materials

The repository currently has workshop poster files but no full manuscript draft
or reference list. Before entering the academic-pipeline integrity gate, create:

```text
paper/outline.md
paper/draft.md or paper/main.tex
paper/references.bib
paper/figures.md
```

## Recommended Next Change

Do not tune another single panda or rabbit sample first. The next useful change
is to create a fixed experiment manifest with prompts, seeds, input images, and
method variants. Once that manifest exists, every new result can be assigned to
a paper table instead of becoming another one-off probe.

## Current Output Record Audit

The reproducibility audit was run with:

```bash
/home/Wu_25R8111/ENTER/envs/flowedit/bin/python scripts/audit_experiment_records.py \
  --outputs-dir outputs \
  --json-output experiments/output_record_audit.json
```

Current result:

```text
runs: 55
complete: 34
incomplete: 21
```

This means the repository has enough exploratory evidence to plan a paper, but
only complete runs should be used in figures or tables. The 2026-05-06 gradient
and side-profile runs mostly need `command.txt`; several older retry/probe
directories should either be archived or rerun into complete folders.
