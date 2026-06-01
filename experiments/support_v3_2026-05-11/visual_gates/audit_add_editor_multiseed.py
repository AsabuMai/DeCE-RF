import json
from pathlib import Path

root = Path("/workspace/rf_h_edit")
method = "support_v3_controller_rmsgap_add_editor_v1"
tasks = ["rabbit_sunglasses", "web_chair_cushion", "web_bowl_spoon", "web_vase_flowers"]
seeds = [10, 11, 12]

ok = True
for task in tasks:
    for seed in seeds:
        out = root / "outputs" / "pretty_matrix" / task / method / f"seed_{seed}"
        required = [out / "result.png", out / "metadata.json", out / "stats.json"]
        missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
        if missing:
            ok = False
            print(f"MISS {task} seed{seed}: {missing}")
            continue
        meta = json.loads((out / "metadata.json").read_text())
        debug_path = out / "masks" / "operation_v3_debug_metadata.json"
        debug = json.loads(debug_path.read_text()) if debug_path.exists() else {}
        dbg_stats = debug.get("stats", {})
        print(
            "{task:22s} seed{seed}: candidate={cand:18s} area={area:.4f} "
            "local={local:.2f} transport={transport:.2f} result={size}".format(
                task=task,
                seed=seed,
                cand=str(meta.get("support_candidate") or meta.get("support_score")),
                area=float(dbg_stats.get("support_area_edit", debug.get("support_area", 0.0))),
                local=float(meta.get("edit_local_target_guidance_scale", 0.0)),
                transport=float(meta.get("region_target_transport_scale", 0.0)),
                size=(out / "result.png").stat().st_size,
            )
        )
raise SystemExit(0 if ok else 1)
