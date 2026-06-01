import json
from pathlib import Path

root = Path("/workspace/rf_h_edit")
tasks = [
    "web_chair_cushion",
    "web_bowl_spoon",
    "web_vase_flowers",
    "web_notebook_pen",
]
methods = [
    "support_v3_controller_rmsgap",
    "support_v3_controller_rmsgap_add_editor_v1",
]

for task in tasks:
    print(f"\n[{task}]")
    for method in methods:
        out = root / "outputs" / "pretty_matrix" / task / method / "seed_10"
        meta_path = out / "metadata.json"
        stats_path = out / "stats.json"
        debug_path = out / "masks" / "operation_v3_debug_metadata.json"
        if not meta_path.exists():
            print(f"  {method}: missing")
            continue
        meta = json.loads(meta_path.read_text())
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
        debug = json.loads(debug_path.read_text()) if debug_path.exists() else {}
        last = stats[-1] if isinstance(stats, list) and stats else {}
        dbg_stats = debug.get("stats", {})
        print(
            "  {method}: candidate={candidate} area={area} bbox={bbox} "
            "local={local} transport={transport} edit_norm={edit_norm:.3f}".format(
                method=method,
                candidate=meta.get("support_candidate") or meta.get("support_score"),
                area=dbg_stats.get("support_area_edit", debug.get("support_area")),
                bbox=debug.get("support_bbox"),
                local=meta.get("edit_local_target_guidance_scale"),
                transport=meta.get("region_target_transport_scale"),
                edit_norm=float(last.get("edit_guidance_norm", 0.0)),
            )
        )
