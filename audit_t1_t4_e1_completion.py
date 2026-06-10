from pathlib import Path


ROOT = Path("/cluster/users/grad/2025/25t8103/project")
TASKS = """
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
bowl_apple_inside
white_bowl_orange_tabletop_phase2
brown_bowl_lemon_phase2
tshirt_star
mug_heart
tote_leaf
red_office_chair_to_blue_office_chair
green_mug_orange_phase2
yellow_vase_blue_phase2
""".split()
METHODS = """
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
""".split()
SEEDS = "10 11 12".split()

missing = []
complete = 0
for task in TASKS:
    for method in METHODS:
        for seed in SEEDS:
            out_dir = ROOT / "outputs" / "pretty_matrix" / task / method / f"seed_{seed}"
            absent = [
                name
                for name in ("result.png", "metadata.json", "stats.json", "command.txt")
                if not (out_dir / name).is_file()
            ]
            if absent:
                missing.append((task, method, seed, ",".join(absent)))
            else:
                complete += 1

print(f"complete={complete}")
print(f"expected={len(TASKS) * len(METHODS) * len(SEEDS)}")
print(f"missing={len(missing)}")
for row in missing[:200]:
    print("\t".join(row))
