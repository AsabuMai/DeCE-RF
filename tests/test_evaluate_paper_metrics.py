from pathlib import Path

from PIL import Image

from scripts.init_pretty_visual_audit import main as init_pretty_visual_audit_main
from scripts.init_baseline_parity_manifest import main as init_baseline_parity_manifest_main
from scripts.evaluate_paper_metrics import evaluate_run, find_run_dirs


def test_evaluate_run_uses_failure_annotations(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "cat_crown" / "full" / "seed_10"
    run_dir.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (128, 128, 128)).save(source)
    Image.new("RGB", (8, 8), (130, 128, 128)).save(run_dir / "result.png")
    (run_dir / "stats.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        '{"image": "%s", "source_prompt": "source", "target_prompt": "target"}\n' % source,
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text("run\n", encoding="utf-8")

    record = evaluate_run(
        run_dir,
        outputs_dir,
        failure_annotations={
            "cat_crown/full/seed_10": {
                "failure_flag": "localization_error",
                "failure_note": "crown too high",
            }
        },
    )

    assert record["complete"] is True
    assert record["failure_flag"] == "localization_error"
    assert record["failure_note"] == "crown too high"


def test_find_run_dirs_ignores_nested_support_artifacts(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "yellow_car_blue" / "full" / "seed_10"
    support_dir = run_dir / "masks" / "vehicle_paint"
    support_dir.mkdir(parents=True)

    for directory in (run_dir, support_dir):
        Image.new("RGB", (8, 8), (130, 128, 128)).save(directory / "result.png")
        (directory / "stats.json").write_text("[]\n", encoding="utf-8")
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        (directory / "command.txt").write_text("run\n", encoding="utf-8")

    assert find_run_dirs(outputs_dir) == [run_dir]


def test_find_run_dirs_can_filter_task_names(tmp_path: Path):
    keep_dir = tmp_path / "outputs" / "cat_crown" / "full" / "seed_10"
    drop_dir = tmp_path / "outputs" / "red_chair_blue" / "full" / "seed_10"

    for directory in (keep_dir, drop_dir):
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")

    assert find_run_dirs(tmp_path / "outputs", task_names={"cat_crown"}) == [keep_dir]


def test_find_run_dirs_can_filter_method_names(tmp_path: Path):
    keep_dir = tmp_path / "outputs" / "cat_crown" / "full" / "seed_10"
    drop_dir = tmp_path / "outputs" / "cat_crown" / "full_no_rec" / "seed_10"

    for directory in (keep_dir, drop_dir):
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")

    assert find_run_dirs(tmp_path / "outputs", method_names={"full"}) == [keep_dir]


def test_find_run_dirs_can_filter_seeds(tmp_path: Path):
    keep_dir = tmp_path / "outputs" / "cat_crown" / "full" / "seed_10"
    drop_dir = tmp_path / "outputs" / "cat_crown" / "full" / "seed_11"

    for directory in (keep_dir, drop_dir):
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")

    assert find_run_dirs(tmp_path / "outputs", seeds={"10"}) == [keep_dir]


def test_init_pretty_visual_audit_writes_full_go_no_go_template(tmp_path: Path, monkeypatch):
    output = tmp_path / "audit.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "init_pretty_visual_audit.py",
            "--output",
            str(output),
            "--tasks",
            "P1 P2",
            "--methods",
            "M0 M5 M4",
            "--seeds",
            "10 11",
        ],
    )

    assert init_pretty_visual_audit_main() == 0

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 2 * 3 * 2
    assert "cat_crown,base_only,10" in lines[1]
    assert any("dog_sunglasses,full_no_ref,11" in line for line in lines)


def test_init_baseline_parity_manifest_writes_matched_rows(tmp_path: Path, monkeypatch):
    output = tmp_path / "baseline_manifest.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "init_baseline_parity_manifest.py",
            "--output",
            str(output),
            "--baselines",
            "flowedit reflex",
            "--tasks",
            "cat_crown mug_heart",
            "--seeds",
            "10",
        ],
    )

    assert init_baseline_parity_manifest_main() == 0

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 2 * 2
    assert any("flowedit,cat_crown,10,pending" in line for line in lines)
    assert any("reflex,mug_heart,10,pending" in line for line in lines)
    assert any("outputs/baselines/reflex/mug_heart/seed_10/result.png" in line for line in lines)
