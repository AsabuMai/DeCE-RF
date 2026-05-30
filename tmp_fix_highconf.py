from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


replace_exact(
    "scripts/run_gated_clean_delta_experiment.py",
    'DEFAULT_OUTPUT_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated"',
    'DEFAULT_OUTPUT_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"',
)
replace_exact(
    "scripts/run_gated_clean_delta_experiment.py",
    '"type": "tiered",',
    '"type": "high_confidence" if args.gate_medium_scale <= 0.0 else "tiered",',
)
replace_exact(
    "scripts/run_gated_clean_delta_experiment.py",
    'parser.add_argument("--gate-low", type=float, default=0.18)',
    'parser.add_argument("--gate-low", type=float, default=0.50)',
)
replace_exact(
    "scripts/run_gated_clean_delta_experiment.py",
    'parser.add_argument("--gate-medium-scale", type=float, default=0.50)',
    'parser.add_argument("--gate-medium-scale", type=float, default=0.0)',
)
replace_exact(
    "scripts/run_gated_clean_delta_experiment.py",
    'parser.add_argument("--protocol-output", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_protocol.json"))',
    'parser.add_argument("--protocol-output", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_protocol.json"))',
)


def rewrite_grid(path: str, protocol_name: str, output_name: str, cross_seed: bool) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        'COLUMNS = [\n'
        '    ("source", "source"),\n'
        '    ("support_v3_controller_rmsgap", "M0 default"),\n'
        '    ("support_v3_controller_rmsgap_completion_clean_delta", "M1 clean_delta"),\n'
        '    ("support_v3_controller_rmsgap_completion_clean_delta_gated", "M2 gated"),\n'
        ']\n',
        'DEFAULT_GATED_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"\n\n'
        'BASE_COLUMNS = [\n'
        '    ("source", "source"),\n'
        '    ("support_v3_controller_rmsgap", "M0 default"),\n'
        '    ("support_v3_controller_rmsgap_completion_clean_delta", "M1 clean_delta"),\n'
        ']\n',
    )
    text = text.replace('[(args.gated_method, M2 highconf)]', '[(args.gated_method, "M2 highconf")]')
    text = text.replace('len(COLUMNS)', 'len(columns)')
    text = text.replace('enumerate(COLUMNS)', 'enumerate(columns)')
    if cross_seed:
        text = text.replace(
            'parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_seeds10_11_12_protocol.json"))',
            'parser.add_argument("--gated-method", default=DEFAULT_GATED_METHOD)\n'
            f'    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/{protocol_name}"))',
        )
        text = text.replace(
            'parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/removal_completion_clean_delta_gated_seeds10_11_12_grid.png"))',
            f'parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/{output_name}"))',
        )
        text = text.replace(
            'args = parser.parse_args()\n\n    seeds =',
            'args = parser.parse_args()\n    columns = BASE_COLUMNS + [(args.gated_method, "M2 highconf")]\n\n    seeds =',
        )
    else:
        if 'columns = BASE_COLUMNS + [(args.gated_method, "M2 highconf")]' not in text:
            text = text.replace(
                'gates = load_gate(args.protocol)\n',
                'gates = load_gate(args.protocol)\n    columns = BASE_COLUMNS + [(args.gated_method, "M2 highconf")]\n',
            )
        text = text.replace(
            'parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_protocol.json"))',
            'parser.add_argument("--gated-method", default=DEFAULT_GATED_METHOD)\n'
            f'    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/{protocol_name}"))',
        )
        text = text.replace(
            'parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/removal_completion_clean_delta_gated_seed10_grid.png"))',
            f'parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/{output_name}"))',
        )
    p.write_text(text, encoding="utf-8")


rewrite_grid(
    "scripts/make_gated_clean_delta_cross_seed_grid.py",
    "removal_completion_clean_delta_gated_highconf_seeds10_11_12_protocol.json",
    "removal_completion_clean_delta_gated_highconf_seeds10_11_12_grid.png",
    True,
)
rewrite_grid(
    "scripts/make_gated_clean_delta_grid.py",
    "removal_completion_clean_delta_gated_highconf_protocol.json",
    "removal_completion_clean_delta_gated_highconf_seed10_grid.png",
    False,
)

replace_exact(
    "scripts/analyze_reliability_vs_gain.py",
    'METHOD_GATED = "support_v3_controller_rmsgap_completion_clean_delta_gated"',
    'DEFAULT_METHOD_GATED = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"',
)
replace_exact(
    "scripts/analyze_reliability_vs_gain.py",
    'parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_seeds10_11_12_protocol.json"))',
    'parser.add_argument("--gated-method", default=DEFAULT_METHOD_GATED)\n'
    '    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_seeds10_11_12_protocol.json"))',
)
replace_exact(
    "scripts/analyze_reliability_vs_gain.py",
    'parser.add_argument("--output-csv", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_seed10_11_12.csv"))',
    'parser.add_argument("--output-csv", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_highconf_seed10_11_12.csv"))',
)
replace_exact(
    "scripts/analyze_reliability_vs_gain.py",
    'parser.add_argument("--output-scatter", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_scatter.png"))',
    'parser.add_argument("--output-scatter", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_highconf_scatter.png"))',
)
replace_exact(
    "scripts/analyze_reliability_vs_gain.py",
    'gated = load_rgb(args.root / task / METHOD_GATED / f"seed_{seed}" / "result.png")',
    'gated = load_rgb(args.root / task / args.gated_method / f"seed_{seed}" / "result.png")',
)
