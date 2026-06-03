import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone SD3 h-Edit prototype.")
    parser.add_argument("--image", required=True, help="Path to source image.")
    parser.add_argument("--source-prompt", required=True, help="Source prompt.")
    parser.add_argument("--prompt", required=True, help="Target prompt.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=512,
        help="Resize the source image so its longest side is at most this size before VAE encoding.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Prefer sequential CPU offload over model CPU offload for small-memory GPUs.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument(
        "--src-guidance-scale",
        type=float,
        default=1.0,
        help="Legacy source guidance default used by inversion/base unless the split source CFG args are set.",
    )
    parser.add_argument("--tar-guidance-scale", type=float, default=10.5)
    parser.add_argument(
        "--inversion-guidance-scale",
        type=float,
        default=None,
        help="Optional source CFG used only for source inversion. Defaults to --src-guidance-scale.",
    )
    parser.add_argument(
        "--base-guidance-scale",
        type=float,
        default=None,
        help="Optional source CFG used only for the reverse ODE base prior. Defaults to --src-guidance-scale.",
    )
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--rec-guidance-scale", type=float, default=0.0)
    parser.add_argument("--struct-guidance-scale", type=float, default=0.5)
    parser.add_argument("--edit-hedit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-src-cfg-scale", type=float, default=None)
    parser.add_argument("--edit-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-region-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-source-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-clip-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-clip-match-base-scale", type=float, default=0.0)
    parser.add_argument("--edit-image-tv-scale", type=float, default=0.0)
    parser.add_argument("--edit-text-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-text-source-scale", type=float, default=0.8)
    parser.add_argument("--edit-text-core-weight", type=float, default=1.0)
    parser.add_argument("--edit-text-subject-weight", type=float, default=0.3)
    parser.add_argument(
        "--edit-text-source-prompt",
        type=str,
        default=None,
        help="Optional local source/old-object phrase for CLIP text edit guidance.",
    )
    parser.add_argument(
        "--edit-text-target-prompt",
        type=str,
        default=None,
        help="Optional local target/new-object phrase for CLIP text edit guidance.",
    )
    parser.add_argument(
        "--edit-local-target-prompt",
        type=str,
        default=None,
        help="Optional local target prompt used to build a target-formation RF clean prior.",
    )
    parser.add_argument("--edit-local-target-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-local-target-cfg-scale", type=float, default=None)
    parser.add_argument("--identity-break-stop", type=float, default=0.55)
    parser.add_argument("--target-attract-start", type=float, default=0.25)
    parser.add_argument("--edit-dds-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-dds-source-scale", type=float, default=0.8)
    parser.add_argument("--edit-app-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-source", type=str, default=None)
    parser.add_argument("--edit-color-target", type=str, default=None)
    parser.add_argument("--edit-color-mask-image", type=str, default=None)
    parser.add_argument("--edit-color-mask-threshold", type=float, default=0.38)
    parser.add_argument("--edit-color-mask-softness", type=float, default=0.10)
    parser.add_argument("--edit-color-luma-gate-min", type=float, default=0.0)
    parser.add_argument("--edit-color-luma-gate-softness", type=float, default=0.08)
    parser.add_argument("--edit-color-detail-protect-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-detail-protect-threshold", type=float, default=0.35)
    parser.add_argument("--edit-color-detail-protect-softness", type=float, default=0.08)
    parser.add_argument("--edit-color-alpha-matte", action="store_true")
    parser.add_argument("--edit-color-alpha-matte-mode", choices=("color", "closed_form"), default="color")
    parser.add_argument("--edit-color-alpha-matte-kernel-size", type=int, default=9)
    parser.add_argument("--edit-color-alpha-matte-threshold", type=float, default=None)
    parser.add_argument("--edit-color-alpha-matte-softness", type=float, default=None)
    parser.add_argument("--edit-color-alpha-matte-max-size", type=int, default=256)
    parser.add_argument("--edit-color-alpha-matte-epsilon", type=float, default=1e-7)
    parser.add_argument("--edit-color-alpha-matte-constraint-scale", type=float, default=100.0)
    parser.add_argument("--edit-color-target-chroma-scale", type=float, default=1.0)
    parser.add_argument("--edit-color-smooth-kernel", type=int, default=5)
    parser.add_argument("--edit-color-luma-preserve-scale", type=float, default=0.35)
    parser.add_argument("--edit-color-luma-gradient-preserve-scale", type=float, default=0.15)
    parser.add_argument("--edit-color-texture-preserve-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-texture-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-boundary-chroma-scale", type=float, default=0.0)
    parser.add_argument("--edit-color-boundary-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-clean-projection-scale", type=float, default=0.0)
    parser.add_argument(
        "--edit-color-clean-projection-mode",
        choices=("soft", "strict", "yuv_texture"),
        default="soft",
    )
    parser.add_argument("--edit-color-clean-projection-texture-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-clean-projection-luma-texture-scale", type=float, default=1.0)
    parser.add_argument("--edit-color-clean-projection-chroma-texture-scale", type=float, default=0.25)
    parser.add_argument("--edit-color-clean-projection-delta-lowpass-kernel", type=int, default=0)
    parser.add_argument("--edit-color-clean-projection-alpha-power", type=float, default=1.0)
    parser.add_argument("--edit-color-clean-projection-boundary-boost", type=float, default=0.0)
    parser.add_argument("--edit-color-clean-projection-boundary-kernel-size", type=int, default=7)
    parser.add_argument("--edit-color-clean-projection-composite-mode", choices=("blend", "matte"), default="blend")
    parser.add_argument("--edit-color-clean-projection-background-kernel-size", type=int, default=31)
    parser.add_argument("--edit-color-clean-projection-target-mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--edit-color-clean-projection-refresh-interval", type=int, default=0)
    parser.add_argument("--edit-color-texture-restore", action="store_true")
    parser.add_argument("--edit-color-texture-restore-mask", type=str, default=None)
    parser.add_argument("--edit-color-texture-restore-strength", type=float, default=0.8)
    parser.add_argument("--edit-color-texture-restore-kernel-size", type=int, default=9)
    parser.add_argument("--edit-ref-guidance-scale", type=float, default=0.0)
    parser.add_argument("--edit-ref-image", type=str, default=None)
    parser.add_argument("--edit-ref-mask", type=str, default=None)
    parser.add_argument("--edit-ref-structure-image", type=str, default=None)
    parser.add_argument("--edit-ref-chroma-mode", choices=("yuv", "yuv_direction"), default="yuv")
    parser.add_argument("--edit-ref-chroma-magnitude-scale", type=float, default=1.0)
    parser.add_argument("--edit-ref-luma-preserve-scale", type=float, default=0.35)
    parser.add_argument("--edit-ref-gradient-preserve-scale", type=float, default=0.15)
    parser.add_argument("--edit-ref-darkness-guard-scale", type=float, default=0.0)
    parser.add_argument("--edit-ref-darkness-guard-margin", type=float, default=0.03)
    parser.add_argument("--edit-ref-smooth-kernel", type=int, default=1)
    parser.add_argument("--edit-ref-lowfreq-suppress-kernel", type=int, default=0)
    parser.add_argument("--edit-ref-lowfreq-suppress-scale", type=float, default=0.0)
    parser.add_argument("--edit-ref-schedule-start", type=float, default=0.0)
    parser.add_argument("--edit-ref-schedule-stop", type=float, default=0.0)
    parser.add_argument("--edit-ref-schedule-power", type=float, default=1.0)
    parser.add_argument("--edit-ref-max-struct-rms-ratio", type=float, default=0.0)
    parser.add_argument("--edit-ref-project-struct-conflict", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-scale", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-image", type=str, default=None)
    parser.add_argument("--completion-clean-delta-mask", type=str, default=None)
    parser.add_argument("--completion-clean-delta-schedule-start", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-stop", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-power", type=float, default=1.0)
    parser.add_argument("--edit-core-scale", type=float, default=1.35)
    parser.add_argument("--edit-subject-scale", type=float, default=0.35)
    parser.add_argument(
        "--source-inject-q-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental FireFlow-style source visual-Q add strength used only "
            "when evaluating the target velocity."
        ),
    )
    parser.add_argument(
        "--source-inject-k-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental FireFlow-style source visual-K add strength used only "
            "when evaluating the target velocity."
        ),
    )
    parser.add_argument(
        "--source-inject-v-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental FireFlow/RF-Solver-style source visual-V injection "
            "strength used only when evaluating the target velocity."
        ),
    )
    parser.add_argument(
        "--source-inject-layer-from",
        type=int,
        default=-1,
        help="First SD3 transformer block for source V injection. -1 uses the last third.",
    )
    parser.add_argument(
        "--source-inject-layer-to",
        type=int,
        default=-1,
        help="Exclusive SD3 transformer block end for source V injection. -1 uses all later blocks.",
    )
    parser.add_argument(
        "--source-inject-steps",
        type=int,
        default=8,
        help="Number of early reverse-ODE steps that use source V injection. <=0 applies to all active steps.",
    )
    parser.add_argument(
        "--source-inject-mask-mode",
        type=str,
        choices=("none", "edit", "core", "preserve", "box"),
        default="none",
        help="Optional spatial gate for source Q/K/V injection.",
    )
    parser.add_argument(
        "--source-inject-mask-box",
        type=str,
        default=None,
        help="Optional normalized source-injection gate box as x0,y0,x1,y1.",
    )
    parser.add_argument("--edit-bound-scale", type=float, default=0.0)
    parser.add_argument("--clip-start-timestep", type=float, default=0.0)
    parser.add_argument("--clip-stop-timestep", type=float, default=0.6)
    parser.add_argument("--preserve-blend-scale", type=float, default=0.0)
    parser.add_argument("--preserve-blend-start-timestep", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--alpha-schedule", type=str, default="constant")
    parser.add_argument("--beta-max", type=float, default=None)
    parser.add_argument("--beta-schedule", type=str, default="constant")
    parser.add_argument(
        "--adaptive-clean-control",
        action="store_true",
        default=False,
        help="Enable per-step clean-estimate diagnostics and closed-loop local guidance scaling.",
    )
    parser.add_argument(
        "--adaptive-edit-target-progress",
        type=float,
        default=0.0,
        help=(
            "Target directional edit progress in clean-estimate space. "
            "When >0, this supersedes --adaptive-edit-target-rms for edit boost."
        ),
    )
    parser.add_argument("--adaptive-edit-target-rms", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-rmsgap-mode",
        type=str,
        default="legacy",
        choices=["legacy", "normgate"],
    )
    parser.add_argument("--adaptive-rmsgap-dead-zone", type=float, default=0.0)
    parser.add_argument("--adaptive-rmsgap-preserve-gate-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-progress-target", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-progress-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-progress-ema-decay", type=float, default=0.0)
    parser.add_argument("--adaptive-hybrid-preserve-gate-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-drift-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-edit-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-edit-weight-min", type=float, default=0.7)
    parser.add_argument("--adaptive-edit-weight-max", type=float, default=1.8)
    parser.add_argument("--adaptive-preserve-weight-min", type=float, default=1.0)
    parser.add_argument("--adaptive-preserve-weight-max", type=float, default=2.0)
    parser.add_argument("--adaptive-projection-scale", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-clean-correction-scale", type=float, default=0.0)
    parser.add_argument(
        "--removal-controller-mode",
        choices=("none", "clean_fill"),
        default="none",
        help="Removal-specific local clean-fill controller. Disabled unless edit operation is remove_object.",
    )
    parser.add_argument("--removal-fill-scale", type=float, default=0.0)
    parser.add_argument("--removal-suppression-scale", type=float, default=0.0)
    parser.add_argument("--removal-ring-rec-scale", type=float, default=0.0)
    parser.add_argument(
        "--velocity-conversion-mode",
        type=str,
        choices=("legacy", "linear_path"),
        default="linear_path",
        help="How clean-space displacement surrogates are converted into velocity corrections.",
    )
    parser.add_argument(
        "--linear-path-t-min",
        type=float,
        default=0.05,
        help="Minimum denominator for linear-path clean-delta-to-velocity conversion in image runs.",
    )
    parser.add_argument(
        "--rec-stop-timestep",
        type=float,
        default=0.08,
        help="Disable reconstruction guidance below this normalized timestep to avoid low-t blow-up.",
    )
    parser.add_argument(
        "--trajectory-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "Blend preserve regions toward the saved source inversion trajectory "
            "after each reverse ODE step. 0 disables trajectory anchoring."
        ),
    )
    parser.add_argument(
        "--trajectory-subject-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "Additionally blend the attention subject ring (subject minus core) "
            "toward the source trajectory. This helps local edits keep identity "
            "while leaving the high-confidence core editable."
        ),
    )
    parser.add_argument(
        "--edit-initial-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Blend the initial inverted latent toward random noise inside the "
            "attention edit region. Useful for local insertions that need more "
            "generation freedom than source inversion provides."
        ),
    )
    parser.add_argument(
        "--edit-initial-noise-region",
        type=str,
        choices=("core", "subject"),
        default="core",
        help="Which attention mask region receives --edit-initial-noise-scale.",
    )
    parser.add_argument(
        "--region-target-transport-scale",
        type=float,
        default=0.0,
        help=(
            "Enable region-conditioned target transport. Values >0 add a scheduled "
            "target-flow takeover in the edit core and a weaker target blend in the ring."
        ),
    )
    parser.add_argument(
        "--region-target-outside-lock-scale",
        type=float,
        default=0.0,
        help=(
            "After each RF step, blend outside/ring latents toward the saved source "
            "trajectory so stronger core target transport does not drift the background."
        ),
    )
    parser.add_argument(
        "--attention-mask-mode",
        type=str,
        choices=("changed_union", "target_changed", "subject_union", "source_subject", "target_subject"),
        default="changed_union",
        help="Which attention map is converted into the editable mask.",
    )
    parser.add_argument(
        "--attention-mask-target-words",
        type=str,
        default=None,
        help=(
            "Optional comma-separated target prompt words used for token attention. "
            "Defaults to the changed words inferred from source/target prompts."
        ),
    )
    parser.add_argument(
        "--attention-mask-source-words",
        type=str,
        default=None,
        help=(
            "Optional comma-separated source prompt words used for token attention. "
            "Defaults to the changed words inferred from source/target prompts."
        ),
    )
    parser.add_argument(
        "--attention-mask-subject-threshold",
        type=float,
        default=0.48,
        help="Threshold on the normalized attention map for the editable subject mask.",
    )
    parser.add_argument(
        "--attention-mask-core-threshold",
        type=float,
        default=0.72,
        help="Threshold on the normalized attention map for the high-confidence edit core.",
    )
    parser.add_argument(
        "--attention-mask-max-area-ratio",
        type=float,
        default=0.25,
        help=(
            "If the final attention edit mask covers more than this binary area ratio, "
            "shrink it to a conservative high-response target-token box. <=0 disables."
        ),
    )
    parser.add_argument(
        "--attention-mask-fallback-threshold",
        type=float,
        default=0.72,
        help="Normalized high-response threshold used by the large-mask fallback box.",
    )
    parser.add_argument(
        "--object-mask-provider",
        choices=(
            "attention",
            "velocity_diff",
            "attention_velocity",
            "semantic",
            "semantic_velocity",
            "generic_support",
            "operation_support_v3",
            "structure",
            "proposal_diff",
            "auto",
        ),
        default="velocity_diff",
        help=(
            "Source used for the object edit support. 'attention' uses target-token attention "
            "and is image-agnostic; 'velocity_diff' uses this model's target-vs-source RF velocity "
            "difference; 'attention_velocity' fuses changed-token attention with RF velocity difference "
            "for the generic no-oracle path; 'semantic' uses --semantic-base-mask; "
            "'semantic_velocity' refines --semantic-base-mask with RF velocity difference; "
            "'generic_support' uses changed-token attention times clean/velocity disagreement; "
            "'operation_support_v3' uses operation-aware relation/surface/segmentation candidates; "
            "'structure' keeps the optional external structure mask path; "
            "'proposal_diff' extracts changed support from a proposal image; "
            "'auto' uses proposal_diff when --proposal-edit-image is set, otherwise attention_velocity."
        ),
    )
    parser.add_argument(
        "--support-mode",
        choices=("generic", "operation_v3"),
        default=None,
        help="Convenience selector. operation_v3 maps --object-mask-provider to operation_support_v3.",
    )
    parser.add_argument(
        "--semantic-base-mask",
        "--support-mask",
        dest="semantic_base_mask",
        type=str,
        default=None,
        help=(
            "Optional support mask consumed only by object providers semantic and semantic_velocity. "
            "--support-mask is the preferred spelling; --semantic-base-mask is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--support-score",
        choices=(
            "attention_only",
            "clean_disagreement_only",
            "velocity_disagreement_only",
            "attention_x_clean",
            "attention_x_velocity",
            "host_x_clean",
            "new_x_host_x_clean",
            "new_plus_host_x_clean",
            "removed_src_x_clean",
            "removed_src_x_velocity",
            "src_tar_attn_x_clean",
            "seg_x_clean",
            "seg_x_velocity",
            "seg_x_response",
            "relation_x_clean",
            "relation_x_velocity",
            "relation_x_response",
            "host_surface_x_clean",
            "host_surface_x_response",
            "new_x_surface_x_clean",
            "surface_local_clean",
            "surface_local_response",
            "decal_surface_local_response",
            "new_x_surface_local_response",
            "spawn_center",
            "spawn_center_x_response",
            "new_x_spawn_center",
            "spawn_lower_center",
            "spawn_lower_center_x_response",
            "host_spawn_center",
            "host_spawn_center_x_response",
            "host_spawn_wide",
            "host_spawn_wide_x_response",
            "host_top_contact",
            "host_top_contact_x_response",
            "auto",
            "operation_default",
            "score_auto",
        ),
        default="attention_x_clean",
        help="Score/candidate used by generic_support or operation_support_v3.",
    )
    parser.add_argument(
        "--support-candidate",
        choices=(
            "attention_only",
            "clean_disagreement_only",
            "velocity_disagreement_only",
            "attention_x_clean",
            "attention_x_velocity",
            "host_x_clean",
            "new_x_host_x_clean",
            "new_plus_host_x_clean",
            "removed_src_x_clean",
            "removed_src_x_velocity",
            "src_tar_attn_x_clean",
            "seg_x_clean",
            "seg_x_velocity",
            "seg_x_response",
            "relation_x_clean",
            "relation_x_velocity",
            "relation_x_response",
            "host_surface_x_clean",
            "host_surface_x_response",
            "new_x_surface_x_clean",
            "surface_local_clean",
            "surface_local_response",
            "decal_surface_local_response",
            "new_x_surface_local_response",
            "spawn_center",
            "spawn_center_x_response",
            "new_x_spawn_center",
            "spawn_lower_center",
            "spawn_lower_center_x_response",
            "host_spawn_center",
            "host_spawn_center_x_response",
            "host_spawn_wide",
            "host_spawn_wide_x_response",
            "host_top_contact",
            "host_top_contact_x_response",
            "auto",
            "operation_default",
            "score_auto",
        ),
        default=None,
        help="Operation-aware support candidate. Overrides --support-score when set.",
    )
    parser.add_argument(
        "--edit-operation",
        choices=("auto", "add_object", "add_decal", "remove_object", "replace", "recolor"),
        default="auto",
        help="Operation label recorded for operation-aware generic support.",
    )
    parser.add_argument("--new-tokens", type=str, default=None)
    parser.add_argument("--host-tokens", type=str, default=None)
    parser.add_argument("--removed-tokens", type=str, default=None)
    parser.add_argument(
        "--relation",
        "--support-relation",
        dest="support_relation",
        choices=("auto", "none", "above_host", "on_face", "on_surface", "remove_source_object", "inside_host", "inside", "inside_container"),
        default="auto",
        help="Operation-level relation/layout proposal used by operation_support_v3.",
    )
    parser.add_argument(
        "--grounding-method",
        choices=("none", "external_mask", "grounded_sam", "sam", "sam2", "clipseg"),
        default="external_mask",
        help=(
            "Grounding source label for operation_support_v3. Current runtime consumes "
            "--support-mask as the external grounding mask and records this label."
        ),
    )
    parser.add_argument(
        "--save-support-debug",
        action="store_true",
        default=False,
        help="Save operation_support_v3 candidate maps when --mask-output-dir is set.",
    )
    parser.add_argument(
        "--support-debug-only",
        action="store_true",
        default=False,
        help="Stop after automatic support estimation/debug output instead of running the edit ODE.",
    )
    parser.add_argument(
        "--support-temporal-aggregation",
        choices=("single", "mean", "max"),
        default="single",
        help="Aggregate operation_support_v3 clean/velocity evidence over one or multiple ODE timesteps.",
    )
    parser.add_argument("--support-attention-power", type=float, default=1.0)
    parser.add_argument("--support-disagreement-power", type=float, default=1.0)
    parser.add_argument("--support-top-percentile", type=float, default=90.0)
    parser.add_argument("--support-min-area-ratio", type=float, default=0.02)
    parser.add_argument("--support-max-area-ratio", type=float, default=0.30)
    parser.add_argument("--support-keep-components", type=int, default=2)
    parser.add_argument("--support-dilate-radius", type=int, default=5)
    parser.add_argument("--support-blur-kernel", type=int, default=5)
    parser.add_argument("--attention-velocity-support-pad-x", type=float, default=0.015)
    parser.add_argument("--attention-velocity-support-pad-y", type=float, default=0.010)
    parser.add_argument("--attention-velocity-support-min-width", type=float, default=0.18)
    parser.add_argument("--attention-velocity-support-min-height", type=float, default=0.065)
    parser.add_argument(
        "--mask-layering-mode",
        choices=("none", "object_contact"),
        default="object_contact",
        help="Split edit support into object/contact/preserve layers before editing.",
    )
    parser.add_argument(
        "--mask-object-threshold",
        type=float,
        default=0.45,
        help="Threshold used to extract the strong object-edit layer from the edit mask.",
    )
    parser.add_argument(
        "--mask-contact-dilate-kernel",
        type=int,
        default=7,
        help="Dilation kernel used to build the weak contact ring around the object mask.",
    )
    parser.add_argument(
        "--mask-contact-scale",
        type=float,
        default=0.25,
        help="Edit strength multiplier inside the contact ring.",
    )
    parser.add_argument(
        "--mask-contact-edge-threshold",
        type=float,
        default=0.55,
        help="Latent edge threshold for protecting strong source structure inside the contact ring.",
    )
    parser.add_argument(
        "--mask-contact-edge-protect-scale",
        type=float,
        default=0.75,
        help="How strongly source latent edges suppress the contact edit ring.",
    )
    parser.add_argument(
        "--mask-output-dir",
        type=str,
        default=None,
        help="Optional directory for saving attention-mask debug PNGs.",
    )
    parser.add_argument(
        "--edit-mask-dilate-kernel",
        type=int,
        default=0,
        help="Optionally dilate the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-erode-kernel",
        type=int,
        default=0,
        help="Optionally erode the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-smooth-kernel",
        type=int,
        default=0,
        help="Optionally smooth the attention-derived edit/core masks before preserve gating.",
    )
    parser.add_argument(
        "--edit-mask-hole-fraction",
        type=float,
        default=0.0,
        help="Drop a random fraction of edit/core mask pixels for support robustness tests.",
    )
    parser.add_argument(
        "--edit-mask-boundary-noise-scale",
        type=float,
        default=0.0,
        help="Add random perturbation on edit/core mask boundaries for support robustness tests.",
    )
    parser.add_argument(
        "--edit-mask-component-threshold",
        type=float,
        default=0.0,
        help="Threshold used by optional attention-mask connected-component filtering. 0 uses 0.5.",
    )
    parser.add_argument(
        "--edit-mask-keep-components",
        type=int,
        default=0,
        help="Keep only the top-N attention-mask connected components by mask mass. 0 keeps all.",
    )
    parser.add_argument(
        "--edit-mask-component-y-min",
        type=float,
        default=None,
        help="Optional minimum normalized component center y for attention-mask component filtering.",
    )
    parser.add_argument(
        "--edit-mask-component-y-max",
        type=float,
        default=None,
        help="Optional maximum normalized component center y for attention-mask component filtering.",
    )
    parser.add_argument(
        "--edit-mask-shift-y",
        type=float,
        default=0.0,
        help="Translate the attention-derived edit/core masks vertically as a fraction of latent height.",
    )
    parser.add_argument(
        "--edit-mask-shift-x",
        type=float,
        default=0.0,
        help="Translate the attention-derived edit/core masks horizontally as a fraction of latent width.",
    )
    parser.add_argument(
        "--auto-local-boxes",
        action="store_true",
        default=False,
        help=(
            "Derive local edit/source-injection/preserve boxes from the attention mask. "
            "Manual box arguments still override their corresponding auto boxes."
        ),
    )
    parser.add_argument(
        "--auto-box-threshold",
        type=float,
        default=0.35,
        help="Attention-mask threshold used to derive --auto-local-boxes anchors.",
    )
    parser.add_argument("--auto-edit-pad-x", type=float, default=0.08)
    parser.add_argument("--auto-edit-pad-y", type=float, default=0.04)
    parser.add_argument("--auto-edit-min-width", type=float, default=0.28)
    parser.add_argument("--auto-edit-min-height", type=float, default=0.10)
    parser.add_argument("--auto-source-pad-x", type=float, default=0.14)
    parser.add_argument("--auto-source-pad-y", type=float, default=0.08)
    parser.add_argument("--auto-preserve-pad-x", type=float, default=0.04)
    parser.add_argument("--auto-preserve-start-offset", type=float, default=0.06)
    parser.add_argument("--auto-preserve-height", type=float, default=0.24)
    parser.add_argument(
        "--auto-structure-boxes",
        action="store_true",
        default=False,
        help=(
            "Estimate local accessory boxes from source-image structure before SD3 runs. "
            "Currently uses a no-model dark-component heuristic for eye-like regions."
        ),
    )
    parser.add_argument(
        "--auto-structure-mode",
        choices=("dark_eyes",),
        default="dark_eyes",
        help="Image-structure heuristic used by --auto-structure-boxes.",
    )
    parser.add_argument(
        "--auto-structure-external-mask",
        action="store_true",
        default=False,
        help="Use an automatically generated glasses-shaped structure mask as M_edit.",
    )
    parser.add_argument(
        "--structure-glasses-angle-mode",
        choices=("auto", "zero"),
        default="auto",
        help="Use detected eye-line angle for generated glasses masks, or force horizontal.",
    )
    parser.add_argument(
        "--structure-glasses-max-angle",
        type=float,
        default=8.0,
        help="Maximum absolute rotation angle in degrees for generated glasses masks.",
    )
    parser.add_argument(
        "--edit-mask-box",
        type=str,
        default=None,
        help="Optional normalized edit box as x0,y0,x1,y1. Useful for local accessory placement.",
    )
    parser.add_argument(
        "--edit-mask-box-mode",
        type=str,
        choices=("replace", "intersect", "union"),
        default="replace",
        help="How --edit-mask-box combines with the attention-derived edit mask.",
    )
    parser.add_argument(
        "--edit-mask-exclude-box",
        type=str,
        default=None,
        help="Optional normalized box subtracted from the final edit/core masks.",
    )
    parser.add_argument(
        "--edit-mask-use-core-as-subject",
        action="store_true",
        default=False,
        help=(
            "Use the high-confidence core mask as the final M_edit support. "
            "Useful for diagnosing local edits where the subject mask is too broad."
        ),
    )
    parser.add_argument(
        "--external-edit-mask",
        "--final-edit-mask",
        dest="external_edit_mask",
        type=str,
        default=None,
        help=(
            "Optional grayscale image applied late to the final M_edit support. "
            "--final-edit-mask is the preferred spelling; --external-edit-mask is kept for compatibility. "
            "This changes only the mask support; the RF/ODE edit dynamics are unchanged."
        ),
    )
    parser.add_argument(
        "--external-edit-mask-mode",
        "--final-edit-mask-mode",
        dest="external_edit_mask_mode",
        type=str,
        choices=("replace", "intersect", "union"),
        default="replace",
        help="How --final-edit-mask combines with the current edit mask.",
    )
    parser.add_argument(
        "--proposal-edit-image",
        type=str,
        default=None,
        help="Optional candidate edited image used by --object-mask-provider proposal_diff.",
    )
    parser.add_argument("--proposal-mask-threshold", type=float, default=0.22)
    parser.add_argument("--proposal-mask-keep-components", type=int, default=2)
    parser.add_argument("--proposal-mask-min-area", type=int, default=24)
    parser.add_argument("--proposal-mask-dilate", type=int, default=9)
    parser.add_argument("--proposal-mask-erode", type=int, default=0)
    parser.add_argument("--proposal-mask-blur", type=int, default=17)
    parser.add_argument("--proposal-mask-dark-bias", type=float, default=1.0)
    parser.add_argument(
        "--edit-field-mode",
        type=str,
        choices=("surrogate", "rf_diff", "text_diff", "rf_text_diff"),
        default="surrogate",
        help=(
            "Edit correction field used in the controlled ODE. "
            "'rf_diff' uses the target-vs-source RF velocity difference; "
            "'text_diff' uses the source-target CLIP text differential reward; "
            "'rf_text_diff' combines both. 'surrogate' keeps all explicitly "
            "enabled legacy surrogate branches for ablations."
        ),
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--clean-estimate-debug-dir", type=str, default=None)
    parser.add_argument("--stats-output", type=str, default=None)
    parser.add_argument("--metadata-output", type=str, default=None)
    parser.add_argument("--mask-blend", action="store_true", default=False)
    parser.add_argument(
        "--mask-blend-mode",
        choices=("subject", "core"),
        default="subject",
        help="Which final edit support is used when --mask-blend is enabled.",
    )
    parser.add_argument(
        "--final-preserve-box",
        type=str,
        default=None,
        help="Optional normalized box pasted back from the source latent after editing.",
    )
    parser.add_argument(
        "--photo-prompt-mode",
        type=str,
        choices=("off", "source", "both"),
        default="source",
        help=(
            "Optionally rewrite prompts into an explicit photorealistic style "
            "without changing the editing math. 'source' only stabilizes the "
            "base prior; 'both' applies the same style prior to source and target."
        ),
    )
    return parser
