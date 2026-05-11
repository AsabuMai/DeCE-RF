from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
import PIL.Image
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.training_utils import set_seed

from sd3_hrec import HRecSD3Edit


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
    parser.add_argument("--edit-color-target-chroma-scale", type=float, default=1.0)
    parser.add_argument("--edit-color-smooth-kernel", type=int, default=5)
    parser.add_argument("--edit-color-luma-preserve-scale", type=float, default=0.35)
    parser.add_argument("--edit-color-luma-gradient-preserve-scale", type=float, default=0.15)
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
    parser.add_argument("--adaptive-preserve-drift-budget", type=float, default=0.0)
    parser.add_argument("--adaptive-edit-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-preserve-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-edit-weight-min", type=float, default=0.7)
    parser.add_argument("--adaptive-edit-weight-max", type=float, default=1.8)
    parser.add_argument("--adaptive-preserve-weight-min", type=float, default=1.0)
    parser.add_argument("--adaptive-preserve-weight-max", type=float, default=2.0)
    parser.add_argument("--adaptive-projection-scale", type=float, default=0.0)
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
            "auto",
            "operation_default",
            "score_auto",
        ),
        default=None,
        help="Operation-aware support candidate. Overrides --support-score when set.",
    )
    parser.add_argument(
        "--edit-operation",
        choices=("auto", "add_object", "add_decal", "remove_object", "replace"),
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
        choices=("auto", "none", "above_host", "on_face", "on_surface", "remove_source_object", "inside_host"),
        default="auto",
        help="Operation-level relation/layout proposal used by operation_support_v3.",
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
        "--edit-mask-smooth-kernel",
        type=int,
        default=0,
        help="Optionally smooth the attention-derived edit/core masks before preserve gating.",
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


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def make_photorealistic_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    style_cues = (
        "photo",
        "photograph",
        "photorealistic",
        "realistic",
        "natural lighting",
        "high-resolution",
    )
    if any(cue in lowered for cue in style_cues):
        return prompt
    return (
        f"A realistic photograph of {prompt.rstrip('.').lower()}, "
        "natural lighting, high-resolution photo."
    )


def parse_normalized_box(value: str | None, arg_name: str = "--edit-mask-box") -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{arg_name} must be formatted as x0,y0,x1,y1")
    box = tuple(float(part) for part in parts)
    if any(coord < 0.0 or coord > 1.0 for coord in box):
        raise ValueError(f"{arg_name} coordinates must be in [0, 1]")
    return box


def parse_word_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    words = [word.strip().lower() for word in value.split(",") if word.strip()]
    return words if words else None


def clamp_normalized_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    return x0, y0, x1, y1


def expand_normalized_box(
    box: tuple[float, float, float, float],
    min_width: float,
    min_height: float,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = clamp_normalized_box(box)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    width = max(x1 - x0 + 2.0 * pad_x, min_width)
    height = max(y1 - y0 + 2.0 * pad_y, min_height)
    return clamp_normalized_box((cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height))


def _preprocess_source_image(image_path: str, max_image_size: int) -> PIL.Image.Image:
    image = PIL.Image.open(image_path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        resized_w = max(16, int(round(image.width * scale)))
        resized_h = max(16, int(round(image.height * scale)))
        image = image.resize((resized_w, resized_h), PIL.Image.Resampling.LANCZOS)
    return image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))


def estimate_dark_eye_structure_boxes(
    image_path: str,
    max_image_size: int,
) -> tuple[dict[str, tuple[float, float, float, float] | None], dict[str, object]]:
    image = _preprocess_source_image(image_path, max_image_size)
    rgb = np.asarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y0_lim = int(0.18 * h)
    y1_lim = int(0.72 * h)
    x0_lim = int(0.08 * w)
    x1_lim = int(0.92 * w)
    roi = gray[y0_lim:y1_lim, x0_lim:x1_lim]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    dark_threshold = min(95.0, max(25.0, float(np.percentile(blur, 18))))
    binary = (blur <= dark_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates: list[dict[str, float]] = []
    image_area = float(h * w)
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        area_ratio = float(area) / image_area
        if area_ratio < 0.00025 or area_ratio > 0.018:
            continue
        if bw <= 2 or bh <= 2:
            continue
        cx = (x0_lim + float(centroids[idx][0])) / w
        cy = (y0_lim + float(centroids[idx][1])) / h
        if cy < 0.33 or cy > 0.62:
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 0.25 or aspect > 4.5:
            continue
        candidates.append(
            {
                "x0": (x0_lim + x) / w,
                "y0": (y0_lim + y) / h,
                "x1": (x0_lim + x + bw) / w,
                "y1": (y0_lim + y + bh) / h,
                "cx": cx,
                "cy": cy,
                "area_ratio": area_ratio,
                "score": area_ratio * (1.0 - abs(cy - 0.45)),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_pair = None
    best_pair_score = -1.0
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            a, b = sorted((left, right), key=lambda item: item["cx"])
            dx = b["cx"] - a["cx"]
            dy = abs(b["cy"] - a["cy"])
            if dx < 0.10 or dx > 0.48 or dy > 0.16:
                continue
            center_y = 0.5 * (a["cy"] + b["cy"])
            pair_score = a["score"] + b["score"] - 0.4 * dy - 0.2 * abs(center_y - 0.46)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (a, b)
    if best_pair is not None:
        selected = list(best_pair)
    else:
        selected = candidates[:1]
    if not selected:
        fallback_boxes, fallback_meta = estimate_foreground_head_structure_boxes(image_path, max_image_size)
        fallback_meta.update(
            {
                "structure_dark_eye_candidates": len(candidates),
                "structure_primary_mode": "dark_eyes",
            }
        )
        return fallback_boxes, fallback_meta
    anchor = clamp_normalized_box(
        (
            min(item["x0"] for item in selected),
            min(item["y0"] for item in selected),
            max(item["x1"] for item in selected),
            max(item["y1"] for item in selected),
        )
    )
    accessory_anchor = clamp_normalized_box(
        (
            anchor[0] - 0.11,
            anchor[1] - 0.050,
            anchor[2] - 0.025,
            anchor[3] - 0.005,
        )
    )
    edit = expand_normalized_box(accessory_anchor, min_width=0.44, min_height=0.14, pad_x=0.02, pad_y=0.008)
    source_inject = expand_normalized_box(accessory_anchor, min_width=0.72, min_height=0.30, pad_x=0.12, pad_y=0.08)
    ex0, ey0, ex1, ey1 = edit
    width = ex1 - ex0
    preserve = clamp_normalized_box(
        (
            ex0 + 0.08 * width,
            min(1.0, min(ey1 + 0.005, accessory_anchor[1] + 0.12)),
            ex1 - 0.02 * width,
            min(1.0, min(ey1 + 0.005, accessory_anchor[1] + 0.12) + 0.30),
        )
    )
    meta = {
        "structure_found": True,
        "structure_mode": "dark_eyes",
        "structure_candidates": len(candidates),
        "structure_selected": len(selected),
        "structure_dark_threshold": dark_threshold,
        "structure_anchor_box": list(anchor),
        "structure_accessory_anchor_box": list(accessory_anchor),
        "structure_edit_mask_box": list(edit),
        "structure_source_inject_mask_box": list(source_inject),
        "structure_preserve_box": list(preserve),
    }
    if len(selected) >= 2:
        left, right = sorted(selected[:2], key=lambda item: item["cx"])
        angle = np.degrees(np.arctan2((right["cy"] - left["cy"]) * h, (right["cx"] - left["cx"]) * w))
        meta["structure_eye_line_angle_deg"] = float(angle)
    else:
        meta["structure_eye_line_angle_deg"] = 0.0
    return {"edit": edit, "source_inject": source_inject, "preserve": preserve, "anchor": anchor}, meta


def estimate_foreground_head_structure_boxes(
    image_path: str,
    max_image_size: int,
) -> tuple[dict[str, tuple[float, float, float, float] | None], dict[str, object]]:
    image = _preprocess_source_image(image_path, max_image_size)
    rgb = np.asarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)

    # Weight-free foreground proposal for animal photos: keep dark/non-green
    # regions and orange/brown saturated regions, then use the largest compact
    # component as the subject support. This is deliberately conservative and
    # only used when eye-pair detection fails.
    greenish = (hue >= 35.0) & (hue <= 95.0) & (sat >= 35.0)
    foreground = (((val < 135.0) & ~greenish) | ((sat > 45.0) & ~greenish)).astype(np.uint8)
    foreground[: int(0.05 * h), :] = 0
    foreground[int(0.92 * h) :, :] = 0
    kernel = np.ones((5, 5), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    components: list[dict[str, float]] = []
    image_area = float(h * w)
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        area_ratio = float(area) / image_area
        if area_ratio < 0.015 or area_ratio > 0.75:
            continue
        if bw < 0.08 * w or bh < 0.08 * h:
            continue
        components.append(
            {
                "x0": x / w,
                "y0": y / h,
                "x1": (x + bw) / w,
                "y1": (y + bh) / h,
                "area_ratio": area_ratio,
                "label": float(idx),
            }
        )
    components.sort(key=lambda item: item["area_ratio"], reverse=True)
    if not components:
        return (
            {"edit": None, "source_inject": None, "preserve": None, "anchor": None},
            {"structure_found": False, "structure_candidates": 0, "structure_mode": "foreground_head"},
        )

    subject = components[0]
    label_id = int(subject["label"])
    x0 = int(round(subject["x0"] * w))
    y0 = int(round(subject["y0"] * h))
    x1 = int(round(subject["x1"] * w))
    y1 = int(round(subject["y1"] * h))
    subject_mask = labels == label_id
    top_y0 = y0
    top_y1 = min(y1, y0 + int(round(0.55 * max(1, y1 - y0))))
    mid_x = x0 + int(round(0.5 * max(1, x1 - x0)))
    left_mass = int(subject_mask[top_y0:top_y1, x0:mid_x].sum())
    right_mass = int(subject_mask[top_y0:top_y1, mid_x:x1].sum())
    head_on_left = left_mass >= right_mass

    sx0, sy0, sx1, sy1 = clamp_normalized_box((subject["x0"], subject["y0"], subject["x1"], subject["y1"]))
    sw = max(1e-6, sx1 - sx0)
    sh = max(1e-6, sy1 - sy0)
    if head_on_left:
        head = clamp_normalized_box((sx0, sy0 + 0.20 * sh, sx0 + 0.46 * sw, sy0 + 0.52 * sh))
    else:
        head = clamp_normalized_box((sx1 - 0.46 * sw, sy0 + 0.20 * sh, sx1, sy0 + 0.52 * sh))

    edit = expand_normalized_box(head, min_width=0.28, min_height=0.13, pad_x=0.015, pad_y=0.01)
    source_inject = expand_normalized_box(head, min_width=0.46, min_height=0.24, pad_x=0.08, pad_y=0.06)
    ex0, ey0, ex1, ey1 = edit
    ew = max(1e-6, ex1 - ex0)
    preserve = clamp_normalized_box(
        (
            ex0 + 0.08 * ew,
            min(1.0, ey1 + 0.035),
            ex1 - 0.02 * ew,
            min(1.0, ey1 + 0.30),
        )
    )
    meta = {
        "structure_found": True,
        "structure_mode": "foreground_head",
        "structure_candidates": len(components),
        "structure_selected": 1,
        "structure_subject_box": [float(subject["x0"]), float(subject["y0"]), float(subject["x1"]), float(subject["y1"])],
        "structure_head_side": "left" if head_on_left else "right",
        "structure_head_box": list(head),
        "structure_edit_mask_box": list(edit),
        "structure_source_inject_mask_box": list(source_inject),
        "structure_preserve_box": list(preserve),
    }
    return {"edit": edit, "source_inject": source_inject, "preserve": preserve, "anchor": head}, meta


def save_structure_glasses_mask(
    image_path: str,
    max_image_size: int,
    edit_box: tuple[float, float, float, float],
    output_path: str,
    angle_deg: float = 0.0,
) -> str:
    image = _preprocess_source_image(image_path, max_image_size)
    w, h = image.size
    x0, y0, x1, y1 = clamp_normalized_box(edit_box)
    width = x1 - x0
    height = y1 - y0
    mask = np.zeros((h, w), dtype=np.float32)

    def box_to_pixels(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        bx0, by0, bx1, by1 = clamp_normalized_box(box)
        ix0 = max(0, min(w, int(round(bx0 * w))))
        ix1 = max(0, min(w, int(round(bx1 * w))))
        iy0 = max(0, min(h, int(round(by0 * h))))
        iy1 = max(0, min(h, int(round(by1 * h))))
        return ix0, iy0, ix1, iy1

    def add_box(box: tuple[float, float, float, float], value: float = 1.0) -> None:
        ix0, iy0, ix1, iy1 = box_to_pixels(box)
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = np.maximum(mask[iy0:iy1, ix0:ix1], value)

    def add_ellipse(box: tuple[float, float, float, float], value: float = 1.0) -> None:
        ix0, iy0, ix1, iy1 = box_to_pixels(box)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        center = (int(round(0.5 * (ix0 + ix1 - 1))), int(round(0.5 * (iy0 + iy1 - 1))))
        axes = (max(1, int(round(0.5 * (ix1 - ix0)))), max(1, int(round(0.5 * (iy1 - iy0)))))
        layer = np.zeros_like(mask)
        cv2.ellipse(layer, center, axes, float(angle_deg), 0, 360, float(value), thickness=-1)
        np.maximum(mask, layer, out=mask)

    def add_rotated_bridge(box: tuple[float, float, float, float], value: float = 0.85) -> None:
        ix0, iy0, ix1, iy1 = box_to_pixels(box)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        rect = (
            (0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1)),
            (max(1.0, float(ix1 - ix0)), max(1.0, float(iy1 - iy0))),
            float(angle_deg),
        )
        points = cv2.boxPoints(rect).round().astype(np.int32)
        layer = np.zeros_like(mask)
        cv2.fillConvexPoly(layer, points, float(value))
        np.maximum(mask, layer, out=mask)

    split = x0 + 0.5 * width
    gap = 0.035 * width
    lens_y0 = y0 + 0.12 * height
    lens_y1 = y1 - 0.02 * height
    add_ellipse((x0 + 0.02 * width, lens_y0, split - gap, lens_y1))
    add_ellipse((split + gap, lens_y0, x1 - 0.02 * width, lens_y1))
    add_rotated_bridge((split - 0.055 * width, y0 + 0.30 * height, split + 0.055 * width, y0 + 0.52 * height), 0.85)

    blur = max(3, int(round(min(w, h) * 0.006)) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    if float(mask.max()) > 1e-6:
        mask = mask / float(mask.max())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PIL.Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype("uint8"), mode="L").save(output_path)
    return output_path


def _normalize01_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - float(values.min())) / max(float(values.max() - values.min()), 1e-6)


def _otsu_threshold(values: np.ndarray, floor: float) -> float:
    values_u8 = np.clip(values.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(threshold) / 255.0, float(floor))


def _keep_top_mask_components(
    binary: np.ndarray,
    score: np.ndarray,
    keep: int,
    min_area: int,
) -> np.ndarray:
    if keep <= 0:
        return binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    components: list[tuple[float, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        components.append((float(score[component].mean()) * np.sqrt(float(area)), label))
    components.sort(reverse=True)
    out = np.zeros_like(binary, dtype=np.uint8)
    for _, label in components[:keep]:
        out[labels == label] = 1
    return out


def build_proposal_diff_mask(
    source_image_path: str,
    proposal_image_path: str,
    max_image_size: int,
    output_path: str,
    threshold: float = 0.22,
    keep_components: int = 2,
    min_area: int = 24,
    dilate: int = 9,
    erode: int = 0,
    blur: int = 17,
    dark_bias: float = 1.0,
) -> tuple[str, dict[str, object]]:
    source = _preprocess_source_image(source_image_path, max_image_size)
    proposal_paths = [path.strip() for path in proposal_image_path.split(",") if path.strip()]
    if not proposal_paths:
        raise ValueError("--proposal-edit-image must contain at least one image path")
    src = np.asarray(source, dtype=np.uint8)
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_luma = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    proposal_scores: list[np.ndarray] = []
    for proposal_path in proposal_paths:
        proposal = PIL.Image.open(proposal_path).convert("RGB").resize(source.size, PIL.Image.Resampling.LANCZOS)
        prop = np.asarray(proposal, dtype=np.uint8)
        prop_lab = cv2.cvtColor(prop, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_diff = _normalize01_array(np.linalg.norm(prop_lab - src_lab, axis=-1))
        prop_luma = cv2.cvtColor(prop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        darker = np.clip(src_luma - prop_luma, 0.0, 1.0)
        proposal_scores.append(_normalize01_array(lab_diff + float(dark_bias) * darker))
    if len(proposal_scores) == 1:
        score = proposal_scores[0]
    else:
        # Consensus suppresses model-specific pose/lighting drift and keeps
        # changes shared by independent edit proposals.
        score = np.minimum.reduce(proposal_scores)

    auto_threshold = _otsu_threshold(score.reshape(-1), threshold)
    binary = (score >= auto_threshold).astype(np.uint8)
    if erode > 0:
        binary = cv2.erode(binary, np.ones((erode, erode), dtype=np.uint8), iterations=1)
    binary = _keep_top_mask_components(binary, score, keep=keep_components, min_area=min_area)
    if dilate > 0:
        binary = cv2.dilate(binary, np.ones((dilate, dilate), dtype=np.uint8), iterations=1)

    soft = binary.astype(np.float32)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        soft = cv2.GaussianBlur(soft, (blur, blur), 0)
        soft = np.clip(soft, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PIL.Image.fromarray((soft * 255.0).round().astype("uint8"), mode="L").save(output_path)
    meta = {
        "proposal_diff_found": bool(float(soft.max()) > 1e-6),
        "proposal_diff_mask": output_path,
        "proposal_edit_image": proposal_image_path,
        "proposal_edit_images": proposal_paths,
        "proposal_diff_consensus_count": len(proposal_paths),
        "proposal_mask_auto_threshold": float(auto_threshold),
        "proposal_mask_mean": float(soft.mean()),
        "proposal_mask_max": float(soft.max()),
        "proposal_mask_keep_components": int(keep_components),
        "proposal_mask_min_area": int(min_area),
        "proposal_mask_dilate": int(dilate),
        "proposal_mask_erode": int(erode),
        "proposal_mask_blur": int(blur),
        "proposal_mask_dark_bias": float(dark_bias),
    }
    return output_path, meta


def load_image_latent(
    pipe,
    image_path: str,
    device: torch.device,
    max_image_size: int,
) -> torch.Tensor:
    image = PIL.Image.open(image_path).convert("RGB")
    if max(image.width, image.height) > max_image_size:
        scale = max_image_size / max(image.width, image.height)
        resized_w = max(16, int(round(image.width * scale)))
        resized_h = max(16, int(round(image.height * scale)))
        image = image.resize((resized_w, resized_h), PIL.Image.Resampling.LANCZOS)
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_src = pipe.image_processor.preprocess(image).to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    return (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def main() -> None:
    run_started_at = time.perf_counter()
    args = build_parser().parse_args()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    edit_mask_box = parse_normalized_box(args.edit_mask_box, "--edit-mask-box")
    edit_mask_exclude_box = parse_normalized_box(args.edit_mask_exclude_box, "--edit-mask-exclude-box")
    source_inject_mask_box = parse_normalized_box(args.source_inject_mask_box, "--source-inject-mask-box")
    final_preserve_box = parse_normalized_box(args.final_preserve_box, "--final-preserve-box")
    external_edit_mask_path = args.external_edit_mask
    external_edit_mask_mode = args.external_edit_mask_mode
    structure_box_metadata: dict[str, object] = {"structure_found": False}
    proposal_mask_metadata: dict[str, object] = {"proposal_diff_found": False}
    if args.auto_structure_boxes:
        structure_boxes, structure_box_metadata = estimate_dark_eye_structure_boxes(
            args.image,
            max_image_size=args.max_image_size,
        )
        if structure_boxes["edit"] is not None and edit_mask_box is None:
            edit_mask_box = structure_boxes["edit"]
            args.auto_local_boxes = False
        if structure_boxes["source_inject"] is not None and source_inject_mask_box is None:
            source_inject_mask_box = structure_boxes["source_inject"]
        if structure_boxes["preserve"] is not None:
            if edit_mask_exclude_box is None:
                edit_mask_exclude_box = structure_boxes["preserve"]
            if final_preserve_box is None:
                final_preserve_box = structure_boxes["preserve"]
        if (
            args.auto_structure_external_mask
            and structure_boxes["edit"] is not None
            and external_edit_mask_path is None
        ):
            if args.mask_output_dir is None:
                raise ValueError("--auto-structure-external-mask requires --mask-output-dir")
            external_edit_mask_path = save_structure_glasses_mask(
                args.image,
                args.max_image_size,
                structure_boxes["edit"],
                os.path.join(args.mask_output_dir, "structure_glasses_edit_mask.png"),
                angle_deg=(
                    0.0
                    if args.structure_glasses_angle_mode == "zero"
                    else max(
                        -abs(args.structure_glasses_max_angle),
                        min(
                            abs(args.structure_glasses_max_angle),
                            float(structure_box_metadata.get("structure_eye_line_angle_deg", 0.0)),
                        ),
                    )
                ),
            )
            external_edit_mask_mode = "replace"
            structure_box_metadata["structure_external_edit_mask"] = external_edit_mask_path
        print(f"[structure] {json.dumps(structure_box_metadata, sort_keys=True)}")
    resolved_object_mask_provider = args.object_mask_provider
    if args.support_mode == "operation_v3":
        resolved_object_mask_provider = "operation_support_v3"
    elif args.support_mode == "generic" and resolved_object_mask_provider == "operation_support_v3":
        resolved_object_mask_provider = "generic_support"
    if resolved_object_mask_provider == "auto":
        resolved_object_mask_provider = "proposal_diff" if args.proposal_edit_image else "attention_velocity"
    if resolved_object_mask_provider == "proposal_diff":
        if not args.proposal_edit_image:
            raise ValueError("--object-mask-provider proposal_diff requires --proposal-edit-image")
        if external_edit_mask_path is None:
            if args.mask_output_dir is None:
                raise ValueError("--object-mask-provider proposal_diff requires --mask-output-dir")
            external_edit_mask_path, proposal_mask_metadata = build_proposal_diff_mask(
                source_image_path=args.image,
                proposal_image_path=args.proposal_edit_image,
                max_image_size=args.max_image_size,
                output_path=os.path.join(args.mask_output_dir, "proposal_diff_edit_mask.png"),
                threshold=args.proposal_mask_threshold,
                keep_components=args.proposal_mask_keep_components,
                min_area=args.proposal_mask_min_area,
                dilate=args.proposal_mask_dilate,
                erode=args.proposal_mask_erode,
                blur=args.proposal_mask_blur,
                dark_bias=args.proposal_mask_dark_bias,
            )
            external_edit_mask_mode = "replace"
            proposal_mask_metadata["proposal_diff_found"] = True
        print(f"[proposal_diff] {json.dumps(proposal_mask_metadata, sort_keys=True)}")
    attention_mask_target_words = parse_word_list(args.attention_mask_target_words)
    attention_mask_source_words = parse_word_list(args.attention_mask_source_words)
    set_seed(args.seed)

    source_prompt = args.source_prompt
    target_prompt = args.prompt
    if args.photo_prompt_mode in {"source", "both"}:
        source_prompt = make_photorealistic_prompt(source_prompt)
    if args.photo_prompt_mode == "both":
        target_prompt = make_photorealistic_prompt(target_prompt)
    if source_prompt != args.source_prompt or target_prompt != args.prompt:
        print(f"[prompt] source: {source_prompt}")
        print(f"[prompt] target: {target_prompt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    )
    scheduler = pipe.scheduler
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()

    x_src = load_image_latent(
        pipe=pipe,
        image_path=args.image,
        device=device,
        max_image_size=args.max_image_size,
    ).to(device)
    x_tar = HRecSD3Edit(
        pipe=pipe,
        scheduler=scheduler,
        x_src=x_src,
        src_prompt=source_prompt,
        tar_prompt=target_prompt,
        negative_prompt="",
        T_steps=args.num_inference_steps,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        inversion_guidance_scale=args.inversion_guidance_scale,
        base_guidance_scale=args.base_guidance_scale,
        n_max=args.n_max,
        eta=args.eta,
        rec_guidance_scale=args.rec_guidance_scale,
        struct_guidance_scale=args.struct_guidance_scale,
        edit_hedit_guidance_scale=args.edit_hedit_guidance_scale,
        edit_field_mode=args.edit_field_mode,
        edit_src_cfg_scale=args.edit_src_cfg_scale,
        edit_guidance_scale=args.edit_guidance_scale,
        edit_region_guidance_scale=args.edit_region_guidance_scale,
        edit_target_guidance_scale=args.edit_target_guidance_scale,
        edit_source_guidance_scale=args.edit_source_guidance_scale,
        edit_clip_guidance_scale=args.edit_clip_guidance_scale,
        edit_clip_match_base_scale=args.edit_clip_match_base_scale,
        edit_image_tv_scale=args.edit_image_tv_scale,
        edit_text_guidance_scale=args.edit_text_guidance_scale,
        edit_text_source_scale=args.edit_text_source_scale,
        edit_text_core_weight=args.edit_text_core_weight,
        edit_text_subject_weight=args.edit_text_subject_weight,
        identity_break_stop=args.identity_break_stop,
        target_attract_start=args.target_attract_start,
        edit_dds_guidance_scale=args.edit_dds_guidance_scale,
        edit_dds_source_scale=args.edit_dds_source_scale,
        edit_app_guidance_scale=args.edit_app_guidance_scale,
        edit_color_guidance_scale=args.edit_color_guidance_scale,
        edit_color_source=args.edit_color_source,
        edit_color_target=args.edit_color_target,
        edit_color_mask_path=args.edit_color_mask_image or None,
        edit_color_mask_threshold=args.edit_color_mask_threshold,
        edit_color_mask_softness=args.edit_color_mask_softness,
        edit_color_luma_gate_min=args.edit_color_luma_gate_min,
        edit_color_luma_gate_softness=args.edit_color_luma_gate_softness,
        edit_color_detail_protect_scale=args.edit_color_detail_protect_scale,
        edit_color_detail_protect_threshold=args.edit_color_detail_protect_threshold,
        edit_color_detail_protect_softness=args.edit_color_detail_protect_softness,
        edit_color_target_chroma_scale=args.edit_color_target_chroma_scale,
        edit_color_smooth_kernel=args.edit_color_smooth_kernel,
        edit_color_luma_preserve_scale=args.edit_color_luma_preserve_scale,
        edit_color_luma_gradient_preserve_scale=args.edit_color_luma_gradient_preserve_scale,
        edit_ref_guidance_scale=args.edit_ref_guidance_scale,
        edit_ref_image_path=args.edit_ref_image or None,
        edit_ref_mask_path=args.edit_ref_mask or None,
        edit_ref_structure_image_path=args.edit_ref_structure_image or None,
        edit_ref_chroma_mode=args.edit_ref_chroma_mode,
        edit_ref_chroma_magnitude_scale=args.edit_ref_chroma_magnitude_scale,
        edit_ref_luma_preserve_scale=args.edit_ref_luma_preserve_scale,
        edit_ref_gradient_preserve_scale=args.edit_ref_gradient_preserve_scale,
        edit_ref_darkness_guard_scale=args.edit_ref_darkness_guard_scale,
        edit_ref_darkness_guard_margin=args.edit_ref_darkness_guard_margin,
        edit_ref_smooth_kernel=args.edit_ref_smooth_kernel,
        edit_ref_lowfreq_suppress_kernel=args.edit_ref_lowfreq_suppress_kernel,
        edit_ref_lowfreq_suppress_scale=args.edit_ref_lowfreq_suppress_scale,
        edit_ref_schedule_start=args.edit_ref_schedule_start,
        edit_ref_schedule_stop=args.edit_ref_schedule_stop,
        edit_ref_schedule_power=args.edit_ref_schedule_power,
        edit_ref_max_struct_rms_ratio=args.edit_ref_max_struct_rms_ratio,
        edit_ref_project_struct_conflict=args.edit_ref_project_struct_conflict,
        edit_core_scale=args.edit_core_scale,
        edit_subject_scale=args.edit_subject_scale,
        source_inject_q_scale=args.source_inject_q_scale,
        source_inject_k_scale=args.source_inject_k_scale,
        source_inject_v_scale=args.source_inject_v_scale,
        source_inject_layer_from=args.source_inject_layer_from,
        source_inject_layer_to=args.source_inject_layer_to,
        source_inject_steps=args.source_inject_steps,
        source_inject_mask_mode=args.source_inject_mask_mode,
        source_inject_mask_box=source_inject_mask_box,
        edit_bound_scale=args.edit_bound_scale,
        clip_start_timestep=args.clip_start_timestep,
        clip_stop_timestep=args.clip_stop_timestep,
        preserve_blend_scale=args.preserve_blend_scale,
        preserve_blend_start_timestep=args.preserve_blend_start_timestep,
        alpha_max=args.alpha_max,
        alpha_schedule=args.alpha_schedule,
        beta_max=args.beta_max,
        beta_schedule=args.beta_schedule,
        adaptive_clean_control=args.adaptive_clean_control,
        adaptive_edit_target_progress=args.adaptive_edit_target_progress,
        adaptive_edit_target_rms=args.adaptive_edit_target_rms,
        adaptive_preserve_drift_budget=args.adaptive_preserve_drift_budget,
        adaptive_edit_gain=args.adaptive_edit_gain,
        adaptive_preserve_gain=args.adaptive_preserve_gain,
        adaptive_edit_weight_min=args.adaptive_edit_weight_min,
        adaptive_edit_weight_max=args.adaptive_edit_weight_max,
        adaptive_preserve_weight_min=args.adaptive_preserve_weight_min,
        adaptive_preserve_weight_max=args.adaptive_preserve_weight_max,
        adaptive_projection_scale=args.adaptive_projection_scale,
        velocity_conversion_mode=args.velocity_conversion_mode,
        linear_path_t_min=args.linear_path_t_min,
        rec_stop_timestep=args.rec_stop_timestep,
        trajectory_preserve_scale=args.trajectory_preserve_scale,
        trajectory_subject_preserve_scale=args.trajectory_subject_preserve_scale,
        edit_initial_noise_scale=args.edit_initial_noise_scale,
        edit_initial_noise_region=args.edit_initial_noise_region,
        attention_mask_mode=args.attention_mask_mode,
        attention_mask_target_words=attention_mask_target_words,
        attention_mask_source_words=attention_mask_source_words,
        attention_mask_subject_threshold=args.attention_mask_subject_threshold,
        attention_mask_core_threshold=args.attention_mask_core_threshold,
        attention_mask_max_area_ratio=args.attention_mask_max_area_ratio,
        attention_mask_fallback_threshold=args.attention_mask_fallback_threshold,
        object_mask_provider=resolved_object_mask_provider,
        semantic_base_mask_path=args.semantic_base_mask,
        support_score=args.support_candidate or args.support_score,
        support_edit_operation=args.edit_operation,
        support_relation=args.support_relation,
        support_new_tokens=parse_word_list(args.new_tokens),
        support_host_tokens=parse_word_list(args.host_tokens),
        support_removed_tokens=parse_word_list(args.removed_tokens),
        support_attention_power=args.support_attention_power,
        support_disagreement_power=args.support_disagreement_power,
        support_top_percentile=args.support_top_percentile,
        support_min_area_ratio=args.support_min_area_ratio,
        support_max_area_ratio=args.support_max_area_ratio,
        support_keep_components=args.support_keep_components,
        support_dilate_radius=args.support_dilate_radius,
        support_blur_kernel=args.support_blur_kernel,
        attention_velocity_support_pad_x=args.attention_velocity_support_pad_x,
        attention_velocity_support_pad_y=args.attention_velocity_support_pad_y,
        attention_velocity_support_min_width=args.attention_velocity_support_min_width,
        attention_velocity_support_min_height=args.attention_velocity_support_min_height,
        mask_layering_mode=args.mask_layering_mode,
        mask_object_threshold=args.mask_object_threshold,
        mask_contact_dilate_kernel=args.mask_contact_dilate_kernel,
        mask_contact_scale=args.mask_contact_scale,
        mask_contact_edge_threshold=args.mask_contact_edge_threshold,
        mask_contact_edge_protect_scale=args.mask_contact_edge_protect_scale,
        mask_output_dir=args.mask_output_dir,
        edit_mask_dilate_kernel=args.edit_mask_dilate_kernel,
        edit_mask_smooth_kernel=args.edit_mask_smooth_kernel,
        edit_mask_component_threshold=args.edit_mask_component_threshold,
        edit_mask_keep_components=args.edit_mask_keep_components,
        edit_mask_component_y_min=args.edit_mask_component_y_min,
        edit_mask_component_y_max=args.edit_mask_component_y_max,
        edit_mask_shift_y=args.edit_mask_shift_y,
        edit_mask_shift_x=args.edit_mask_shift_x,
        auto_local_boxes=args.auto_local_boxes,
        auto_box_threshold=args.auto_box_threshold,
        auto_edit_pad_x=args.auto_edit_pad_x,
        auto_edit_pad_y=args.auto_edit_pad_y,
        auto_edit_min_width=args.auto_edit_min_width,
        auto_edit_min_height=args.auto_edit_min_height,
        auto_source_pad_x=args.auto_source_pad_x,
        auto_source_pad_y=args.auto_source_pad_y,
        auto_preserve_pad_x=args.auto_preserve_pad_x,
        auto_preserve_start_offset=args.auto_preserve_start_offset,
        auto_preserve_height=args.auto_preserve_height,
        edit_mask_box=edit_mask_box,
        edit_mask_box_mode=args.edit_mask_box_mode,
        edit_mask_exclude_box=edit_mask_exclude_box,
        edit_mask_use_core_as_subject=args.edit_mask_use_core_as_subject,
        external_edit_mask_path=external_edit_mask_path,
        external_edit_mask_mode=external_edit_mask_mode,
        log_every=args.log_every,
        stats_output_path=args.stats_output,
        mask_blend=args.mask_blend,
        mask_blend_mode=args.mask_blend_mode,
        final_preserve_box=final_preserve_box,
    )

    x_tar_denorm = (x_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image_tar[0].save(args.output)
    metadata_output = args.metadata_output
    if metadata_output is None:
        root, _ = os.path.splitext(args.output)
        metadata_output = f"{root}_metadata.json"
    metadata = {
        "image": args.image,
        "source_prompt": args.source_prompt,
        "effective_source_prompt": source_prompt,
        "target_prompt": args.prompt,
        "effective_target_prompt": target_prompt,
        "output": args.output,
        "stats_output": args.stats_output,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "n_max": args.n_max,
        "src_guidance_scale": args.src_guidance_scale,
        "tar_guidance_scale": args.tar_guidance_scale,
        "inversion_guidance_scale": (
            args.inversion_guidance_scale
            if args.inversion_guidance_scale is not None
            else args.src_guidance_scale
        ),
        "base_guidance_scale": (
            args.base_guidance_scale
            if args.base_guidance_scale is not None
            else args.src_guidance_scale
        ),
        "edit_src_cfg_scale": (
            args.edit_src_cfg_scale
            if args.edit_src_cfg_scale is not None
            else (
                args.base_guidance_scale
                if args.base_guidance_scale is not None
                else args.src_guidance_scale
            )
        ),
        "rec_guidance_scale": args.rec_guidance_scale,
        "struct_guidance_scale": args.struct_guidance_scale,
        "edit_hedit_guidance_scale": args.edit_hedit_guidance_scale,
        "edit_field_mode": args.edit_field_mode,
        "edit_guidance_scale": args.edit_guidance_scale,
        "edit_region_guidance_scale": args.edit_region_guidance_scale,
        "edit_target_guidance_scale": args.edit_target_guidance_scale,
        "edit_source_guidance_scale": args.edit_source_guidance_scale,
        "edit_clip_guidance_scale": args.edit_clip_guidance_scale,
        "edit_text_guidance_scale": args.edit_text_guidance_scale,
        "edit_dds_guidance_scale": args.edit_dds_guidance_scale,
        "edit_app_guidance_scale": args.edit_app_guidance_scale,
        "edit_color_guidance_scale": args.edit_color_guidance_scale,
        "edit_color_source": args.edit_color_source,
        "edit_color_target": args.edit_color_target,
        "edit_color_mask_image": args.edit_color_mask_image or None,
        "edit_color_mask_threshold": args.edit_color_mask_threshold,
        "edit_color_mask_softness": args.edit_color_mask_softness,
        "edit_color_luma_gate_min": args.edit_color_luma_gate_min,
        "edit_color_luma_gate_softness": args.edit_color_luma_gate_softness,
        "edit_color_detail_protect_scale": args.edit_color_detail_protect_scale,
        "edit_color_detail_protect_threshold": args.edit_color_detail_protect_threshold,
        "edit_color_detail_protect_softness": args.edit_color_detail_protect_softness,
        "edit_color_target_chroma_scale": args.edit_color_target_chroma_scale,
        "edit_color_smooth_kernel": args.edit_color_smooth_kernel,
        "edit_color_luma_preserve_scale": args.edit_color_luma_preserve_scale,
        "edit_color_luma_gradient_preserve_scale": args.edit_color_luma_gradient_preserve_scale,
        "edit_ref_guidance_scale": args.edit_ref_guidance_scale,
        "edit_ref_image": args.edit_ref_image or None,
        "edit_ref_mask": args.edit_ref_mask or None,
        "edit_ref_structure_image": args.edit_ref_structure_image or None,
        "edit_ref_chroma_mode": args.edit_ref_chroma_mode,
        "edit_ref_chroma_magnitude_scale": args.edit_ref_chroma_magnitude_scale,
        "edit_ref_luma_preserve_scale": args.edit_ref_luma_preserve_scale,
        "edit_ref_gradient_preserve_scale": args.edit_ref_gradient_preserve_scale,
        "edit_ref_darkness_guard_scale": args.edit_ref_darkness_guard_scale,
        "edit_ref_darkness_guard_margin": args.edit_ref_darkness_guard_margin,
        "edit_ref_smooth_kernel": args.edit_ref_smooth_kernel,
        "edit_ref_lowfreq_suppress_kernel": args.edit_ref_lowfreq_suppress_kernel,
        "edit_ref_lowfreq_suppress_scale": args.edit_ref_lowfreq_suppress_scale,
        "edit_ref_schedule_start": args.edit_ref_schedule_start,
        "edit_ref_schedule_stop": args.edit_ref_schedule_stop,
        "edit_ref_schedule_power": args.edit_ref_schedule_power,
        "edit_ref_max_struct_rms_ratio": args.edit_ref_max_struct_rms_ratio,
        "edit_ref_project_struct_conflict": args.edit_ref_project_struct_conflict,
        "source_inject_q_scale": args.source_inject_q_scale,
        "source_inject_k_scale": args.source_inject_k_scale,
        "source_inject_v_scale": args.source_inject_v_scale,
        "source_inject_layer_from": args.source_inject_layer_from,
        "source_inject_layer_to": args.source_inject_layer_to,
        "source_inject_steps": args.source_inject_steps,
        "source_inject_mask_mode": args.source_inject_mask_mode,
        "source_inject_mask_box": source_inject_mask_box,
        "alpha_max": args.alpha_max if args.alpha_max is not None else args.rec_guidance_scale,
        "alpha_schedule": args.alpha_schedule,
        "beta_max": args.beta_max if args.beta_max is not None else 1.0,
        "beta_schedule": args.beta_schedule,
        "adaptive_clean_control": args.adaptive_clean_control,
        "adaptive_edit_target_progress": args.adaptive_edit_target_progress,
        "adaptive_edit_target_rms": args.adaptive_edit_target_rms,
        "adaptive_preserve_drift_budget": args.adaptive_preserve_drift_budget,
        "adaptive_edit_gain": args.adaptive_edit_gain,
        "adaptive_preserve_gain": args.adaptive_preserve_gain,
        "adaptive_edit_weight_min": args.adaptive_edit_weight_min,
        "adaptive_edit_weight_max": args.adaptive_edit_weight_max,
        "adaptive_preserve_weight_min": args.adaptive_preserve_weight_min,
        "adaptive_preserve_weight_max": args.adaptive_preserve_weight_max,
        "adaptive_projection_scale": args.adaptive_projection_scale,
        "velocity_conversion_mode": args.velocity_conversion_mode,
        "linear_path_t_min": args.linear_path_t_min,
        "rec_stop_timestep": args.rec_stop_timestep,
        "trajectory_preserve_scale": args.trajectory_preserve_scale,
        "trajectory_subject_preserve_scale": args.trajectory_subject_preserve_scale,
        "edit_initial_noise_scale": args.edit_initial_noise_scale,
        "edit_initial_noise_region": args.edit_initial_noise_region,
        "attention_mask_mode": args.attention_mask_mode,
        "attention_mask_target_words": attention_mask_target_words,
        "attention_mask_source_words": attention_mask_source_words,
        "attention_mask_subject_threshold": args.attention_mask_subject_threshold,
        "attention_mask_core_threshold": args.attention_mask_core_threshold,
        "attention_mask_max_area_ratio": args.attention_mask_max_area_ratio,
        "attention_mask_fallback_threshold": args.attention_mask_fallback_threshold,
        "object_mask_provider": resolved_object_mask_provider,
        "requested_object_mask_provider": args.object_mask_provider,
        "semantic_base_mask": args.semantic_base_mask,
        "support_mode": args.support_mode,
        "support_score": args.support_score,
        "support_candidate": args.support_candidate or args.support_score,
        "edit_operation": args.edit_operation,
        "support_relation": args.support_relation,
        "new_tokens": parse_word_list(args.new_tokens),
        "host_tokens": parse_word_list(args.host_tokens),
        "removed_tokens": parse_word_list(args.removed_tokens),
        "support_attention_power": args.support_attention_power,
        "support_disagreement_power": args.support_disagreement_power,
        "support_top_percentile": args.support_top_percentile,
        "support_min_area_ratio": args.support_min_area_ratio,
        "support_max_area_ratio": args.support_max_area_ratio,
        "support_keep_components": args.support_keep_components,
        "support_dilate_radius": args.support_dilate_radius,
        "support_blur_kernel": args.support_blur_kernel,
        "attention_velocity_support_pad_x": args.attention_velocity_support_pad_x,
        "attention_velocity_support_pad_y": args.attention_velocity_support_pad_y,
        "attention_velocity_support_min_width": args.attention_velocity_support_min_width,
        "attention_velocity_support_min_height": args.attention_velocity_support_min_height,
        "mask_layering_mode": args.mask_layering_mode,
        "mask_object_threshold": args.mask_object_threshold,
        "mask_contact_dilate_kernel": args.mask_contact_dilate_kernel,
        "mask_contact_scale": args.mask_contact_scale,
        "mask_contact_edge_threshold": args.mask_contact_edge_threshold,
        "mask_contact_edge_protect_scale": args.mask_contact_edge_protect_scale,
        "mask_output_dir": args.mask_output_dir,
        "edit_mask_dilate_kernel": args.edit_mask_dilate_kernel,
        "edit_mask_smooth_kernel": args.edit_mask_smooth_kernel,
        "edit_mask_component_threshold": args.edit_mask_component_threshold,
        "edit_mask_keep_components": args.edit_mask_keep_components,
        "edit_mask_component_y_min": args.edit_mask_component_y_min,
        "edit_mask_component_y_max": args.edit_mask_component_y_max,
        "edit_mask_shift_y": args.edit_mask_shift_y,
        "edit_mask_shift_x": args.edit_mask_shift_x,
        "auto_local_boxes": args.auto_local_boxes,
        "auto_box_threshold": args.auto_box_threshold,
        "auto_edit_pad_x": args.auto_edit_pad_x,
        "auto_edit_pad_y": args.auto_edit_pad_y,
        "auto_edit_min_width": args.auto_edit_min_width,
        "auto_edit_min_height": args.auto_edit_min_height,
        "auto_source_pad_x": args.auto_source_pad_x,
        "auto_source_pad_y": args.auto_source_pad_y,
        "auto_preserve_pad_x": args.auto_preserve_pad_x,
        "auto_preserve_start_offset": args.auto_preserve_start_offset,
        "auto_preserve_height": args.auto_preserve_height,
        "auto_structure_boxes": args.auto_structure_boxes,
        "auto_structure_mode": args.auto_structure_mode,
        "auto_structure_external_mask": args.auto_structure_external_mask,
        **structure_box_metadata,
        **proposal_mask_metadata,
        "proposal_edit_image": args.proposal_edit_image,
        "proposal_mask_threshold": args.proposal_mask_threshold,
        "proposal_mask_keep_components": args.proposal_mask_keep_components,
        "proposal_mask_min_area": args.proposal_mask_min_area,
        "proposal_mask_dilate": args.proposal_mask_dilate,
        "proposal_mask_erode": args.proposal_mask_erode,
        "proposal_mask_blur": args.proposal_mask_blur,
        "proposal_mask_dark_bias": args.proposal_mask_dark_bias,
        "edit_mask_box": edit_mask_box,
        "edit_mask_box_mode": args.edit_mask_box_mode,
        "edit_mask_exclude_box": edit_mask_exclude_box,
        "edit_mask_use_core_as_subject": args.edit_mask_use_core_as_subject,
        "external_edit_mask": external_edit_mask_path,
        "external_edit_mask_mode": external_edit_mask_mode,
        "edit_field_mode": args.edit_field_mode,
        "mask_blend": args.mask_blend,
        "mask_blend_mode": args.mask_blend_mode,
        "final_preserve_box": final_preserve_box,
        "photo_prompt_mode": args.photo_prompt_mode,
        "git_commit": current_git_commit(),
        "runtime_seconds": time.perf_counter() - run_started_at,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else None
        ),
    }
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved result to {args.output}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
