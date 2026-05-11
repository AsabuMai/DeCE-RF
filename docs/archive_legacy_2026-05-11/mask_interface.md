# Mask Interface

The edit pipeline has three separate mask layers. Keep them distinct when
running experiments.

## 1. Object Support Provider

This selects how the initial object support is estimated.

```bash
--object-mask-provider attention
--object-mask-provider velocity_diff
--object-mask-provider attention_velocity
--object-mask-provider semantic
--object-mask-provider semantic_velocity
--object-mask-provider proposal_diff
```

Use this when the question is: "How should the method discover the edit
support?"

## 2. Support Mask

Preferred CLI:

```bash
--support-mask path/to/mask.png
```

Compatibility name:

```bash
--semantic-base-mask path/to/mask.png
```

This mask is consumed only by:

```bash
--object-mask-provider semantic
--object-mask-provider semantic_velocity
```

It is not a late override. Passing a support mask while using
`attention_velocity` does not mean the final edit region is replaced by that
mask. The shell scripts now reject that combination to avoid accidental no-op
experiments.

## 3. Final Edit Mask

Preferred CLI:

```bash
--final-edit-mask path/to/mask.png
--final-edit-mask-mode replace
```

Compatibility name:

```bash
--external-edit-mask path/to/mask.png
--external-edit-mask-mode replace
```

This is applied late, after the provider-derived `M_edit` has been built.

Modes:

```text
replace   M_edit = final_mask
intersect M_edit = M_edit * final_mask
union     M_edit = max(M_edit, final_mask)
```

Use this when the question is: "Given a trusted mask, can the ODE produce the
edit inside this exact support?"

## 4. Layering

After the final `M_edit` is selected, `--mask-layering-mode object_contact`
can split it into:

```text
core
contact
preserve
structure_edge
```

This changes how edit and preserve forces are distributed around the support.
It does not choose the support source.

## Practical Rules

For automatic semantic support:

```bash
OBJECT_MASK_PROVIDER=semantic_velocity SUPPORT_MASK=mask.png scripts/run_ode_decoupled_edit.sh
```

For trusted exact local masks, such as `profile_eye`:

```bash
FINAL_EDIT_MASK=mask.png FINAL_EDIT_MASK_MODE=replace scripts/run_ode_decoupled_edit.sh
```

For generic no-oracle runs:

```bash
OBJECT_MASK_PROVIDER=attention_velocity scripts/run_ode_decoupled_edit.sh
```

Do not use `SUPPORT_MASK` with `attention_velocity`. Use `FINAL_EDIT_MASK` if
the intent is to force the final edit region.
