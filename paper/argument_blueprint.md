# DeCE-RF Argument Blueprint

## Central Thesis

DeCE-RF argues that localized Rectified Flow editing should be treated as
decoupled clean-estimate displacement control over a source-conditioned
trajectory, because direct target velocity entangles local semantic change with
global source drift, while the RF linear path lets us decompose the desired
clean displacement into edit and preserve components and map it back to
velocity.

## Sub-Arguments

### 1. Direct target velocity is the wrong primitive for local editing

- **Claim**: Target-conditioned velocity mixes desired local edits with
  unwanted global regeneration.
- **Evidence needed**: Direct-target baseline images and outside-mask drift
  metrics across the main tasks.
- **Reasoning**: If target guidance improves prompt alignment while increasing
  outside-region change, it demonstrates the edit-preserve coupling that the
  paper addresses.
- **Counter-argument**: Existing masked target guidance may already solve this.
- **Rebuttal**: Compare against local/support-gated variants and emphasize that
  DeCE-RF defines both target matching and source reconstruction in
  clean-estimate space, not only by masking a target velocity.

### 2. Clean-estimate space provides the decoupling space

- **Claim**: Under the RF linear path, DeCE-RF can design a clean displacement
  \(\Delta_t^0=\Delta_t^{\mathrm{edit}}+\Delta_t^{\mathrm{pres}}\) and map it
  back to velocity as \(v_{\mathrm{DeCE}}=v_{\mathrm{src}}-t^{-1}\Delta_t^0\).
- **Evidence needed**: Derivation of \(\Delta_t^{\mathrm{edit}}\),
  \(\Delta_t^{\mathrm{pres}}\), their mapping to \(v_{\mathrm{tar}}-v_{\mathrm{src}}\)
  and \((\hat{x}_{0,\mathrm{src}}-x_s)/t\), and controller diagnostics showing
  edit gap/preserve drift.
- **Reasoning**: The two branches are not independent modules; both are RF
  velocity forms of clean displacement components.
- **Counter-argument**: The displacement formulation may be a post-hoc explanation for an
  engineering system.
- **Rebuttal**: Tie every implemented term to a clean displacement component and
  report ablations that remove preserve displacement, geometry, and feedback
  weights separately.

### 3. Operation-conditioned geometry is necessary but not the central claim

- **Claim**: Support estimation supplies the geometry of the decoupled clean
  displacement: where edit displacement, preserve displacement, and soft
  transitions should act.
- **Evidence needed**: Visualizations of \(M_{\mathrm{edit}}\),
  \(M_{\mathrm{core}}\), \(M_{\mathrm{contact}}\), \(M_{\mathrm{preserve}}\),
  plus comparisons against generic/attention-only support and manual support.
- **Reasoning**: Good control cannot compensate for wrong geometry; therefore
  support quality should be evaluated as a bottleneck, not hidden.
- **Counter-argument**: The method is just a mask heuristic.
- **Rebuttal**: Describe support as a geometry estimator for clean displacement,
  keep `support_v3` as an implementation label only, and include manual support
  as an upper-bound diagnostic.

### 4. Feedback weighting makes the controller closed-loop

- **Claim**: Clean-estimate feedback updates edit and preserve displacement weights
  along the ODE trajectory.
- **Evidence needed**: Curves for edit gap, preserve drift, adaptive edit
  weight, adaptive preserve weight, and projection magnitude.
- **Reasoning**: Fixed weights cannot handle variable target progress and
  preserve drift across timesteps and tasks.
- **Counter-argument**: Adaptive weights may simply tune hyperparameters rather
  than add a principled mechanism.
- **Rebuttal**: Show interpretable weight changes under edit-strength sweeps and
  support perturbations, with improved or stabilized edit-preserve tradeoffs.

## Logical Flow

1. **Problem**: Target velocity couples local edit and source drift.
2. **Observation**: RF linear paths induce clean estimates at every timestep.
3. **Formulation**: Define a clean displacement
   \(\Delta_t^0=\Delta_t^{\mathrm{edit}}+\Delta_t^{\mathrm{pres}}\).
4. **Mapping**: Map clean displacement to RF velocity with
   \(v_{\mathrm{DeCE}}=v_{\mathrm{src}}-t^{-1}\Delta_t^0\).
5. **Geometry**: Estimate displacement geometry from operation-conditioned evidence.
6. **Feedback**: Update displacement weights from edit progress and preserve drift.
7. **Evaluation**: Separate displacement, geometry, and feedback through ablations.

## Argument Strength Assessment

| Sub-Argument | Evidence Strength | Logic Validity | Counter-Arg Risk |
| --- | --- | --- | --- |
| Direct target velocity is coupled | Moderate; needs final metric table | Strong | Medium |
| Clean displacement unifies branches | Strong derivation; needs empirical linkage | Strong | Medium |
| Operation geometry is necessary | Moderate; depends on support visual evidence | Valid with scoped claim | High |
| Feedback weighting is closed-loop | Moderate; needs trajectory curves | Valid if diagnostics are clear | Medium |

## Notes for Drafting

- Do not frame the novelty as a generic velocity split.
- Lead with \(v_{\mathrm{DeCE}}=v_{\mathrm{src}}-t^{-1}\Delta_t^0\) and
  \(\Delta_t^0=\Delta_t^{\mathrm{edit}}+\Delta_t^{\mathrm{pres}}\).
- Use `operation-conditioned control geometry` in method-facing prose.
- Use `support_v3` only for implementation or experiment labels.
- Present auxiliary text, color, reference, and local-target guidance as
  edit-side energy instantiations, not as core contributions.
- Keep the claim conservative: DeCE-RF improves the edit-preserve tradeoff under
  reasonable support; support quality remains the primary bottleneck.
