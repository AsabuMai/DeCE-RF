# ODE / Rectified Flow Image Editing: Current Summary

## 1. Research Goal

The current research goal is to migrate the core **reconstruction / editing decoupling** idea from **h-Edit** into an **ODE / Rectified Flow / Flow Matching** framework for image editing.

More specifically, the goal is **not** to reproduce h-Edit itself, and also **not** to study RF image editing in a broad or vague way. Instead, the focus is:

> Can we build a **decoupled editing framework** in Rectified Flow / Flow Matching, where **faithfulness** and **editing effectiveness** are explicitly modeled as two different dynamical components?

The target problem is image editing, not unconditional generation, so the method must simultaneously address:

1. **Faithfulness**: preserve the source image content that should not be changed.
2. **Editing effectiveness**: successfully move the image toward the desired editing target.

---

## 2. Why This Direction Makes Sense

In **h-Edit**, the update is explicitly decomposed into:

- a **reconstruction/base term**
- an **editing term**

This decomposition is important because image editing is naturally a dual-objective problem: the edited result should remain faithful to the source image, while also achieving the target transformation.

Our current idea is to keep this structural insight, but rewrite it in the language of **ODE-based generative dynamics** rather than diffusion bridges or SDE sampling.

So the main difference is:

- **h-Edit**: diffusion / bridge / SDE / Doob's h-transform formulation
- **our current direction**: ODE / Rectified Flow / velocity-field / controlled trajectory formulation

In short, we are not translating h-Edit literally. We are extracting its most important structural principle and reformulating it in an ODE-native way.

---

## 3. Current Mathematical Setup

### 3.1 Linear Path Parameterization

We currently fix the standard linear interpolation path in Rectified Flow:

\[
x_t = (1-t)x_0 + t x_1, \qquad t \in [0,1]
\]

where:

- \(x_0\): clean image
- \(x_1\): noise sample

The ideal path derivative is:

\[
\frac{dx_t}{dt} = x_1 - x_0
\]

The pretrained Rectified Flow model provides a velocity field:

\[
v_\theta(x_t,t)
\]

that approximates this transport direction.

---

### 3.2 Path-Induced Clean Estimate

Under the linear path setting, we define the clean-image estimate as:

\[
\hat x_0(x_t,t) = x_t - t\,v_\theta(x_t,t)
\]

This is one of the key ingredients of the framework. It allows us to interpret the current intermediate trajectory state \(x_t\) in the clean-image space, where editing objectives and reconstruction constraints are easier to define.

This means that reconstruction and editing are not directly defined on \(x_t\), but on the clean-image estimate induced by the current trajectory.

---

## 4. Reconstruction Energy

To preserve the non-edited content of the source image, we define a reconstruction energy:

\[
E_{\mathrm{rec}}(x_t,t)
=
\frac{1}{2}
\left\|
(1-M)\odot\bigl(\hat x_0(x_t,t)-x^{src}\bigr)
\right\|^2
\]

where:

- \(x^{src}\): source image
- \(M\): edit-region mask
- \(1-M\): non-edited region mask

This energy measures how much the current clean-image estimate deviates from the source image on the non-edited region.

The role of this term is to explicitly encode **faithfulness**.

---

## 5. Editing Energy: The Most Important Open Problem

At the current stage, the most critical unresolved component is:

\[
E_{\mathrm{edit}}(x_t,t)
\]

This is the core of the whole method, because it determines:

- what counts as a good edit,
- what direction the trajectory should move toward,
- whether the method behaves like real editing rather than full-image regeneration.

The current most promising first formulation is a **source-target differential editing energy**:

\[
E_{\mathrm{edit}}(x_t,t)
=
R_{\mathrm{text}}\bigl(\hat x_0(x_t,t), c^{edit}\bigr)
-
\lambda_{\mathrm{src}}
R_{\mathrm{text}}\bigl(\hat x_0(x_t,t), c^{src}\bigr)
\]

where:

- \(c^{edit}\): target editing prompt
- \(c^{src}\): source prompt
- \(R_{\mathrm{text}}\): text-image alignment reward

This formulation is currently favored over a pure target-only reward because it better captures the nature of editing as a transition from source semantics to target semantics.

---

## 6. Mathematical Interpretation of Edit

A very important clarification is that **edit is not the prompt itself**, and not even the reward value itself.

Mathematically, edit should be understood as a **control vector field** induced by the editing energy:

\[
E_{\mathrm{edit}}(x_t,t)
\rightarrow
\nabla_{x_t} E_{\mathrm{edit}}(x_t,t)
\rightarrow
v_{\mathrm{edit}}(x_t,t)
\]

So the actual editing effect entering the ODE is:

\[
v_{\mathrm{edit}}(x_t,t)
=
\nabla_{x_t} E_{\mathrm{edit}}(x_t,t)
\]

This means that editing is ultimately modeled as a **target-seeking correction field** on top of the base flow.

---

## 7. Why Decoupling Is Necessary

We have already clarified the theoretical motivation for decoupling reconstruction and editing.

Image editing is inherently a **dual-objective problem**:

- preserve what should stay unchanged,
- modify what should change.

If one uses only a single editing guidance term, such as:

\[
\dot x_t = v_\theta(x_t,t) + \beta(t)\nabla E_{\mathrm{edit}}(x_t,t)
\]

then the dynamics only know how to move toward the target editing objective, but do not explicitly know what content should remain stable.

This can lead to:

- global image drift,
- destruction of non-edited regions,
- unintended changes in background or structure.

Therefore, the motivation for decoupling is:

> A single editing field cannot explicitly encode both semantic transformation and source-image preservation. These two roles should be separated at the dynamical level.

---

## 8. Final ODE Formulation We Currently Prefer

Originally, we considered the expanded three-term form:

\[
\dot x_t = v_\theta - \alpha(t)\nabla E_{\mathrm{rec}} + \beta(t)\nabla E_{\mathrm{edit}}
\]

However, after discussion, we concluded that this form is less faithful to the spirit of h-Edit, because it appears to contain three parallel components.

A more appropriate formulation is to group the original RF velocity and the reconstruction correction into a single **reconstruction-aware base field**:

\[
v_{\mathrm{rec}}(x_t,t)
=
v_\theta(x_t,t)-\alpha(t)\nabla_{x_t}E_{\mathrm{rec}}(x_t,t)
\]

Then define the editing field as:

\[
v_{\mathrm{edit}}(x_t,t)
=
\nabla_{x_t}E_{\mathrm{edit}}(x_t,t)
\]

The preferred total dynamics becomes:

\[
\dot x_t
=
v_{\mathrm{rec}}(x_t,t)
+
\beta(t)\,v_{\mathrm{edit}}(x_t,t)
\]

This is much closer to the structural form of h-Edit:

- one **reconstruction/base term**
- one **editing term**

At the same time, the conceptual decoupling is still important:

- the **reconstruction guidance** is responsible for faithfulness,
- the **editing guidance** is responsible for target transformation.

So even if the RF prior and reconstruction correction are grouped into the reconstruction-aware base field \(v_{\mathrm{rec}}\), the method should still be interpreted as having **two distinct guidance roles** at the dynamical level.

---

## 9. Base Point Interpretation

We also clarified how to interpret the **base point** in the ODE / RF setting.

In h-Edit, the base point is the faithful intermediate point computed before applying the explicit editing update.

In the ODE / RF version, the continuous-time counterpart is the reconstruction-aware base field \(v_{\mathrm{rec}}\). If we discretize one local step with step size \(\Delta t\), then the corresponding base point can be written as:

\[
x_t^{base} = x_t + \Delta t\, v_{\mathrm{rec}}(x_t,t)
\]

This means:

> The base point is the local point that should be reached when one first follows the base generative dynamics while enforcing faithfulness, before explicitly pushing toward the editing target.

Thus, the base point is not chosen arbitrarily; it is induced by the reconstruction-aware dynamics.

---

## 10. Relation to h-Edit

### Common point

Both h-Edit and our current framework share the same key structural idea:

- a part dedicated to reconstruction / faithfulness
- a part dedicated to editing / target transformation

### Difference

But the mathematical objects are different.

**h-Edit** is built on:

- diffusion models,
- reverse-time bridges,
- Doob's h-transform,
- SDE / discrete diffusion sampling,
- \(\nabla \log h\)-based editing.

**Our current framework** is built on:

- ODE / Rectified Flow / Flow Matching,
- velocity fields,
- path-induced clean estimates,
- reconstruction and editing energies,
- controlled trajectory dynamics.

So the most accurate statement is:

> We are not reproducing h-Edit, but reformulating its decoupled editing principle in an ODE-native RF framework.

---

## 11. Current Research Status

At this stage, the work can already be considered a **valid research direction**, because it now has:

- a clear problem definition,
- a meaningful connection to prior work,
- a distinct mathematical reformulation,
- a concrete ODE-based framework skeleton,
- a clear next bottleneck.

However, it is still more accurate to call it a **method framework skeleton** rather than a full method paper.

What we already have:

- the research question,
- the linear path setup,
- the path-induced clean estimate,
- the reconstruction/editing decoupling idea,
- the reconstruction-aware base field,
- the base-point interpretation,
- the relation to h-Edit.

What is still missing:

- a strong and convincing definition of \(E_{\mathrm{edit}}(x_t,t)\),
- a complete algorithm pipeline,
- implementation details,
- experimental validation.

In addition, the current experiments already reveal a specific practical difficulty:

- multiple candidate editing energies can produce visible edit signals,
- but the resulting trajectory can still fall into a **symbolic / overlay-like local optimum**,
- where the source structure is largely preserved but the edited subject looks like a target-shaped visual overlay rather than a fully rewritten realistic object.

So the current bottleneck is not merely that \(E_{\mathrm{edit}}(x_t,t)\) is undefined, but that a convincing editing energy must also avoid this failure mode.

---

## 12. Experimental Goal for the First Version

The first-stage experiment does not need to be very large. The purpose is to verify the central claim:

> Compared with direct editing guidance, the decoupled reconstruction-editing formulation achieves a better trade-off between editing effectiveness and faithfulness in ODE-based Rectified Flow image editing.

A minimum viable experiment should include:

### Task
- text-guided real image editing

### Baselines
1. base flow only
2. direct editing guidance
3. decoupled guidance (our method)

### Metrics
- editing effectiveness
- faithfulness
- efficiency

### Key ablation
- only reconstruction
- only editing
- reconstruction + editing

This is enough to support the necessity of decoupling.

---

## 13. Most Important Next Step

The single most important next step is to define:

\[
E_{\mathrm{edit}}(x_t,t)
\]

in a convincing way.

This is the central unresolved issue because it determines:

- what editing means mathematically,
- how the trajectory is pushed toward the target,
- whether the framework behaves like real editing rather than target-conditioned regeneration.

So from now on, the main focus should be:

> designing and justifying a strong editing energy.

---

## 14. One-Sentence Summary

We are currently building an ODE / Rectified Flow image editing framework that reformulates the h-Edit-style reconstruction/editing decoupling principle in velocity-field dynamics; under the linear path setting, we have already defined a path-induced clean estimate, a reconstruction-aware base field, and an editing field, while the next crucial step is to design a convincing editing energy.
