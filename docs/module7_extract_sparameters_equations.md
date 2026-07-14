# Module 7 — `extract.sparameters`: Modal Amplitude Extraction, S-Parameters, De-embedding

Companion to the top-level architecture doc and Modules 0–6. Same conventions: no code,
precise equations with derivations, step-by-step build order, validation targets. This module
is purely post-processing — it solves nothing new, only reads out what Module 6 already solved
— but it's where the tool's original motivating physics finally becomes visible: **mode
conversion**, the reason ports were required to capture $\ge2$ modes in the very first design
discussion of this project, is a quantity this module computes and the earlier modules only
made possible.

---

## 0. What Module 7 owns vs. consumes

**Consumes**: Module 4's `PortMode` objects (per port, per tracked mode, per frequency —
$\gamma_m,\mathbf e_m,\mathbf h_m$, already power-normalized, already tracked continuously
across the sweep by Module 6); Module 6's `SweepResult` list (one entry per
(frequency, excitation) pair: the full solved coefficient vector $\mathbf a$ and that
frequency's tracked `port_modes`).

**Owns**: modal amplitude extraction from $\mathbf a$ at each port (§2); raw generalized
S-parameter construction, including the dominant-mode/mode-conversion distinction (§3);
de-embedding (§4, applying Module 4 §6's already-derived formula); the extended
energy-conservation check that properly accounts for mode conversion (§5); final aggregation
into the deliverable $S(\omega)$ dataset (§6).

**Does not own**: anything about solving the field itself — this module reads $\mathbf a$, it
doesn't produce it.

---

## 1. A geometric fact this module relies on, confirmed rather than assumed

The projection formula (§2) sums over edges belonging to the port face. It's worth confirming
this is exactly correct — that edges *not* on the port face, but belonging to a tet that
touches it, contribute nothing — rather than treating it as an assumption.

Consider a tet with a face on the port boundary, spanned by local vertices $a,b,c$, with $d$ the
opposite (non-boundary) vertex. On this face, $\lambda_d\equiv0$. For an edge connecting a face
vertex to the opposite vertex (e.g. $(a,d)$, not one of the face's own three edges):

$$\mathbf W_{(a,d)}(\mathbf r)\big|_{\text{face}} = \lambda_a(\mathbf r)\nabla\lambda_d - \underbrace{\lambda_d(\mathbf r)}_{=0}\nabla\lambda_a = \lambda_a(\mathbf r)\,\nabla\lambda_d$$

By Module 1 §2.4's identity, $\nabla\lambda_d \propto \hat{\mathbf n}_d^{\text{out}}$ — the
outward normal of the face *opposite vertex $d$*, which is exactly this face. So
$\nabla\lambda_d$ is **purely normal** to the face the point lies on. $\mathbf W_{(a,d)}$ is
therefore nonzero at the face (a normal-direction vector, scaled by $\lambda_a$), but its
**tangential** component there is exactly zero. Only edges belonging to the face's own three
edges have nonzero tangential trace on it — confirming the projection sum below is complete as
written, not an approximation that happens to work.

---

## 2. Modal amplitude extraction

For a `SweepResult` at frequency $\omega$ with excitation $(q,1)$ and solved $\mathbf a$: at
each port $p$ (both ports, not just the excited one) and each of its tracked modes $m$, restrict
the tangential field trace to that port's own edges (§1) and apply Module 4's corrected §4.3
projection:

$$\boxed{\;a_m^{(p),\text{total}} = \frac{1}{N_m}\int_{S_p}\Big(\mathbf E_t^{(p)}\times\mathbf h_m^{(p)}\Big)\cdot\hat x\,dS, \qquad \mathbf E_t^{(p)}(\mathbf r) = \sum_{j\,\in\,\text{port }p\text{'s edges}} a_j\,(\mathbf W_j)_t(\mathbf r)\;}$$

with $N_m$ the same self-normalization quantity Module 4 §4.3 established (confirm which of the
two equivalent conventions — pre-normalized-to-1, or explicit division — is actually in effect,
per the open note carried from Module 4/6). Module 4's own biorthogonality argument confirms
this recovers exactly $a_m^{(p),\text{total}} = a_m^{+}+a_m^{-}$ at that port for that mode.

---

## 3. Raw generalized S-parameters

### 3.1 Isolating the reflected/transmitted amplitude

At the **excited** port ($p=q$), the known incident amplitude must be subtracted; at every
**other** port, the projected amplitude *is* the transmitted amplitude directly (no incident
wave originates there):

$$S_{(p,m),(q,1)} = a_m^{(p),\text{total}} - \begin{cases}1 & \text{if } (p,m)=(q,1)\\ 0 & \text{otherwise}\end{cases}$$

using the unit-incident-amplitude convention already fixed by Module 6 §7's excitation set
($a_1^{+,\text{inc},(q)}=1$).

### 3.2 The dominant-mode block vs. the mode-conversion rows — the point of this whole design

Because only each port's **dominant** mode is ever excited (Module 4 §5.3, Module 6 §7), the
generalized S-"matrix" is not square: for $n_{\text{modes}}$ tracked modes per port and 2 ports,
there are $2n_{\text{modes}}$ possible $(p,m)$ *outputs* but only 2 possible $(q,1)$
*excitations*. Two things worth separating explicitly, since conflating them would either cause
confusion about why $S$ isn't square or — worse — cause the actual point of this design to be
silently dropped from the output:

- **$S^{\text{dominant}}$**: the $m=1$ rows only, $S_{(p,1),(q,1)}$ — a genuine square $2\times2$
  matrix, exactly the traditional network-parameter $S_{11},S_{12},S_{21},S_{22}$. This is what
  the top-level doc's Phase 1–4 reciprocity/passivity gates are stated against, and it's the
  quantity a phase-shifter design is ultimately characterized by.
- **$S^{\text{conversion}}$**: the $m>1$ rows, $S_{(p,m>1),(q,1)}$ — how much power scatters
  into higher-order/cross-polarized modes at each port due to the LC's off-diagonal
  $\varepsilon_r$. **This is not a secondary diagnostic — capturing this is the entire reason
  ports were required to hold $\ge2$ modes** (top-level architecture doc, ports design
  invariant), tracing directly back to the very first framing of this project: "the director
  field generically produces off-diagonal tensor components... this requires the model to
  capture mode conversion." A pipeline that computes and discards these rows, reporting only
  $S^{\text{dominant}}$, would be discarding the specific physical effect the anisotropic-tensor
  FEM formulation was built to see that HFSS's slab-clustering approach couldn't.

Report both, always — not $S^{\text{conversion}}$ only on request.

### 3.3 A consequence of the excitation convention worth stating plainly

Because higher-order modes are never excited (only read out), **full generalized reciprocity
across all mode pairs cannot be checked** with this excitation set — e.g. there is no way to
compare "mode 2 emerging from a mode-1 excitation" against "mode 1 emerging from a mode-2
excitation," since no mode-2 excitation exists. Only the dominant-to-dominant reciprocity
$S_{(p,1),(q,1)}=S_{(q,1),(p,1)}$ is checkable. If full cross-mode reciprocity were ever wanted,
it would require extending Module 6 §7's excitation set to include higher modes — an easy
architectural extension (the excitation set is just a configuration list) but a deliberate scope
decision, not something to add silently here.

---

## 4. De-embedding

Applying Module 4 §6's formula directly, using each port's *own* tracked mode's $\gamma$ at that
frequency (already available from `port_modes`, no re-solving needed):

$$S^{\text{ref}}_{(p,m),(q,1)} = S_{(p,m),(q,1)}\;e^{+\gamma_m^{(p)}(\omega)\,d_p}\;e^{+\gamma_1^{(q)}(\omega)\,d_q}$$

---

## 5. Energy conservation — the extended sum, and why the simple two-term check is insufficient once LC is present

For a lossless structure, the standard 2-port check is $|S_{11}|^2+|S_{21}|^2=1$. **This is only
exact when no mode conversion occurs.** Phase 1–3 (isotropic or axis-aligned tensor, no
mode-converting off-diagonal terms reaching the ports significantly) satisfy it essentially as
stated. **Once LC is present (Phase 4), the correct lossless energy-conservation statement must
include every captured mode, not just the dominant one:**

$$\boxed{\;\sum_{p}\sum_{m=1}^{n_{\text{modes}}} \big|S^{\text{ref}}_{(p,m),(q,1)}\big|^2 \approx 1 \qquad \text{(lossless, excitation } q\text{)}\;}$$

**Why this matters enough to state emphatically**: a correctly-implemented, genuinely lossless,
mode-converting structure will generally show $|S_{(1,1),(q,1)}|^2+|S_{(2,1),(q,1)}|^2 < 1$ —
some power has gone into the $m>1$ terms, not been lost. Checking only the dominant-mode sum
against $1$ would misdiagnose correct mode-conversion behavior as a passivity violation or a
leaky/lossy bug, sending an implementer chasing an error that isn't there. The extended sum is
the check that actually distinguishes "power converted to a captured higher mode" from "power
genuinely lost or leaked."

**Honest limitation on this check's precision**: it is only exact in the limit of capturing
every mode carrying non-negligible power. With a finite $n_{\text{modes}}$ (2–3, per the
top-level doc), a **small** deficit even for a truly lossless configuration is expected —
representing power scattered into uncaptured higher-order modes — and is not itself a bug. A
**large** deficit is the actual warning sign, and the first thing to check in that case is
whether $n_{\text{modes}}$ needs increasing (more of the scattered power needs to be captured
to close the balance), not whether the assembly or extraction has an error.

---

## 6. Sweep aggregation

Collect the per-(frequency, excitation) results into the final deliverable: an
$S(\omega)$ dataset indexed by $(p,m,q)$ (the excitation's mode index is always $1$ by
convention, so it's dropped from the key rather than carried as a redundant always-$1$ field),
with $S^{\text{dominant}}$ and $S^{\text{conversion}}$ as documented views over the same
underlying structure rather than separately stored data.

---

## 7. Step-by-step build order

1. Confirm §1's tangential-trace-restriction fact numerically on a hand-built two-tet-with-one-
   boundary-face test case, before writing any extraction code against it — this is the
   assumption the whole module's edge-restriction depends on.
2. Implement modal amplitude extraction (§2) for a single `SweepResult`, single port, single
   mode — validate against a case where the answer is known by construction (e.g. drive port 1
   at mode 1 on a long matched line and confirm port 1's own mode-1 amplitude reads back close
   to the expected reflection for a known termination).
3. Extend to all tracked modes at all ports (§3.1) — implement $S^{\text{dominant}}$ and
   $S^{\text{conversion}}$ as explicit, separately-named views from the start (§3.2), not as an
   afterthought bolted onto a dominant-only implementation.
4. De-embedding (§4) — low-risk, direct application of an already-derived formula.
5. The extended energy-conservation check (§5) — implement it as the *default* passivity/energy
   diagnostic, not the simple two-term version, so Phase 4 testing doesn't need a separate,
   easily-forgotten upgrade later.
6. Sweep aggregation (§6) into the final dataset structure.
7. Run the full §8 validation suite — this is where the top-level doc's Phase 1 gate (flagged
   as the real acceptance criterion since Module 4) finally runs end-to-end for the first time
   in this project.

---

## 8. Validation targets

- **§1's geometric fact**: numerically confirm zero tangential trace for non-face edges on the
  hand-built test case.
- **Phase 1 gate, finally end-to-end**: $S^{\text{dominant}}_{21}=S^{\text{dominant}}_{12}$ and
  $|S^{\text{dominant}}_{11}|^2+|S^{\text{dominant}}_{21}|^2\approx1$ for a uniform lossless
  line. If this fails, the first two places to check remain Module 4 §3.6/§5.1's flagged sign
  and arrangement questions — this module's extraction logic doesn't independently introduce
  new sign risk (see the note below), so a failure here is diagnostic of something upstream.
- **Extended energy conservation** (§5): on a Phase 4 (LC-present) test case with a deliberately
  rotated director, confirm the extended sum closes near 1 while the dominant-only sum does
  not — this is the test that specifically distinguishes correct mode-conversion behavior from
  a passivity bug, and its absence would leave that distinction untested.
- **De-embedding sanity**: shifting $d_p$ by a known amount changes $\arg(S^{\text{ref}})$ by
  exactly the expected $\mathrm{Im}(\gamma_m)\,d_p$ — a cheap, self-contained check on the
  formula's sign and application.
- **Mode-conversion magnitude sanity, sharpened to an exact symmetry test** (Phase 4, ties to
  the top-level doc's original director-tilt gate): $S^{\text{conversion}}$ must come out
  **exactly zero** (to numerical tolerance) for any director field confined to the microstrip's
  vertical mirror-symmetry plane (through the trace centerline) — the LC-loaded cross-section
  retains that mirror symmetry for such a field, and the dominant and cross-polarized modes
  belong to different symmetry classes of it, so a symmetric perturbation cannot couple them
  regardless of the accumulated differential phase between the LC region's own birefringent
  eigenmodes. This is a real selection rule, not a magnitude heuristic — treat a nonzero result
  on an in-plane test director as a bug, not as "small but expected." Once the director field
  has any out-of-plane component, $S^{\text{conversion}}$ should become genuinely nonzero and
  grow with the out-of-plane tilt — the mechanism is the same birefringence-driven modal
  mismatch that produces the device's useful differential phase shift in the first place
  (mode-mismatch at the LC entrance, differential phase $(\beta_1-\beta_2)L_{\text{lc}}$ over
  the cavity, mode-mismatch again at the exit), not a separate effect and not something
  reciprocity would cancel — reciprocity relates $S_{ij}$ to $S_{ji}$, it does not imply a
  device undoes whatever mode conversion it caused between entrance and exit.
- **Reciprocity scope note**: confirm only $S^{\text{dominant}}$'s reciprocity is checked, and
  that this is documented as a consequence of the excitation convention (§3.3), not silently
  presented as if full cross-mode reciprocity had been verified when it structurally cannot be
  with the current excitation set.

**A reassurance worth stating explicitly, not just implying**: Module 4 §3.6 and §5.1 both carry
open honesty flags about internal sign/arrangement conventions in the port eigenproblem and
surface-term derivation. This module's extraction formula does not independently risk a *new*
sign error on top of those — it uses the identical incident-amplitude convention
($a_1^{+,\text{inc}}=1$) that Module 6 used to build the excitation in the first place, so as
long as that convention is applied consistently (which it structurally is, both being read from
the same `PortMode`/excitation-set objects), an internal sign inconsistency in Modules 3–5 would
most likely surface as a magnitude or passivity failure in the Phase 1 gate above, not as a
silent, undetectable error localized to this module.

---

## 9. Interface / class contract

```
# extract.sparameters
def port_face_edges(port_tag: str, mesh) -> list[int]     # global edge indices, this port's own face edges only

def project_amplitude(a: array, port_edges: list[int], mode: PortMode) -> complex   # a_m^{(p),total}, section 2

def raw_s_parameters(sweep_results: list[SweepResult], ports: list[str], n_modes: int) \
    -> dict[(str,int,str), complex]     # keyed (port_p, mode_m, excited_port_q); mode index on q always 1, omitted from key

def deembed(S_raw: dict, port_modes_by_freq: dict, offsets: dict[str, float]) -> dict

def energy_balance(S_deembedded: dict, excitation_port: str, n_modes: int) -> float  # section 5's extended sum

def assemble_sweep_dataset(frequencies: list[float], S_by_freq: list[dict]) -> SParameterDataset

class SParameterDataset:
    frequencies: array
    S_dominant: array            # (n_freq, n_ports, n_ports) complex — the traditional S-matrix
    S_conversion: dict           # (port_p, mode_m>1, excited_port_q) -> array over frequency
```

This is the last module in the driven-solver pipeline (Modules 0–7); Module 8 (`validation`) is
the test harness that exercises all of them together against the top-level doc's Phase 1–4
gates, using exactly the `SParameterDataset` this module produces.
