# Detecting Learned Shortcuts from the Internal State of Neural Networks
### A Multi-Metric Framework, Its Robustness, and a Principled Limit

**Suela Kociu** — Independent Researcher, Turin, Italy · suelaresuli@gmail.com

*Rilevare le scorciatoie apprese dallo stato interno delle reti neurali: un framework multi-metrica, la sua robustezza e un limite di principio.*

---

## What this is / Cos'è

Neural networks can reach high training accuracy while secretly relying on **spurious features ("shortcuts")** that fail out of distribution. This repository contains a multi-metric framework that inspects the **internal state** of a trained network and detects shortcut reliance — **without ever consulting a future test set** — together with the demonstration of a **principled limit**: some shortcut classes are invisible to internal inspection.

Le reti neurali possono raggiungere alta accuratezza basandosi su **caratteristiche spurie ("scorciatoie")** che falliscono fuori distribuzione. Questo repository contiene un framework multi-metrica che ispeziona lo **stato interno** della rete e rileva la dipendenza da scorciatoie — **senza mai consultare un test futuro** — insieme alla dimostrazione di un **limite di principio**: alcune classi di scorciatoie sono invisibili all'ispezione interna.

## Key results / Risultati principali

| Result | Value |
|---|---|
| TSS correlation with learning (2 tasks, 30 nets each) | r = 0.72 / 0.89 |
| Shuffled-label control | r = 0.08 (collapses ✓) |
| Combined metrics predict system state | R² = 0.84 |
| Shortcut detection (neutral probe + leave-one-seed-pair-out) | 100% |
| Robust to: width, depth, ReLU, 3–5 shortcuts, adversarial hiding | 93–100% |
| **Unseen shortcut types**: threshold / multiplicative / sparse | **100% / ~chance / 0%** |
| Three independent strategies on sparse shortcut | all fail (≤ chance) |

**The limit is the finding**: shortcuts that do not reorganize the bulk of the representation (sparse, multiplicative) are indistinguishable from genuine strategies by internal inspection alone — at any point during training.

## The three metrics / Le tre metriche

- **TSS (Task-Structured Separability)** — between-class separation over within-class dispersion in hidden space: `TSS = B / (W + B)` (Fisher-style separability on hidden representations; implemented in code as `delta_phi_structured`)
- **Ψ (spectral concentration)** — `1 −` normalized eigenvalue entropy of the hidden correlation matrix. The shortcut signature: *suspiciously over-ordered* internal state (Ψ ≈ 1.0)
- **Ω (local sensitivity)** — correlation between input-perturbation size and hidden-state change, averaged over many perturbations

## Repository structure / Struttura

```
paper/     Full bilingual paper (IT/EN) with all equations, tables, figures
code/      All experiment scripts (see order below)
figures/   All figures, generated from real experimental outputs
```

## Reproduce everything / Riproduci tutto

```bash
pip install -r requirements.txt
```

Suggested order (each script is standalone and prints its results):

| # | Script | What it shows |
|---|---|---|
| 1 | `kociu_v2_step1.py`, `kociu_v2_causal.py` | Negative controls: naive metrics that fail |
| 2 | `kociu_v3_structured.py` | TSS tracks learning (the working metric) |
| 3 | `kociu_validation.py` | Validation on 2 tasks + shuffled-label control |
| 4 | `kociu_3metrics.py`, `kociu_independence.py` | Ψ, Ω; complementarity (R²=0.84) |
| 5 | `kociu_emergence.py`, `kociu_emergence_validation.py` | Shortcut detection (98.8% on 80 nets) |
| 6 | `kociu_stresstest.py`, `st_part1.py`, `st_part2.py`, `st_adversarial.py` | Stress tests incl. adversarial |
| 7 | `kociu_corrected_test.py` | Neutral probe + leave-one-seed-pair-out (100%) |
| 8 | `kociu_unseen_shortcut.py` | **The principled limit** (unseen shortcut types) |
| 9 | `kociu_second_sense.py`, `kociu_trajectory.py` | Two more strategies that fail on sparse |

Runtimes: seconds to a few minutes each on CPU.

## Honest scope / Ambito onesto

- Synthetic data, small/medium networks only. Images, text, large models: untested.
- This work makes **no claims about consciousness**, in either direction. Whether task-structured separability of internal representations bears any empirical relation to integrated-information measures is left as an **open question** for future work (see paper, Conclusions): only direct comparison on systems where both quantities are computable can establish or exclude it.
- A clean internal reading certifies the absence of *detectable* shortcuts, not of all shortcuts.
- **Independent replication is invited** — that is what this repository is for.

## Citation / Citazione

See `CITATION.cff`. Code developed with AI-assisted programming; every result verified by direct execution.

## License / Licenza

CC BY 4.0 — free to share and adapt with attribution.
