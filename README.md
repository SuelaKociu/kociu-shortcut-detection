# Internal Representation Profiles for Detecting Compressive Shortcut Learning

> **Candidate status (26 July 2026).** Version 2.1.0 is a methodological-correction candidate, not a published confirmatory result. It corrects Primary B so that the combined profile+output detector is compared against the output-only detector. Full confirmatory outputs remain pending.

**Suela Kociu** — Independent Researcher, Turin, Italy

This repository studies whether a small profile of hidden-representation statistics can distinguish paired neural networks trained with and without a controlled shortcut.

The version-1 pilot found perfect separation for a strong linear shortcut under a neutral labeled probe and leave-one-seed-pair-out validation. Transfer was high to a closely related signed-threshold generator but poor to product and historical sparse generators. Version 2 treats that failure as empirical non-transfer, not as an impossibility theorem.

## Scientific status

- Archived pilot summaries: exploratory evidence.
- Confirmatory protocol: frozen in `configs/confirmatory.yaml`.
- Confirmatory neural results: not yet generated.
- Real-world validation: not performed.

## Metrics

1. **Task-Structured Separability (TSS)** — bounded between/within-class separation on a labeled probe.
2. **Spectral Concentration (SC)** — one minus normalized eigenvalue entropy of the hidden correlation matrix.
3. **Local Sensitivity (LS)** — average correlation between input- and hidden-space perturbation norms.

Because TSS uses probe labels, this is a representation-based diagnostic on a controlled labeled probe, not an intrinsic label-free inspection method.

## Main correction in v2.1

Primary B now compares **profile + output features** against **output-only features**. The previous implementation compared profile-only against output-only and therefore did not match the written protocol.

## Scope

This project does not establish that every shortcut has a detectable internal signature, that a clean profile proves shortcut absence, or that these statistics measure consciousness or phenomenology.

The defensible claim is narrower: some shortcut generators produce global representational changes detectable under controlled conditions, while transfer to structurally different generators can fail sharply.

See the candidate branch files, tests, methodological correction note, and manuscript for details.
