# Methodological correction in v2.1.0 candidate

## Issue

The manuscript defines Primary B as the incremental held-out accuracy obtained by adding hidden-representation statistics to output-only probe features. In v2.0.0, the implementation instead compared the profile-only detector with the output-only detector. That comparison does not estimate the prespecified incremental contribution.

## Correction

Version 2.1.0 candidate compares:

- **combined detector:** TSS + spectral concentration + local sensitivity + probe accuracy + mean confidence + predictive entropy;
- **output-only detector:** probe accuracy + mean confidence + predictive entropy.

The estimate and pair-bootstrap interval are stored as `combined_minus_output_accuracy`. Primary B passes only when the estimate is at least the configured margin and its pair-bootstrap 95% lower bound is above zero.

## Scientific consequence

No confirmatory claim may be made from the v2.0.0 Primary B implementation. The full confirmatory experiment must be run from the corrected version with raw outputs preserved. This correction does not alter the archived pilot; it changes only the prospective decision procedure.
