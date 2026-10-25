# Notes on the analysis references

The frozen analysis inputs and expected outputs preserve the reviewed submission. The current comparison report is `provenance/analysis_reproduction.json`.

## Historical manuscript rounding

An earlier comparison of the ten-model scoreboard and seven strict-completion rows found one discrepancy: Qwen2.5 Omni's overall upper 95% confidence bound is `0.24449956986593954` on the unit scale, which the v5 analysis displays as **24.4%**; the earlier manuscript printed **24.5%**. The scientific calculation is unchanged. The manuscript is not included here, so its current wording has not been rechecked. Agreement with the frozen analysis references does not resolve this manuscript discrepancy.

## Ancillary figure corrections

During preparation, nine ancillary workspace figures were identified as stale. They were replaced with independent renders from unchanged original v5 source and original result inputs. All five main quantitative figure references remained unchanged. The affected relative paths under `analysis/expected/paper_plot_v5/` were:

- `supplement/panel_06/plot_component_active_vs_full_context_gap.png`
- `supplement/panel_06/plot_semantic_belief_profile_and_component_gap.png`
- `supplement/panel_07/plot_semantic_revision_and_belief_inertia.png`
- `supplement/panel_11/plot_marginal_completion_stage_profile.png`
- `supplement/panel_13/plot_efficiency_frontier.png`
- `supplement/panel_13/plot_evidence_belief_dissociation.png`
- `supplement/panel_13/plot_track2_acquisition_coverage.png`
- `supplement/panel_14/plot_uncertainty_awareness_and_supported_commitment.png`
- `supplement/panel_14/plot_uncertainty_awareness_and_supported_commitment_simplified.png`

The current canonical file hashes are recorded in `provenance/analysis_expected.json`. Subsequent full reproduction verified all 40 numerical tables and all 20 figure pixel arrays against these references.
