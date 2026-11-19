# Colour-envelope convergence

Run `python analyze_envelope_population_convergence_fast.py` and answer the
terminal questions: parent population, colour, stage, nested sizes, repetitions,
time grid, depth scenario, comparison tolerances and output folder.

The population is loaded once. Each repetition draws a reproducible ordering;
smaller samples are prefixes without replacement. Every selected event enters
the empirical quantiles. There is no training partition, minimum tail count or
resampling uncertainty calculation.

The default fixed-percentile mode compares p5–p95, p16–p84 and p25–p75.
It reports percentile-value convergence separately from temporal-extent
convergence. Compatibility modes compare envelope values or the last
consecutive nonempty bin. Missing reference-required bins make a comparison
fail rather than being silently dropped.

A tested size qualifies when the requested fraction of repetitions passes and
all larger tested sizes pass too. A size equal to the maximum reference is a
self-comparison, not evidence for a plateau. The result is conditional on the
saved population, priors, distance range, bands, depth cuts and time grid.

Outputs include per-size/per-repetition profiles, reference profiles, raw and
summary comparisons, figures and configuration metadata. Use a new output
folder when changing conventions; older checkpoints must not be reused.

For the fixed-percentile outputs, see
[the multi-percentile guide](../MULTI_PERCENTILE_CONVERGENCE.md).
