# Replotting the fixed-percentile convergence analysis

The expensive bootstrap does not need to be repeated to change the figures or
to extend the percentile-value comparison horizon.  The saved
`percentile_profiles.csv` already contains every fixed percentile, its
bootstrap interval, and its statistical-validity flags in every temporal bin.

From the repository root, run:

```bash
python replot_envelope_population_convergence.py \
  --analysis-dir /path/to/the/existing/convergence_analysis \
  --shape-horizon-mode reference-minus-tolerance
```

The default output directory is:

```text
/path/to/the/existing/convergence_analysis/replot_dynamic_reference_horizon/
```

For each percentile family, `reference-minus-tolerance` compares the
percentile values from `t_min` through

```text
last statistically valid reference edge - tolerance_days
```

Thus a candidate allowed to end one 0.4-day bin before the reference must
also reproduce all percentile values over the complete interval that it is
required to cover.

To reproduce the former fixed 0.4--4.0 day analysis while only changing the
visual style:

```bash
python replot_envelope_population_convergence.py \
  --analysis-dir /path/to/the/existing/convergence_analysis \
  --shape-horizon-mode fixed \
  --shape-t-max 4.0
```

This operation reads only saved CSV files.  It does not reload the simulated
light curves and does not perform new bootstrap resampling.

Both convergence figures contain a single panel.  The repetition success
fractions remain available in the CSV summaries and PNG value tables, but are
not repeated as a second graphical panel.
