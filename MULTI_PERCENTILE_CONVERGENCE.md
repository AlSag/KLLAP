# Fixed-percentile convergence

```bash
python analyze_envelope_population_convergence_fast.py
```

Choose **Fixed percentiles: shape and temporal extent** in the terminal.
Enter the parent run, colour, stage (`no_m5` or `m5`), nested population sizes,
repetitions, grid, tolerances, depth scenario and output directory.
No long command line is needed.

The parent is loaded once. Smaller samples are nested subsets without
replacement; all their events enter the empirical quantiles. The maximum
selected N is the reference. The calculation uses p5, p16, p25, p50, p75,
p84 and p95, grouped into three fixed intervals.

Shape convergence compares the requested values in every required reference
bin. Missing candidate values fail the comparison. Extent convergence compares
the last consecutive supported edge with the reference. Every nonempty bin is
eligible, including a singleton. The outputs report counts; there is no
minimum-tail rejection, training split or resampling uncertainty calculation.

The required N is the first tested size whose passing fraction reaches the
chosen level and whose larger tested sizes also pass. No intermediate N is
interpolated. The maximum-N self-comparison cannot establish convergence beyond
that finite reference population.

Outputs: `percentile_profiles.csv`, `percentile_reference_profiles.csv`,
`convergence_thresholds.csv`, shape/extent comparisons and figures. Configuration
hashes guard resumed checkpoints. Use a new output folder after this update.
