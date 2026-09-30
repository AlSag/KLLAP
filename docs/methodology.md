# Scientific methodology

## KNe colour envelopes

For each event and each requested band, magnitudes are interpolated in
`log10(time)` only inside the numerical support saved by the generator. There
is no extrapolation. A colour is available only where both bands are
available. The population colour distribution is then summarized in fixed
time bins by symmetric percentile bounds chosen from the configured grid.

The four stages isolate two effects:

1. intrinsic colour support, without m5 and without foreground MW extinction;
2. the same saved population with MW extinction, without m5;
3. intrinsic magnitudes after applying the two-band m5 selection;
4. MW-extincted magnitudes after applying the same m5 selection.

This separation is important: the second stage answers what dust does to the
observed colours without a depth cut, while the difference between stages 3
and 4 measures the additional availability loss due to dust in a depth-limited
sample.

## Galactic-latitude loss experiment

The recommended design uses disjoint `|b|` bins rather than cumulative
`|b| < b_cut` and `|b| >= b_cut` populations. All bins reuse the same intrinsic
catalogue, distance, orientation, and event IDs. Their sky positions, merger
epochs, and noise are drawn with distinct sky seeds.

For each event over the selected time window, the analyzer records the number
of colour bins available in the baseline, after m5 without MW, and after m5
with MW. The comparator estimates three loss rates per latitude bin:

- m5-only loss;
- additional MW loss among the m5-supported opportunities;
- combined m5+MW loss.

Rates are divided by the corresponding rate in the fixed lowest-latitude
reference bin. This relative design is useful when the simulated population is
not intended to represent an absolute astrophysical rate.

The bootstrap resamples matched `event_id` rows together across latitude bins.
A plateau is operationally defined as the first bin for which it and all later
bins remain within the selected relative tolerance of their common median,
with at least the configured number of bins. Report together:

- the best-estimate boundary;
- its bootstrap interval;
- the fraction of bootstrap samples in which a plateau was found;
- the event counts and raw loss rates underlying every ratio.

A low plateau-detection fraction or a wide interval means that the simulations
do not identify a precise threshold. The plateau is a property of the selected
cadence, m5 scenario, colour, time window, distance distribution, dust map,
binning, and tolerance—not a universal Galactic constant.

## Pairing and seeds

Use the same intrinsic catalogue and intrinsic seed for every latitude bin.
Use a different sky/noise seed for each bin. Reusing the same sky seed would
not make the sky fields identical because the eligible-field sets differ, and
could introduce unwanted algorithmic correlations. Pairing is carried by the
shared intrinsic `event_id`, not by identical random streams for the sky.
