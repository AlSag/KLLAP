# Scientific conventions

## Magnitudes and population membership

Generation saves the model magnitudes without adding foreground Milky Way
extinction. Envelope construction uses saved magnitudes directly. Latitude
selection remains an independent sky-population choice. Legacy attenuated
populations must be regenerated if attenuation is unwanted.

Envelope estimation uses 100% of the explicitly selected population. There is
no training/validation/test split. `max_events` and latitude cuts, when chosen,
still restrict that population and are recorded.

## Percentile envelopes

The terminal selects p_q–p_(100-q). Estimates use NumPy's linear empirical
percentiles. Every finite nonempty bin is kept, even with one event. Empty bins
remain undefined. Counts and the selected percentiles accompany each bin.
There is no minimum-tail rule or resampling-based validity criterion.

`no_m5` uses all numerically supported colours. `m5` requires both selected
magnitudes to meet the chosen band depths, inclusively. PS1 colour selection
uses the paired LSST magnitudes; ZTF uses its supplied depth distributions.
Interpolation never extrapolates outside saved model support. Numerical model
floor removal is retained: it is a model-validity condition, not a sample-size cut.

## Upper limits

For `band1-band2`, both detections yield a colour. A band1 non-detection and
band2 detection yield `m5_band1-mag_band2`; the plotted statistics describe
these limit values together with exact colours of two-band detections.
They do not identify both sides of the true-colour distribution. The reverse
case is counted separately. Two non-detections give no finite colour constraint.
Unknown model support is distinct from a non-detection. The count table and
finite-grid horizon retain that distinction. Empty bins are not bridged.

## Subsets and realization comparisons

Convergence experiments draw nested subsets without replacement; the largest
selected subset is their reference. All events within each subset contribute.
Realization experiments compare separately generated populations. These are
descriptive population-size experiments, not resampling uncertainty intervals.
An empirical percentile based on very few events is defined but can change
substantially when more events are generated; counts make this visible.

## Latitude experiments

The dedicated latitude-loss workflow still compares m5 losses before and after
adding field attenuation. It retrieves E(B-V) from the OpSim field index for
new model-magnitude populations. This optional experiment does not change the
generator or envelope magnitudes. Loss ratios with zero denominators remain
undefined. Plateau decisions use observed tail variation and a chosen span,
without resampling or arbitrary minimum event counts.
