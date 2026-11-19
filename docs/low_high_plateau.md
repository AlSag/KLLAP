# Pertes MW relatives high / low et plateau opérationnel

## Question et définition

Le mode « stabilité d'une coupure low/high » de
`analyze_kne_latitude.py` calcule à chaque seuil b :

    low : |latitude galactique| < b
    high : |latitude galactique| >= b
    f_MW = nombre de pertes MW supplémentaires / nombre de KNe avec couleur baseline
    Q(b) = f_MW(high) / f_MW(low)

Une perte MW supplémentaire désigne un événement ayant au moins un bin couleur
passant les deux seuils m5 sans poussière, mais aucun bin passant ces seuils
après ajout de l'extinction. Les événements déjà perdus par m5 avant ajout de MW
ne sont pas comptés comme pertes MW supplémentaires. Le dénominateur reste le
nombre de KNe ayant au moins un bin couleur avant toute coupure, pour conserver
la définition de l'analyse précédente.

Un événement conservant un seul bin après MW n'est pas perdu selon ce critère,
même s'il a perdu beaucoup d'autres bins. Cette métrique mesure la perte complète
de disponibilité de couleur dans la fenêtre temporelle, pas la perte de durée.

Q = 10 % signifie que le taux high est dix fois plus petit que le taux low,
soit une baisse relative de 90 %. Q = 100 % signifie des taux égaux. Q > 100 %
signifie davantage de pertes à haute latitude. Aucun rapport avec dénominateur
nul n'est remplacé par 1 ; il reste indéfini.

## Interactive workflow

Run `python analyze_kne_latitude.py`, select the low/high-cut stability mode,
then supply either an all-sky availability table or a list of low/high pairs.
For paired latitude experiments, reuse the same physical catalogue and event
IDs. The script reports the observed counts and loss ratios directly.

A candidate plateau must include the requested number of thresholds and angular
span, have defined ratios throughout its tail, and satisfy the maximum relative
deviation from that tail's median. These conditions define a plateau in angle;
they are not minimum event-count requirements. A zero denominator remains
undefined. No uncertainty band is inferred by resampling events.

The optional latitude experiment retains its attenuation comparison. It reads
field E(B-V) from the original OpSim index for new `model` populations. Ordinary
generation and colour-envelope construction do not apply attenuation.

The output records the tolerance, selected boundary (if any), counts and raw
ratios. This is a descriptive result for the simulated population and grid,
not a universal Galactic-latitude threshold.
