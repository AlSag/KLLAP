# Convergence multi-percentiles de l'enveloppe couleur

Le mode par défaut de `analyze_envelope_population_convergence_fast.py`
répond maintenant à deux questions distinctes.

1. **Forme de l'enveloppe** : sur une fenêtre temporelle fixée avant
   l'analyse (par défaut 0,4–4,0 jours), les valeurs des percentiles du
   sous-échantillon diffèrent-elles de celles de la plus grande injection de
   moins de 0,10 mag ?
2. **Étendue temporelle** : jusqu'à quel dernier bin consécutif chaque famille
   de percentiles satisfait-elle les critères de support statistique ?

Chaque sous-échantillon est utilisé à **100 %** : il n'existe aucune séparation
train/validation/test dans ce mode. Les percentiles fixes calculés sont p5,
p16, p25, p50, p75, p84 et p95. Les trois déclarations conjointes sont :

- `central_90` : p5 + p50 + p95 ;
- `central_68` : p16 + p50 + p84.
- `central_50` : p25 + p50 + p75.

Aucune borne adaptative n'est calculée ni affichée dans ce mode.

## Exemple de commande finale

```bash
cd ~/repos/kne-lsst-analysis

python analyze_envelope_population_convergence_fast.py \
  --run-dir runs/convergence_500000_10Mpc_500Mpc_seed_2334271287 \
  --color-pair g-r \
  --photometric-system lsst \
  --stage m5_with_mw \
  --sample-sizes 10000,30000,50000,75000,100000,150000,200000,250000,300000,350000,400000,450000,500000 \
  --repetitions 10 \
  --subsample-seed 20261002 \
  --builder-seed 12345 \
  --t-min 0.4 \
  --shape-t-max 4.0 \
  --t-max 16.0 \
  --bin-width 0.4 \
  --n-boot 200 \
  --minimum-tail-events 20 \
  --bootstrap-threshold-mag 0.10 \
  --m5-scenario best_p75 \
  --threshold-scope common_global \
  --stability-metric colour \
  --tolerance-mode absolute \
  --absolute-tolerance-mag 0.10 \
  --tolerance-days 0.4 \
  --required-fraction 0.9 \
  --output-dir runs/convergence_500000_10Mpc_500Mpc_seed_2334271287/analysis/convergence_multiquantile_g_r
```

Le réglage recommandé est de conserver 10 répétitions et 200 bootstraps. Dix
répétitions donnent une résolution de 10 points de pourcentage au critère de
réussite : `--required-fraction 0.9` signifie alors exactement 9 répétitions
réussies sur 10. Les 200 bootstraps servent uniquement à estimer la largeur
d'incertitude de chaque percentile dans chaque bin. La population de référence
complète est calculée une seule fois puis réutilisée pour les dix répétitions.

## Sorties principales

- `percentile_value_convergence.png` : convergence des valeurs de couleur ;
- `temporal_extent_convergence.png` : convergence du dernier bin valide ;
- `convergence_thresholds.csv` : seuils de population séparés et combinés ;
- `percentile_value_convergence_summary.csv` : résultats par famille et N ;
- `temporal_extent_convergence_summary.csv` : étendue par famille et N ;
- `percentile_profiles.csv` : toutes les valeurs et diagnostics par bin.

Dans `convergence_thresholds.csv`, `N_recommended_both` est le maximum entre
le seuil de forme et le seuil d'étendue. Si ce nombre est égal à la plus grande
injection, la convergence sous la référence n'est pas démontrée : la plus
grande injection est nécessairement identique à elle-même.

Le critère de 20 événements attendus dans la queue reste un critère de
**validité**, pas un critère d'accord en magnitude. Il intervient donc dans
l'étendue temporelle et dans le support des bins de la fenêtre de forme.

