# Convergence des valeurs de l'enveloppe couleur

Le script rapide propose deux modes : `--stability-metric colour` (nouveau
mode par défaut) et `--stability-metric last-edge` (ancienne analyse du dernier
bin, inchangée). Les courbes sauvegardées sont lues une fois. Le code ne génère
pas de nouvelles KNe. Il utilise des sous-échantillons sans remise, emboîtés
dans une même répétition, et les splits train/validation du builder.

## Critère

Pour chaque répétition, la référence est le sous-échantillon de plus grande
taille N testé. Dans **tous ses bins initiaux consécutifs valides**, comparer
séparément `color_min`, `color_median`, `color_max` aux valeurs du candidat :

    abs(C_N - C_reference) <= max(0.10 * abs(C_reference), absolute_tolerance_mag)

Le plancher absolu vaut **0 mag par défaut** : le critère est donc strictement
relatif. Une référence nulle exige une différence nulle, à l'arrondi machine
près. Les couleurs négatives sont traitées avec la valeur absolue de la
référence. Près de zéro, le critère peut être extrêmement strict. Un plancher
optionnel `--absolute-tolerance-mag 0.02`, par exemple, change le critère en
un critère mixte et doit être explicitement annoncé dans les résultats.

Une répétition passe si **les trois valeurs passent dans tous les bins**.
Un bin de référence absent ou invalide chez le candidat fait échouer la
répétition. On n'utilise pas seulement l'intersection des bins disponibles.
N requis est la première taille testée où au moins `--required-fraction 0.9`
des répétitions passent, **et où toutes les tailles testées plus grandes
passent aussi**. Aucun N intermédiaire n'est interpolé.

Les min/max désignent les **bornes percentiles de l'enveloppe**, pas les
couleurs extrêmes des événements individuels. La validité des percentiles
reste celle de `build_color_envelopes.py` (bootstrap et effectif dans les
queues). Le bootstrap interne et les répétitions de sous-échantillonnage ont
des fonctions différentes.

`--bound-mode adaptive` compare les bornes effectivement sélectionnées pour
chaque taille, comme dans les enveloppes usuelles. Les percentiles peuvent
donc différer entre N et la référence. `--bound-mode reference-percentiles`
impose, dans chaque bin, la même paire percentile que la référence ; si
cette paire n'est pas statistiquement valide pour N, ce bin échoue.

p50-p50 est exclu par défaut puisqu'il n'a pas de largeur percentile. Pour
reproduire le comportement de l'ancien critère adaptatif, fournir
`--allow-degenerate-envelope`. Il faut alors préciser qu'une médiane seule
peut compter comme bin valide. Un intervalle percentile non nul peut aussi
avoir une largeur numérique nulle si les couleurs sont identiques : il reste
valide, car le critère d'exclusion porte sur les percentiles.

## Commande

La commande publique est `analyze_envelope_population_convergence_fast.py`,
à la racine du dépôt. Le générateur et le builder n'ont pas à être remplacés.

```bash
python analyze_envelope_population_convergence_fast.py \
  --run-dir runs/500000_40.7Mpc_intrinsic_seed_12345_sky_seed_67890 \
  --color-pair g-r --photometric-system lsst --stage m5_with_mw \
  --sample-sizes 1000,10000,50000,75000,100000,200000,300000,400000,450000,500000 \
  --repetitions 10 --subsample-seed 20261002 --builder-seed 12345 \
  --t-min 0.4 --t-max 16.0 --bin-width 0.4 --n-boot 200 \
  --m5-scenario best_p75 --threshold-scope common_global \
  --stability-metric colour --relative-tolerance 0.10 \
  --bound-mode adaptive --required-fraction 0.9 \
  --output-dir runs/500000_40.7Mpc_intrinsic_seed_12345_sky_seed_67890/analysis/convergence_colours_g_r
```

Pour une référence à 100 000, remplacer la grille par
`1000,10000,50000,75000,100000`. Si le run sauvegardé en contient davantage,
la référence est alors un sous-échantillon de 100 000 par répétition.

## Sorties

- `envelope_colour_convergence.png` : écart relatif maximal entre toutes les
  valeurs comparées (médiane et 16-84% entre répétitions), puis fraction des
  répétitions passant le critère. Une ligne grise = une répétition. Les erreurs
  infinies dues aux bins absents ou aux références nulles sont plafonnées
  explicitement sur le graphe ; les CSV conservent l'infini.
- `envelope_colour_convergence_summary.csv` : résultats par taille et seuil.
- `envelope_colour_convergence_raw.csv` : résultats par taille/répétition.
- `envelope_colour_convergence_per_bin.csv` : les trois couleurs de chaque
  candidat et de la référence, erreurs et validité, pour chaque bin testé.
- `envelope_colour_profiles.csv` : profils par taille/répétition, avec
  percentiles, effectifs et motifs d'invalidité.
- `envelope_colour_reference.csv` : bins et valeurs de référence.
- `envelope_colour_convergence_results.json` : paramètres et définition du
  critère.

Relancer la même commande reprend les profils complets déjà calculés. Les
anciens checkpoints de temps seul ne contiennent pas les couleurs et ne sont
pas réutilisés. Utiliser un **nouveau dossier de sortie**. Une configuration
différente est refusée en reprise ; choisir un autre dossier ou `--no-resume`.

Au plus grand N, les profils sont comparés à eux-mêmes. Ce point passe donc
par construction : un seuil égal au N maximal n'établit pas de plateau. La
convergence mesurée est conditionnelle à la plus grande population sauvegardée,
à ses priors, à la distance, aux filtres, au stade et à la fenêtre temporelle.
Une tolérance sur des couleurs en magnitudes n'est pas une tolérance identique
sur les flux ni sur leur rapport. Le plancher absolu, s'il est utilisé, doit
figurer dans toute présentation du critère.

## Vérification

```bash
python -m py_compile analyze_envelope_population_convergence_fast.py envelope_colour_convergence.py
PYTHONPATH=. python -m unittest discover -s tests -p 'test_envelope_colour_convergence.py' -v
```
