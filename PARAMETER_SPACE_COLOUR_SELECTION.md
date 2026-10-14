# Espace des paramètres sélectionné par une couleur

Installer `plot_kne_colour_parameter_space.py` à la racine du dépôt
`kne-lsst-analysis`, à côté de `build_color_envelopes.py`,
`dynamic_color_cut_common.py` et `kne_pipeline_common.py`. Aucun script
existant n'est remplacé. Dépendances : NumPy, pandas, Matplotlib et SciPy ;
PyArrow pour les runs Parquet. L'environnement du pipeline contient normalement
ces dépendances. Aucune évaluation FIESTA ni nouvelle simulation n'est requise.

## Mode interactif

```bash
python plot_kne_colour_parameter_space.py
```

Le script demande le dossier du run, la paire de couleurs, le système, le stade,
les temps (défaut 1, 3, 6, 10 jours), les plans de paramètres et le critère
couleur. Il n'impose **aucun intervalle scientifique arbitraire**. Il faut
choisir des bornes, par exemple un intervalle qui représente une couleur
observée avec son incertitude, ou les bornes d'une enveloppe déjà calculée.

## Exemple : intervalle de couleur manuel

Les valeurs suivantes sont uniquement un **exemple de syntaxe** à remplacer,
pas des recommandations de sélection KNe :

```bash
python plot_kne_colour_parameter_space.py \
  --run-dir runs/TON_RUN \
  --photometric-system lsst \
  --color-pair g-r \
  --stage m5_no_mw \
  --times 1,3,6,10 \
  --parameter-pairs Ye_wind:log10_mej_dyn,Ye_dyn:log10_mej_wind \
  --color-ranges=-0.2:0.5,0.3:1.0,0.8:1.5,1.0:2.0 \
  --m5-scenario best_p75 \
  --threshold-scope common_global \
  --output-dir runs/TON_RUN/analysis/parameter_space_g_r_manual
```

Un seul intervalle (`--color-ranges=0.2:0.5`) peut s'appliquer à tous les temps.
Pour sélectionner autour d'une valeur mesurée `C ± sigma_C`, entrer
`C-sigma_C:C+sigma_C`. Le script ne propage pas d'autres erreurs de mesure.

## Exemple : utiliser un fichier de bornes existant

```bash
python plot_kne_colour_parameter_space.py \
  --run-dir runs/TON_RUN \
  --photometric-system lsst \
  --color-pair g-r \
  --stage m5_with_mw \
  --times 1,3,6,10 \
  --parameter-pairs Ye_wind:log10_mej_dyn \
  --bounds-file runs/TON_RUN/analysis/TA_CONVERGENCE \
  --envelope-family p16-p84 \
  --hdr-level 90 \
  --reference-repetition 0 \
  --output-dir runs/TON_RUN/analysis/parameter_space_g_r_p16_p84
```

Formats acceptés :

- `dynamic_color_cut_rules.csv` ou `dynamic_color_cut_rules_all_valid.csv` :
  `t_min_days`, `t_max_days`, `color_min`, `color_max` ; les bornes sont lues
  telles quelles (elles peuvent être adaptatives si le CSV source l'est).
- `percentile_reference_profiles.csv` : `t_low_days`, `t_high_days`,
  `percentile`, `value`, `valid` ; les percentiles fixes demandés sont extraits.
  Un percentile invalide n'est pas utilisé. Lorsque le CSV contient plusieurs
  répétitions, `--reference-repetition` choisit une référence.

`--bounds-file` accepte aussi directement le dossier de convergence. Le script
y découvre automatiquement `percentile_reference_profiles.csv`. Choisir la
famille fixe avec `--envelope-family p5-p95`, `p16-p84` (défaut) ou `p25-p75`.
Le nom des PNG et leur métadonnée `Title` enregistrent la famille utilisée,
mais aucun titre global n'est imprimé sur la figure.

Un CSV de résultats point par point tel que `color_epoch_cut_results.csv`
n'est pas une table de bornes par bin et ne doit pas être donné à
`--bounds-file`.

Le bin contenant le temps est utilisé : intervalle temporel `[t_min,t_max[`,
avec fermeture du dernier bord. On n'extrapole pas une enveloppe manquante
jusqu'à 10 jours. Les bornes provenant d'une référence calculée seulement
jusqu'à 8 jours laisseront donc le panneau 10 jours sans sélection valide.
Les bornes d'un bin sont estimées à son centre dans le builder : appliquer ces
bornes à un autre temps du même bin est une sélection par règle de bin, pas
une interpolation de la valeur du percentile à ce temps exact.

**Vérifier la cohérence** du stade, du système photométrique, de la paire de
couleurs et de la population de référence : un CSV de profils longs ne porte
pas nécessairement ces informations dans ses colonnes.

## Ce qui est calculé

1. Lecture de `summary.csv` : un événement = un poids, aucun split
   train/validation/test. Avec `--max-events 0`, tous les événements injectés
   sont considérés, y compris ceux dont la couleur est indisponible.
2. Lecture unique des deux bandes dans les shards Parquet (ou les CSV legacy).
3. Interpolation des magnitudes sauvegardées linéairement en `log10(t_obs)`
   aux temps demandés, exactement comme dans le pipeline. Pas d'extrapolation.
4. Les magnitudes sauvegardées contiennent MW. Dans les stades sans MW,
   le script soustrait `A_band`, en utilisant en priorité les valeurs stockées.
5. Pour `m5_no_mw` / `m5_with_mw`, les deux bandes doivent satisfaire
   `mag_band <= m5_band`, sur les magnitudes du stade correspondant.
   Les profondeurs LSST viennent de la table existante, sans les recalculer.
   En PS1 avec m5, la couleur est PS1 mais la sélection utilise les magnitudes
   **LSST sauvegardées des mêmes événements**. En ZTF, saisir les deux seuils
   `--ztf-m5-g`, `--ztf-m5-r` ou `--ztf-m5-i` correspondants.
6. Sélection si la couleur est disponible et comprise entre les deux bornes
   inclusives au temps considéré. Chaque temps est indépendant : ce ne sont
   pas forcément les mêmes événements rouges dans les quatre panneaux.
7. Projection de la population sélectionnée dans les plans `(x,y)` choisis.
   Les autres paramètres, distances et extinctions restent distribués comme
   dans le run : la projection les marginalise implicitement.

Le temps par défaut est celui de l'observateur, post-merger, comme les
enveloppes du pipeline. `--time-frame source` convertit les temps demandés
événement par événement avec `t_obs = (1+z)t_source` ; utiliser alors des
bornes manuelles, pas un CSV d'enveloppe en temps observateur.

## Interpréter les points et les contours

- Rouge : événement passant le stade et l'intervalle couleur.
- Gris : couleur disponible, mais hors intervalle.
- Gris clair : couleur indisponible ou rejetée par m5.
- Un seul contour est tracé : par défaut, la région ordonnée par densité
  (HDR empirique) contenant environ 90 % des événements sélectionnés.
  `--hdr-level 50` permet à la place de montrer seulement le noyau le plus
  dense. Les deux niveaux ne sont jamais superposés sur la même figure.

Les contours peuvent avoir plusieurs composantes déconnectées. Ils sont
obtenus par histogramme 2D normalisé, lissé avec un noyau gaussien. On évalue
ensuite la densité interpolée **à la position de chacun des événements
sélectionnés**, puis choisit le seuil de densité incluant 50 ou 90 % de ces
événements. On ne prend **pas** le percentile 50/90 des hauteurs aux cellules
de grille, car cela donnerait une fraction de cellules plutôt qu'une fraction
de population. Le seuil est calibré sur les données sélectionnées : sa
couverture sur ce même échantillon n'est pas une validation indépendante.
Le lissage utilise `scipy.ndimage.gaussian_filter`, avec réflexion aux bords
du rectangle échantillonné pour ne pas perdre la masse au-delà de ses bornes :
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>.

Les pourcentages portent sur la **population sélectionnée ayant les deux
paramètres finis**, pas sur toutes les injections. Ce ne sont pas des
intervalles de confiance et ils ne donnent pas la probabilité qu'un événement
dans le contour passe la sélection. Un point gris peut être à l'intérieur,
parce que ses autres paramètres sont différents. Les contours dépendent des
priors, de la projection et du lissage. Une population sélectionnée presque
uniforme ne définit pas une zone HDR unique et physiquement informative.

`contour_diagnostics.csv` donne la masse du modèle lissé incluse et la fraction
**réellement observée** d'événements dont la densité interpolée dépasse le
seuil. Cette dernière est 50/90 % à l'arrondi d'effectif près (davantage en cas
d'ex aequo). La masse intégrée du modèle lissé peut être différente : il s'agit
d'un contour calibré sur les points, pas d'une intégration exacte d'une KDE
continue. Le tracé Matplotlib reste une approximation sur une grille finie.
Tester par exemple `--grid-size 80/120` et `--smooth-sigma 1.0/2.0` avant
d'interpréter des petites îles comme des structures physiques. Avec moins de
100 sélectionnées (seuil configurable), les points sont affichés mais les
contours sont omis ; ce seuil est une précaution d'affichage, pas une preuve
de significativité statistique.

## Paramètres, axes et modèles

Le plan par défaut est x=`Ye_wind`, y=`log10_mej_dyn` (Bu2026_MLP).
Pour la masse physique en unités solaires, utiliser y=`mej_dyn`.
La densité/HDR est estimée **dans les coordonnées affichées** : une HDR en
`log10(M)` n'est pas la même région qu'une HDR calculée en `M`.
Une prior uniforme en `log10(M)` n'est pas uniforme en `M` ; une inclinaison
isotrope est uniforme en `cos(iota)` mais pas en `iota`.

Autres colonnes possibles : `Ye_dyn`, `v_ej_dyn`, `v_ej_wind`,
`log10_mej_wind`, `inclination_EM`, `luminosity_distance`, `ebv_mw`.
Transformations disponibles : `mej_dyn`, `mej_wind`, `inclination_deg`,
`cos_inclination`. D'autres colonnes numériques de `summary.csv` sont acceptées.
Un paramètre constant ne définit pas un plan 2D et est refusé.

Bu2019_MLP n'a pas `Ye_wind` ou `Ye_dyn` : choisir par exemple
`phi:log10_mej_dyn` ou `log10_mej_wind:log10_mej_dyn`.
Pour Metzger, choisir les paramètres réellement présents dans le run,
par exemple `log10_kappa_r:log10_mej`.

## Sorties et coûts

- Un PNG de quatre panneaux par plan de paramètres, si les quatre temps sont
  demandés.
- `selection_summary.csv` : effectifs disponibles/sélectionnés par temps,
  bornes appliquées et fractions avec dénominateur explicite.
- `contour_diagnostics.csv` : contrôles de couverture des contours.
- `event_colour_selection.csv.gz` : paramètres, couleurs et masques par événement.
- `parameter_space_configuration.json` : conventions, profondeurs, empreintes.

Le calcul n'utilise ni bootstrap ni nouvelle génération. Tous les points
sélectionnés servent aux contours. `--max-scatter-points` limite uniquement
le nombre de points dessinés par catégorie, pas les effectifs ni la densité.
`--max-events`, en revanche, réduit réellement la population analysée.
Le lecteur est compatible avec les shards du pipeline, mais la lecture de
millions de courbes reste la partie la plus coûteuse.

Les fichiers sont sauvegardés dans un dossier distinct du builder/de la
convergence. Un dossier non vide est refusé sans `--overwrite`.

## Vérification

```bash
python -m py_compile plot_kne_colour_parameter_space.py
PYTHONPATH=. python -m unittest discover -s tests -p 'test_kne_colour_parameter_space.py' -v
```

Ce module réutilise les lecteurs et les conventions du pipeline KLLAP,
qui développe notamment un exemple FIESTA. Il ne contient pas les poids
FIESTA/POSSIS ni de fichiers NMMA joblib. Conserver les notices tierces du
dépôt et citer le pipeline ainsi que FIESTA et le modèle physique utilisés.
