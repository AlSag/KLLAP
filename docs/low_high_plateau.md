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

## Lancement interactif

Depuis le dépôt, dans l'environnement habituel :

```bash
python analyze_kne_latitude.py
```

Deux modes d'entrée sont proposés :

1. Une population all-sky et son `event_window_availability.csv`, déjà produit
   par `analyze_kne_latitude.py population`. Chaque seuil sépare cette population en
   low et high. Les effectifs ne sont pas forcément égaux.
2. Des populations low/high réellement simulées à chaque seuil, avec leur CSV
   de disponibilité. Le mode apparié nécessite le même catalogue synthétique
   partagé et les mêmes event_id dans **tous** les runs. Les sky seeds peuvent
   varier. L'identité des event_id ne suffit pas à prouver l'appariement : il
   faut avoir réutilisé le même catalogue. Le mode indépendant nécessite des
   populations simulées indépendamment, y compris entre les seuils.

Le script est autonome : numpy, pandas et matplotlib suffisent. Il ne génère
pas de courbes de lumière et ne relit pas les shards Parquet.

## Une population all-sky existante

Lancer le script, choisir « stabilité d'une coupure low/high », puis « une
table all-sky ». Le terminal demande le chemin de
`event_window_availability.csv`, les seuils, le nombre de bootstraps, la
tolérance et le dossier de sortie.

Adapter le chemin du CSV. Pour reprendre une autre analyse couleur, distance,
fenêtre temporelle ou profondeur, produire le CSV correspondant avec l'analyseur
de pertes, puis relancer ce script sur ce CSV.

## Runs appariés low/high existants

Dans le même menu, choisir « runs low/high séparés ». Le terminal demande les
deux tables pour chaque seuil et si les mêmes KNe et `event_id` ont été
réutilisés. Cette réponse détermine le bootstrap apparié ou indépendant.

Un dossier est accepté seulement s'il contient exactement un CSV de
disponibilité ; sinon préciser le CSV. Les configurations scientifiques sont
comparées quand les fichiers `analysis_configuration.json` sont disponibles.
Tous les runs doivent employer les mêmes modèle, distributions de paramètres,
distances, couleur, fenêtre, cadence et seuils m5. Un m5 global commun aide à
isoler l'effet de la poussière. Le script ne vérifie pas automatiquement les
catalogues physiques ni la configuration du générateur.

Les runs en tranches disjointes ([0,5), [5,10), ...) ne sont **pas** des runs
low/high. Les concaténer avec le même poids par tranche changerait la distribution
du ciel ; ce mode d'entrée n'est donc pas proposé.

## Sorties et critère

Seulement deux PNG sont créés :

- `kne_low_high_mw_plateau.png` : une courbe Q, une bande bootstrap, la coupure
  envisagée à 20 degrés, et le plateau compatible s'il est trouvé.
- `kne_low_high_mw_values.png` : pertes et dénominateurs low/high, leurs taux,
  Q et son intervalle, baisse relative, statut du plateau à chaque seuil.

Le plateau est défini **avant** de regarder les résultats. Pour chaque seuil
candidat, le script examine tous les seuils suivants jusqu'au maximum demandé :

    D = max |Q / médiane(Q sur cette suite) - 1|

Par défaut, D doit être <= 10 %, avec au moins trois seuils couvrant au moins
10 degrés, au moins 100 KNe baseline et 20 pertes MW de chaque côté à chaque
seuil. Le script recalcule D à chaque bootstrap ; sa borne percentile supérieure
unilatérale à 95 % doit aussi être <= 10 %. Il s'agit d'un critère opérationnel
conservateur, pas d'une garantie exacte de couverture du bootstrap.

Les intervalles de la courbe sont point par point ; ils ne sont pas des bandes
simultanées. Le bootstrap partagé sur tous les seuils conserve les dépendances
des groupes cumulatifs en mode all-sky, et l'appariement des paramètres en mode
paired. Le test de D utilise cette dépendance sur toute la suite des seuils.

« Plateau non démontré » ne signifie pas qu'aucun plateau existe. La suite
peut continuer à évoluer, ou les événements perdus peuvent être trop rares pour
conclure avec cette tolérance. Il peut être nécessaire d'augmenter le nombre de
KNe ou d'affiner les seuils autour de 15–30 degrés.

Un plateau de Q est un plateau du **contraste entre deux populations cumulatives**.
Le groupe low et le groupe high changent tous les deux quand le seuil augmente.
Il ne démontre pas une frontière physique de la poussière. Une coupure à 20 degrés
peut être motivée par ce contraste sous les hypothèses de simulation ; son coût
en couverture du ciel et en KNe conservées doit être étudié séparément pour
justifier l'optimisation complète des cuts.

## Vérification

```bash
python -m py_compile analyze_kne_latitude.py
python -m unittest discover -s tests -p 'test_latitude_plateau.py' -v
```

Le terminal demande explicitement si les deux PNG existants doivent être
remplacés. Changer le dossier de sortie permet de conserver plusieurs analyses.
