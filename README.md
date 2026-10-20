# Interpretable zone-of-influence rules learned from tree networks

This repository contains the code used for the article *Learning interpretable and transferable zone-of-influence rules from networks of interacting trees*.

## Files

- `zoi_core.py` contains the shared methods for loading the networks, fitting zone-of-influence rules by linear programming, the evolutionary search of rule structures, stability selection, robustness certificates, conflict sets, the baselines and the planted rules of the semi-synthetic study.
- `zoi_run.py` is the command-line program for the analyses of the forest networks and for the semi-synthetic experiments.
- `zoi_repeat.py` repeats the transfer experiments of `zoi_run.py` with ten seeds.
- `zoi_blackbox.py` evaluates random forest and gradient boosting classifiers without hyperparameter tuning in the transfer experiments.
- `zoi_blackbox_tuned.py` evaluates the same classifiers with hyperparameters and decision thresholds selected by grouped cross-validation on the training plots.
- `zoi_transfer_summary.py` summarises the repeated transfer experiments over seeds and compares them with the tuned classifiers.
- `zoi_synth_stats.py` summarises the semi-synthetic study with F1 scores, structural recovery, falsely selected and missed terms and exact McNemar tests with Holm correction.
- `zoi_rate.py` simulates how fast the set of rules that reproduce all edges shrinks as the number of trees grows.

## Installation

The code was run with Python 3.11, numpy 2.1.2, pandas 2.3.3, scipy 1.16.2 and scikit-learn 1.7.2. Install the required packages with the command below.

```
pip install -r requirements.txt
```

## Data

The forest networks belong to the Slovenian Forestry Institute and are not included, because the data owner does not permit their redistribution. Access to the data is described in the Data availability statement of the article.

The scripts read the networks from the folder `data`. Another folder can be given with `--data`. Each plot is a text file named after its forest type and plot, for example `ABIES_000.txt`. The part of the name before the underscore is the forest type. The file starts with the line `$vertices`, followed by a whitespace-separated table with a header row and one row per tree. The table has the columns `name`, `x` and `y` with the position in metres, `dv` with the species code and the attributes `dbh`, `h`, `v`, `soc` and `z`. The line `$edges` is followed by a table with the header `from to` and one edge per row, given by the names of its two trees. Species with fewer than 20 trees over all plots are merged into one group.

## Running the experiments

Run the commands from the repository folder in the order given below. They work in Windows cmd and in Unix shells. Each step writes its results to a subfolder of `results`. The option `--jobs` sets the number of parallel processes. Several steps take hours on a 16-core workstation. `zoi_repeat.py` and `zoi_blackbox_tuned.py` skip finished parts when they are restarted.

### Forest networks

In-sample diagnostics of each plot, including the check whether free radii reproduce the network exactly.

```
python zoi_run.py diag --out results/diag --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075
```

Selection of the rule by stability selection, repeated ten times.

```
python zoi_run.py search --out results/search --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075 --runs 25 --evals 300 --delta 0.001 --repeat 10 --jobs 16
```

Coefficients, robustness certificates and conflict sets of the final rule.

```
python zoi_run.py explain --out results/explain --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075 --form "log(dbh)@sp + p067(dbh)@sp"
```

Power-law baseline with species-specific coefficients and a common exponent chosen on a grid, fitted on all plots and in the transfer experiments.

```
python zoi_run.py power --out results/power --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075
```

Transfer to held-out plots and forest types with ten seeds, followed by the black-box classifiers and the summary.

```
python zoi_repeat.py --jobs 16 --parallel 10
python zoi_blackbox.py --features sym --jobs 16
python zoi_blackbox_tuned.py --jobs 16
python zoi_transfer_summary.py
```

The grid for gradient boosting in `zoi_blackbox_tuned.py` is the last of the three grids described in the article. Every setting selected over the combined grid lies in this grid, so a single run gives the reported values.

The sensitivity analysis repeats the selection with other term libraries and tolerances. Library L1 corresponds to `--attrs dbh,h,v,soc,z --transforms id,sq,sqrt,log` and library L2 to `--attrs dbh,h,v,soc,z --transforms id,sq,sqrt,log,p033,p067,p075`. The tolerance is set with `--delta`.

### Semi-synthetic study

The semi-synthetic study plants known rules in the forest networks and tests whether they are recovered. The planted rules are evaluated on the plots whose names end with `_100`. Each of the four commands below is run with `TRUTH` set to `species_dbh`, `sqrt_z` and `pair`, `SIGMA` set to `0`, `0.3` and `1.0` and `WIDTH` set to `0.2`. The last three commands are also run in the noise-free setting with `SIGMA` and `WIDTH` set to `0`.

Stability selection with 200 evaluations per search and a single search with 2000 evaluations.

```
python zoi_run.py synth --out results/synth_m200 --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075 --delta 0.001 --truth TRUTH --sigma SIGMA --width WIDTH --reps 10 --runs 5 --evals 200 --jobs 16
```

Stability selection with 400 evaluations per search, the structure of its stable terms, and the aggregation and union of searches on all data.

```
python zoi_run.py synth --out results/synth_m400 --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075 --delta 0.001 --truth TRUTH --sigma SIGMA --width WIDTH --reps 10 --runs 5 --evals 400 --skip-single --multistart --jobs 16
```

Single search with 4000 evaluations.

```
python zoi_run.py synth --out results/synth_single4000 --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075 --delta 0.001 --truth TRUTH --sigma SIGMA --width WIDTH --reps 10 --runs 5 --evals 400 --skip-stability
```

Stability selection with 800 evaluations per search.

```
python zoi_run.py synth --out results/synth_m800 --attrs dbh,soc,z --transforms id,sq,sqrt,log,p033,p067,p075 --delta 0.001 --truth TRUTH --sigma SIGMA --width WIDTH --reps 10 --runs 5 --evals 800 --skip-single --jobs 16
```

Summary of all four runs.

```
python zoi_synth_stats.py
```

### Identification rate

The simulation of the identification rate needs no data.

```
python zoi_rate.py
```

## Results

The folder `results` contains the outputs of the runs reported in the article. `results/rate` holds the identification-rate simulation from `zoi_rate.py`, `results/synth_stats` the summary of the semi-synthetic study from `zoi_synth_stats.py` and `results/transfer_summary` the summary of the transfer experiments from `zoi_transfer_summary.py`. These files contain only summary values and no tree data, so they can be compared with the article without the forest networks. The steps above write new versions of them to the same folders.
