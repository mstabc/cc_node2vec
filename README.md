# cc_node2vec

This repository evaluates discovered Petri net models against event logs using graph construction, random walks, Word2Vec embeddings, Procrustes alignment, and weighted distance scoring.

## Required Inputs

Reproducing the experiments requires both:

- event logs in `.xes` format
- discovered process models in `.pnml` format

The `.pnml` files used in our experiments are included in this repository under `data/<dataset>/`.

The `.xes` files are not fully bundled here because of repository size limits. You must place one `.xes` log inside each dataset folder, for example:

```text
data/
  01-Sepsis/
    data.xes
    data_ilp_reduced.pnml
    data_inductive_reduced.pnml
    data_heuristics_reduced.pnml
    data_split_reduced.pnml
    data_gcn_*.pnml
```

The benchmark repository of Sommers et al. can be used as a convenient download source, but it is not a strict dependency of this framework. If that source is unavailable, the event logs can still be obtained from public benchmark repositories such as 4TU, and process models can be rediscovered with standard tools such as PM4Py.

## Setup

Use Python 3 and install the required packages:

```bash
pip install pm4py gensim networkx numpy pandas scipy matplotlib scikit-learn
```

## Run

Main entry point:

```bash
python src/main.py --alpha 1.0 --dim 128 --min_count 200 --dataset all --method all --scenario both --lambda_val inf --output_dir results/ --disable_pca_plots
```

Useful options:

- `--dataset all` or a single dataset such as `sepsis`
- `--method all` or one of `ilp`, `inductive`, `heuristics`, `split`, `gnn`
- `--scenario static`, `context_aware`, or `both`
- `--disable_pca_plots` for batch runs

## Batch Runs

Baseline run:

```bash
bash ./run_baseline.sh
```

Ablation run:

```bash
bash ./run_ablation.sh
```

On Windows PowerShell, you can also run the main command directly instead of using `bash`.

## Output

Main results are written to:

```text
results/results_master.csv
```

Summary tables can be generated with:

```bash
python summarize_results.py --input_csv results/results_master.csv --output_dir results/
```

## Reference

Sommers, V. Menkovski, and D. Fahland, "Supervised learning of process discovery techniques using graph neural networks," Information Systems, vol. 115, 2023.
