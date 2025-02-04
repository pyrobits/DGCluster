# DGCLUSTER: A Neural Framework for Attributed Graph Clustering via Modularity Maximization

This repository is a reference implementation of DGCluster as described in the paper:
<br/>
> DGCLUSTER: A Neural Framework for Attributed Graph Clustering via Modularity Maximization.<br>
> Aritra Bhowmick, Mert Kosan, Zexi Huang, Ambuj Singh, Sourav Medya<br>
> Association for the Advancement of Artificial Intelligence (AAAI), 2024.<br>
> [https://ojs.aaai.org/index.php/AAAI/article/view/28983](https://ojs.aaai.org/index.php/AAAI/article/view/28983)

## Requirements

- Install Torch from https://pytorch.org/get-started/locally/
- Run `install.py` to install the required packages after installing Torch.

```setup
python install.py
```

or

- Install the required packages from `env.yml` file with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```setup
conda env create -f environment.yml
```

## Reproducing Results

To generate the result for a specific dataset (e.g., cora) and lambda parameter (e.g., 0.2), run the following:

```train
python main.py --dataset cora --lam 0.2
```

We also provide shell scripts `run.sh`, `run_alp.sh`, `run_base_all.sh` to reproduce all results data for all datasets and parameters. You can just run the following commands to generate the data for the corresponding tables or figures.

- `./run.sh`: Table 2, Table 3, and Figure 1.
- `./run_alp.sh`: Table 5.
- `./run_base_all.sh`: Table 4.

Additionally, `plots.py` and `plots_num_clusters.py` can be used to generate Figure 1 and Figure 2.


 
