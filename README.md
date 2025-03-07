# Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution

Code accompanying the [NeurIPS 2022 paper](https://neurips.cc/virtual/2022/poster/53227) ([PDF](https://openreview.net/pdf?id=vRrFVHxFiXJ)).

![architecture of CCPA](docs/chemCPA.png)

Our talk on chemCPA at the M2D2 reading club is available [here](https://m2d2.io/talks/m2d2/predicting-single-cell-perturbation-responses-for-unseen-drugs/).
A [previous version](https://arxiv.org/abs/2204.13545) of this work was a spotlight paper at ICLR MLDD 2022.
Code for this previous version can be found under the `v1.0` git tag.

## Codebase overview

For the final models, we provide [weight checkpoints](https://f003.backblazeb2.com/file/chemCPA-models/chemCPA_models.zip) as well as the [hyperparameter configuration](https://f003.backblazeb2.com/file/chemCPA-models/finetuning_num_genes.json).
The raw datasets can be downloaded from a [FAIR server](https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar).
We also provide our processed datasets for reproducibility: sci-Plex [shared gene set](https://f003.backblazeb2.com/file/chemCPA-datasets/sciplex_complete_middle_subset_lincs_genes.h5ad) & [extended gene set](https://f003.backblazeb2.com/file/chemCPA-datasets/sciplex_complete_middle_subset.h5ad), [LINCS](https://f003.backblazeb2.com/file/chemCPA-datasets/lincs_full.h5ad.gz). Embeddings can be downloaded [here](https://drive.google.com/drive/folders/1KzkhYptcW3uT3j4GQpDdAC1DXEuXe49J?usp=share_link).

To setup the environment, install conda and run:

```python
conda env create -f environment.yml
python setup.py install -e .
```

``` sh
conda create -n chemCPA python=3.8 -y
conda activate chemCPA 
conda install scanpy=1.8.1 -c conda-forge -y
conda install pytorch=1.12.0 cudatoolkit=10.2 -c pytorch -y
conda install seml=0.3.5 sacred=0.8.4 -c conda-forge -y
conda install rdkit=2021.09.2 -c conda-forge -y
conda install pyarrow=5.0.0 -c conda-forge
#conda install fastparquet=0.7.1 -c conda-forge -y
#pip install ipywidgets # Preprocessing 1_lincs.ipynb

# lincs_full.h5ad
wget https://f003.backblazeb2.com/file/chemCPA-datasets/lincs_full.h5ad.gz # to datasets/
wget https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar
tar -xvf cpa_binaries.tar
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo.txt.gz
gzip -dk GSE92742_Broad_LINCS_pert_info.txt.gz

```

- `chemCPA/`: contains the code for the model, the data, and the training loop.
- `embeddings`: There is one folder for each molecular embedding model we benchmarked. Each contains an `environment.yml` with dependencies. We generated the embeddings using the provided notebooks and saved them to disk, to load them during the main training loop.
- `experiments`: Each folder contains a `README.md` with the experiment description, a `.yaml` file with the seml configuration, and a notebook to analyze the results.
- `notebooks`: Example analysis notebooks.
- `preprocessing`: Notebooks for processing the data. For each dataset there is one notebook that loads the raw data.
- `tests`: A few very basic tests.

All experiments where run through [seml](https://github.com/TUM-DAML/seml).
The entry function is `ExperimentWrapper.__init__` in `chemCPA/seml_sweep_icb.py`.
For convenience, we provide a script to run experiments manually for debugging purposes at `chemCPA/manual_seml_sweep.py`.
The script expects a `manual_run.yaml` file containing the experiment configuration.

All notebooks also exist as Python scripts (converted through [jupytext](https://github.com/mwouts/jupytext)) to make them easier to review.

Some of the notebooks use a *drugbank_all.csv* file, which can be downloaded from [here](https://go.drugbank.com/) (registration needed).

## Citation

You can cite our work as:

```
@inproceedings{hetzel2022predicting,
  title={Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution},
  author={Hetzel, Leon and Böhm, Simon and Kilbertus, Niki and Günnemann, Stephan and Lotfollahi, Mohammad and Theis, Fabian J},
  booktitle={NeurIPS 2022},
  year={2022}
}
```
