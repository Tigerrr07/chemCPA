from pathlib import Path
from pprint import pprint
from chemCPA.model import MLP, ComPert
import torch
from chemCPA.data import SubDataset
from seml.config import generate_configs, read_config
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import json

from chemCPA.experiments_run import ExperimentWrapper

def compute_r2(y_true, y_pred):
    """
    Computes the r2 score for `y_true` and `y_pred`,
    returns `-1` when `y_pred` contains nan values
    """
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    # metric = R2Score().to(y_true.device)
    # metric.update(y_pred, y_true)  # same as sklearn.r2_score(y_true, y_pred)
    score = r2_score(y_pred.cpu().numpy(), y_true.cpu().numpy())
    return score


def bool2idx(x):
    """
    Returns the indices of the True-valued entries in a boolean array `x`
    """
    return np.where(x)[0]

def repeat_n(x, n):
    """
    Returns an n-times repeated version of the Tensor x,
    repetition dimension is axis 0
    """
    # copy tensor to device BEFORE replicating it n times
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device).view(1, -1).repeat(n, 1)

def compute_prediction(autoencoder: ComPert, genes, emb_drugs, emb_covs):
    """
    Computes the prediction of a ComPert `autoencoder` and
    directly splits into `mean` and `variance` predictions
    """
    if autoencoder.use_drugs_idx:
        assert len(emb_drugs) == 2
        genes_pred = autoencoder.predict(
            genes=genes,
            drugs_idx=emb_drugs[0],
            dosages=emb_drugs[1],
            covariates=emb_covs,
        )[0].detach()
    else:
        genes_pred = autoencoder.predict(
            genes=genes, drugs=emb_drugs, covariates=emb_covs
        )[0].detach()
    dim = genes.size(1)
    mean = genes_pred[:, :dim]
    var = genes_pred[:, dim:]
    return mean, var

def fetch_predicted(autoencoder, dataset, genes_control):
    """
    Evaluate on ood dataset:
    Use test control (ood control has the same size)
    """
    
    pred_dict = {}
    autoencoder.eval()
    with torch.no_grad():
        n_rows = genes_control.size(0)
        genes_control = genes_control.to(autoencoder.device)
        pert_categories_index = pd.Index(dataset.pert_categories, dtype="category")
        for cell_drug_dose_comb, category_count in zip(
            *np.unique(dataset.pert_categories, return_counts=True)
        ):
            mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
            if dataset.perturbation_key is None:
                break

            # estimate metrics only for reasonably-sized drug/cell-type combos
            if category_count <= 5:
                continue

            # doesn't make sense to evaluate DMSO (=control) as a perturbation
            if (
                "dmso" in cell_drug_dose_comb.lower()
                or "control" in cell_drug_dose_comb.lower()
            ):
                continue

            # dataset.var_names is the list of gene names
            # dataset.de_genes is a dict, containing a list of all differentiably-expressed
            # genes for every cell_drug_dose combination.
            bool_de = dataset.var_names.isin(
                np.array(dataset.de_genes[cell_drug_dose_comb])
            )
            idx_de = bool2idx(bool_de)

            # need at least two genes to be able to calc r2 score
            if len(idx_de) < 2:
                continue

            bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
            idx_all = bool2idx(bool_category)
            idx = idx_all[0]

            # NOTE: Repeat (drugs, covs, dose) to the size of genes_control
            # NOTE: genes_control as raw input, (drugs, covs, dose) input in latent space
            emb_covs = [repeat_n(cov[idx], n_rows) for cov in dataset.covariates]
            if dataset.use_drugs_idx:
                emb_drugs = (
                    repeat_n(dataset.drugs_idx[idx], n_rows).squeeze(),
                    repeat_n(dataset.dosages[idx], n_rows).squeeze(),
                )
            else:
                emb_drugs = repeat_n(dataset.drugs[idx], n_rows)
            mean_pred, var_pred = compute_prediction(
                autoencoder,
                genes_control,
                emb_drugs,
                emb_covs,
            )

            # copies just the needed genes to GPU
            # Could try moving the whole genes tensor to GPU once for further speedups (but more memory problems)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            y_true = dataset.genes[idx_all, :].to(device=device)

            # true means and variances
            yt_m = y_true.mean(dim=0)
            yt_v = y_true.var(dim=0)
            # predicted means and variances
            yp_m = mean_pred.mean(dim=0)
            yp_v = var_pred.mean(dim=0)

            r2_m = compute_r2(yt_m, yp_m)
            r2_v = compute_r2(yt_v, yp_v)
            r2_m_de = compute_r2(yt_m[idx_de], yp_m[idx_de])
            r2_v_de = compute_r2(yt_v[idx_de], yp_v[idx_de])

            # to be investigated
            if r2_m_de == float("-inf") or r2_v_de == float("-inf"):
                continue

            # mean_score.append(max(r2_m, 0.0))
            # var_score.append(max(r2_v, 0.0))
            # mean_score_de.append(max(r2_m_de, 0.0))
            # var_score_de.append(max(r2_v_de, 0.0))
            r2_m = max(r2_m, 0.0)
            r2_v = max(r2_v, 0.0)
            r2_m_de = max(r2_m_de, 0.0)
            r2_v_de = max(r2_v_de, 0.0)

            pred_dict[cell_drug_dose_comb] = {
                'mean_pred': mean_pred.cpu().numpy(), 'true': y_true.cpu().numpy(),
                'mean_score': r2_m, 'var_score': r2_v,
                'mean_score_de': r2_m_de, 'var_score_de': r2_v_de}

        return pred_dict
  
if __name__ == "__main__":
    exp = ExperimentWrapper(init_all=False)

    # this is how seml loads the config file internally
    assert Path("manual_eval.yaml").exists(), "config file not found"
    seml_config, slurm_config, experiment_config = read_config("manual_eval.yaml")
    # we take the first config generated
    configs = generate_configs(experiment_config)
    if len(configs) > 1:
        print("Careful, more than one config generated from the yaml file")
    args = configs[0]
    pprint(args)

    exp.seed = 1337
    # loads the dataset splits
    exp.init_dataset(**args["dataset"])

    exp.init_drug_embedding(embedding=args["model"]["embedding"])
    exp.init_model(
        hparams=args["model"]["hparams"],
        additional_params=args["model"]["additional_params"],
        load_pretrained=args["model"]["load_pretrained"],
        append_ae_layer=args["model"]["append_ae_layer"],
        enable_cpa_mode=args["model"]["enable_cpa_mode"],
        pretrained_model_path=args["model"]["pretrained_model_path"],
        pretrained_model_hashes=args["model"]["pretrained_model_hashes"],
    )

    datasets = exp.datasets
    autoencoder = exp.autoencoder

    eval_dataset = datasets["ood"]
    genes_control = datasets["test_control"].genes
    
    pred_dict = fetch_predicted(autoencoder, eval_dataset, genes_control)
    # setup the torch DataLoader
    # exp.update_datasets()

    # exp.train(**args["training"])
    np.savez("prediction_results.npz", **pred_dict)