import os
import numpy as np
import torch
from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer


def train(params, z_rep, labels, net_model):
    device = params.device
    # model setup
    if params.dif_training_load_weight is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            params.dif_sav_dir, params.dif_training_load_weight), map_location=device), strict=False)
        print("Model weight load down.")

    trainer = GaussianDiffusionTrainer(
        net_model, params.dif_beta_1, params.dif_beta_T, params.dif_T).to(device)

    # start training
    b = z_rep.shape[0]
    x_0 = z_rep.to(device)
    labels = labels.to(device)
    if np.random.rand() < 0.1:
        labels = torch.zeros_like(labels).to(device)
    labels = labels.long()
    loss = trainer(x_0, labels)
    loss = loss.sum() / b ** 2.

    return loss