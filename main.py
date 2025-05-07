import os
import sys
import pandas as pd
from scipy import stats
from pandas import DataFrame
import argparse
from argparse import ArgumentParser
import wandb
from tqdm import tqdm
import torch
from model.data import str2data
from torch import nn, optim
from model.JTAE.models_condif_1d import jtae
from model.ConDiff.Scheduler import GradualWarmupScheduler
from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--dataset", default="GFP", type=str)
    parser.add_argument("--part", default="train", type=str)  # train val test
    parser.add_argument("--sav_dir", default="./train_logs/", type=str)
    parser.add_argument("--input_dim", default=24, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--project_name", default="JT-AE", type=str)
    parser.add_argument("--use_wandb", default=False, type=bool)

    # jt-ae
    parser.add_argument("--alpha_val", default=1.0, type=float)
    parser.add_argument("--beta_val", default=0.0005, type=float)
    parser.add_argument("--gamma_val", default=1.0, type=float)
    parser.add_argument("--sigma_val", default=1.5, type=float)
    parser.add_argument("--eta_val", default=0.001, type=float)
    parser.add_argument("--reg_ramp", default=False, type=str2bool)
    parser.add_argument("--vae_ramp", default=True, type=str2bool)
    parser.add_argument("--wl2norm", default=False, type=str2bool)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--n_epochs", default=500, type=int)
    parser.add_argument("--dev", default=False, type=str2bool)
    parser.add_argument("--seq_len", default=0, type=int)
    parser.add_argument("--embedding_dim", default=100, type=int)
    parser.add_argument("--kernel_size", default=4, type=int)
    parser.add_argument("--latent_dim", default=64, type=int)
    parser.add_argument("--hidden_dim", default=200, type=int)
    parser.add_argument("--layers", default=6, type=int)
    parser.add_argument("--probs", default=0.2, type=float)
    parser.add_argument("--auxnetwork", default="dropout_reg", type=str)

    # diffusion
    parser.add_argument("--dif_T", default=500, type=int, help="condiffusion sample steps")
    parser.add_argument("--dif_channel", default=128, type=int, help="channels for unet")
    parser.add_argument("--dif_channel_mult", default=[1, 2, 2, 2], type=list,
                        help="architecture params for unet")
    parser.add_argument("--dif_res_blocks", default=2, type=int)
    parser.add_argument("--dif_dropout", default=0.15, type=float)
    parser.add_argument("--dif_multiplier", default=2.5, type=float, help="multiplier for lr warmup")
    parser.add_argument("--dif_beta_1", default=1e-4, type=float)
    parser.add_argument("--dif_beta_T", default=0.028, type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dif_w", default=1.8, type=float,
                        help="hyperparameter for classifier-free condiffusion strength")
    parser.add_argument("--num_labels", default=0,
                        type=int)

    # mode
    parser.add_argument("--mode", default="train", type=str)  # or eval
    parser.add_argument("--training_load_epoch", default=None, type=str)
    parser.add_argument("--device_id", default=[0, 1, 2, 3], type=list)
    parser.add_argument("--multiplier", default=2.5, type=float)
    parser.add_argument("--eval_load_epoch", default=None, type=int)
    parser.add_argument("--multi_gpu", default=True, type=bool)

    # sample
    parser.add_argument("--load_path", default="./train_logs", type=str)
    parser.add_argument("--dif_sample_size", default=64, type=int)
    parser.add_argument("--dif_sample_label", default=None, type=int)
    parser.add_argument("--dif_sample_epoch", default=None, type=int)
    parser.add_argument("--dif_outlier_step", default=0, type=int)
    parser.add_argument("--identity", default=False, type=bool)

    args = parser.parse_args()
    return args


def get_pearson_r2(x, y):
    return stats.pearsonr(x, y)[0]


def get_spearman_r2(x, y):
    return stats.spearmanr(x, y)[0]


SEQ2IND = {"I": 0,
           "L": 1,
           "V": 2,
           "F": 3,
           "M": 4,
           "C": 5,
           "A": 6,
           "G": 7,
           "P": 8,
           "T": 9,
           "S": 10,
           "Y": 11,
           "W": 12,
           "Q": 13,
           "N": 14,
           "H": 15,
           "E": 16,
           "D": 17,
           "K": 18,
           "R": 19,
           "X": 20,
           "J": 21,
           "*": 22,
           "-": 23
           }  # J = padding, * = any amino

IND2SEQ = {ind: AA for AA, ind in SEQ2IND.items()}


def inds_to_seq(seq):
    return [IND2SEQ[int(i)] for i in seq]


def seq_to_inds(seq):
    return [SEQ2IND[i] for i in seq]


def get_all_fitness_pred_metrics(targets_list, predictions_list):
    train_targs = targets_list.cpu().detach()
    train_fit_pred = predictions_list.cpu().detach()

    # R2
    train_p_r_val = get_pearson_r2(train_targs.numpy().flatten(),
                                   train_fit_pred.numpy().flatten())

    train_s_r_val = get_spearman_r2(train_targs.numpy().flatten(),
                                    train_fit_pred.numpy().flatten())

    # MSE
    train_mse = nn.MSELoss()(train_fit_pred.squeeze(), train_targs.squeeze())

    # L1
    train_l1 = nn.L1Loss()(train_fit_pred.squeeze(), train_targs.squeeze())
    return train_p_r_val, train_s_r_val, train_mse, train_l1


def loss_function(args, predictions, targets, z_rep, diff_loss):
    # unpack everything
    x_hat, y_hat = predictions
    x_true, y_true, labels = targets
    seq_len = x_true.shape[1]
    x_true = x_true.to(device=x_hat.device)
    y_true = y_true.to(device=y_hat.device)
    # lower weight of padding token in loss
    ce_loss_weights = torch.ones(args.input_dim).cuda()
    ce_loss_weights[21] = 0.8
    ae_loss = nn.CrossEntropyLoss(weight=ce_loss_weights)(x_hat, x_true)
    ae_loss = args.gamma_val * ae_loss

    # enrichment pred loss
    reg_loss = nn.MSELoss()(y_hat.flatten(), y_true.flatten())
    reg_loss = args.alpha_val * reg_loss

    # RAE L_z loss
    # only penalize real zs
    zrep_l2_loss = 0.5 * torch.norm(z_rep, 2, dim=1) ** 2

    if args.wl2norm:
        y_true_shift = y_true + torch.abs(y_true.min())
        w_fit_zrep = nn.ReLU()(y_true_shift / y_true_shift.sum())
        zrep_l2_loss = torch.dot(zrep_l2_loss.flatten(), w_fit_zrep.flatten())
    else:
        zrep_l2_loss = zrep_l2_loss.mean()

    zrep_l2_loss = zrep_l2_loss * args.eta_val
    diff_loss = diff_loss * args.sigma_val

    # MSA LOSS
    if args.dataset == "MSA" or args.dataset == "MSA_RAW" or args.dataset == "MDH":
        total_loss = ae_loss + zrep_l2_loss + diff_loss
    else:
        total_loss = ae_loss + reg_loss + zrep_l2_loss + diff_loss

    seq_difference = x_true.to(device=x_hat.device) - x_hat.argmax(dim=1)
    seq_difference = seq_difference.count_nonzero() / args.batch_size / seq_len
    if args.use_wandb:
        wandb.log({"seq difference": seq_difference})
        wandb.log({"diff_loss": diff_loss})
        wandb.log({"ae_loss": ae_loss})
        wandb.log({"reg_loss": reg_loss})
        wandb.log({"z_loss": zrep_l2_loss})
        wandb.log({"total_loss": total_loss})

    return total_loss


def train(args):
    print("training start!")
    save_dir = f"train_logs/{args.dataset}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    proto_data = str2data(args.dataset)

    data = proto_data(
        dataset=args.dataset,
        batch_size=args.batch_size
    )
    args.seq_len = data.seq_len
    dataloader = data.train_dataloader()
    model = jtae(hparams=args).to(device=args.device)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    if args.training_load_epoch is not None:
        if args.multi_gpu:
            model.module.load_state_dict(torch.load(os.path.join(
                save_dir, 'epoch_' + str(args.training_load_epoch) + '.pt')),
                strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(
                save_dir, 'epoch_' + str(args.training_load_epoch) + '.pt')),
                strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.n_epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier,
                                             warm_epoch=args.n_epochs // 10, after_scheduler=cosineScheduler)
    for e in range(args.n_epochs):
        with tqdm(dataloader, dynamic_ncols=True, leave=False) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                optimizer.zero_grad()
                data, *targets = batch

                # data = data.cuda()
                preds, z_rep, diff_loss = model(batch)
                diff_loss = diff_loss.mean()
                train_loss = loss_function(
                    args=args, predictions=preds, targets=targets,
                    z_rep=z_rep.to("cpu"), diff_loss=diff_loss.to("cpu"))
                train_loss.backward()
                optimizer.step()
                tqdmDataLoader.set_description(f'Epoch [{e}/{args.n_epochs}]')
        warmUpScheduler.step()
        if args.use_wandb:
            wandb.log({"total_train_loss": train_loss}, step=e)
        if (e + 1) % 50 == 0:
            if args.multi_gpu:
                torch.save(model.module.state_dict(),
                           os.path.join(args.sav_dir,
                                        args.dataset + '/dropout_epoch_' + str(e + 1) + ".pt"))
            else:
                torch.save(model.state_dict(),
                           os.path.join(args.sav_dir, args.dataset + '/dropout_epoch_' + str(e + 1) + ".pt"))
        elif e == 0:
            if args.multi_gpu:
                torch.save(model.module.state_dict(),
                           os.path.join(args.sav_dir,
                                        args.dataset + '/dropout_epoch_' + str(e + 1) + ".pt"))
            else:
                torch.save(model.state_dict(),
                           os.path.join(args.sav_dir, args.dataset + '/dropout_epoch_' + str(e + 1) + ".pt"))
    print("training end!")


def eval(args):
    print("evaluating start!")
    save_dir = f"test_output/{args.dataset}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device(args.device)
    proto_data = str2data(args.dataset)
    data = proto_data(
        dataset=args.dataset,
        batch_size=args.batch_size
    )
    args.seq_len = data.seq_len
    dataloader = data.test_dataloader()
    with torch.no_grad():
        model = jtae(args).to(device)
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
        list_metrics = []
        index = []
        pred_y_list = torch.empty(0).to(device)
        pred_x_list = torch.empty(0).to(device)
        label_list = torch.empty(0).to(device)
        test_targs = torch.empty(0).to(device)
        ckpt = torch.load(os.path.join(
            args.sav_dir, args.dataset + '/dropout_tiny_epoch_' + str(args.eval_load_epoch) + ".pt"))
        model.module.load_state_dict(ckpt)
        model.eval()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                _, *targets = batch
                _, y_true, labels = targets
                test_targs = torch.cat((test_targs, y_true.to(device)), dim=0)
                preds, z_rep, diff_loss = model(batch)
                x_hat, y_hat = preds
                pred_x_list = torch.cat((pred_x_list, x_hat), dim=0)
                pred_y_list = torch.cat((pred_y_list, y_hat), dim=0)
                label_list = torch.cat((label_list, labels.to(device)), dim=0)
        pearson, spearman, mse, L1 = get_all_fitness_pred_metrics(targets_list=test_targs,
                                                                    predictions_list=pred_y_list)
        list_metrics.append([pearson, spearman, mse.cpu().numpy(), L1.cpu().numpy()])
        index.append(args.eval_load_epoch)
        dataset_seq_attribute = []
        for i in range(data.test_N):
            dataset_seq_attribute.append('test')
        test_seq_dict = {
            "fitness": pred_y_list.squeeze(1).detach().cpu(),
            "label": label_list.detach().cpu(),
            "attribute": dataset_seq_attribute
        }
        torch.save(pred_x_list, os.path.join(save_dir, "epoch_" + str(args.eval_load_epoch) + ".pt"))
        test_seq_df = pd.DataFrame(test_seq_dict)
        test_seq_df.to_csv(os.path.join(save_dir, "epoch" + str(args.eval_load_epoch) + ".csv"))

        column = ["pearson", "spearman", "mse", "L1"]
        df = DataFrame(list_metrics, columns=column, index=index)
        df.to_csv(os.path.join(save_dir, "dropout_pred_metrics.csv"))
    print("evaluating end!")


def sample(args):
    device = torch.device(args.device)
    proto_data = str2data(args.dataset)
    data = proto_data(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )
    args.seq_len = data.seq_len
    model_no_dp = jtae(args).to(device)
    for label in range(args.dif_sample_label, args.dif_sample_label + 1):
        for epoch in range(args.dif_sample_epoch, args.dif_sample_epoch + 1):
            ckpt = torch.load(os.path.join(
                args.load_path, args.dataset + '/dropout_tiny_epoch_' + str(epoch) + ".pt"))
            model_no_dp.load_state_dict(ckpt)
            with torch.no_grad():
                diff_model = model_no_dp.diff_model
                save_dir = f"./generated_seq/{args.dataset}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                labels = (label * torch.ones(args.dif_sample_size)).long().to(device)
                sampler = GaussianDiffusionSampler(
                    diff_model, args.dif_beta_1, args.dif_beta_T, args.dif_T, args).to(device)

                noisy_z = torch.randn(
                    size=[args.dif_sample_size, 1, args.latent_dim], device=device)
                sampled_z = sampler(noisy_z, labels)
                sampled_z = torch.squeeze(sampled_z, 1).to(torch.float32)
                decoder = model_no_dp.decode
                regressor = model_no_dp.regressor_module
                x_hat = decoder(sampled_z)
                x_hat = x_hat.argmax(dim=1)
                print(x_hat, x_hat.shape)
                pred_seq = []
                if args.identity:
                    for seq in x_hat:
                        seq = "".join(inds_to_seq(seq))
                        seq = seq.replace("J", "")
                        seq = seq.replace("X", "")
                        seq = seq.replace("*", "")
                        seq = seq.replace("-", "")
                        pred_seq.append(seq)
                else:
                    for seq in x_hat:
                        seq = "".join(inds_to_seq(seq))
                        pred_seq.append(seq)

                pred_fit = regressor(sampled_z)
                pred_fit = list(pred_fit.squeeze(dim=1).cpu().numpy())
                print("fit max: ", max(pred_fit))
                print("fit min: ", min(pred_fit))
                # sys.exit()
                data = {}
                data['pred_seq'] = pred_seq
                data['pred_fit'] = pred_fit
                df = DataFrame(data)
                if args.identity:
                    df.to_csv(os.path.join(save_dir, "1epoch_" + str(epoch) + "label_" + str(label) + ".csv"))
                else:
                    df.to_csv(
                        os.path.join(save_dir, "full_tiny_epoch_" + str(epoch) + "label_" + str(label) + ".csv"))


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    args = parser()

    if args.dataset == "NESP" or args.dataset == "ube4b":
        args.num_labels = 5
    elif args.dataset == "MSA" or args.dataset == "MSA_RAW" or args.dataset == "MDH":
        args.num_labels = 0
    else:
        args.num_labels = 8
    if args.use_wandb:
        wandb.init(name=args.dataset, project=args.project_name)
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval(args)
    else:
        sample(args)
