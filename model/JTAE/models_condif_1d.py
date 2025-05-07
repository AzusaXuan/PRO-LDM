import numpy as np
import torch
from torch import nn
import argparse
from model.JTAE.bneck import BaseBottleneck
from model.JTAE.convolutional import Block
from model.JTAE.transformers import PositionalEncoding, TransformerEncoder
from model.JTAE.auxnetwork import str2auxnetwork
from model.ConDiff.DiffusionFreeGuidence.ModelCondition import UNet
from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer
import sys


class jtae(nn.Module):
    def __init__(self, hparams):
        super(jtae, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams
        self.dataset = hparams.dataset
        self.num_labels = hparams.num_labels
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim  # nhid
        self.embedding_dim = hparams.embedding_dim
        self.kernel_size = hparams.kernel_size
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = 4
        self.src_mask = None
        self.bz = hparams.batch_size

        self.lr = hparams.lr
        self.model_name = "jtae"

        self.alpha = hparams.alpha_val
        self.gamma = hparams.gamma_val

        self.sigma = hparams.sigma_val
        self.diff_loss = 0
        self.device = hparams.device
        try:
            self.eta = hparams.eta_val
        except:
            self.eta = 1.0

        self.seq_len = hparams.seq_len

        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)  # input_dim: amino types

        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim, max_len=self.seq_len
        )

        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.embedding_dim, 1), nn.Softmax(dim=1)
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.embedding_dim,
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )
        self.diff_model = UNet(T=hparams.dif_T, num_labels=hparams.num_labels, ch=hparams.dif_channel,
                               ch_mult=hparams.dif_channel_mult, num_res_blocks=hparams.dif_res_blocks,
                               dropout=hparams.dif_dropout)
        self.Gaussian_Diffusion_Trainer = GaussianDiffusionTrainer(model=self.diff_model, beta_1=hparams.dif_beta_1,
                                                                   beta_T=hparams.dif_beta_T, T=hparams.dif_T)

        # make decoder)
        self._build_decoder(hparams)

        # for los and gradient checking
        self.z_rep = None

        # auxiliary network
        self.bottleneck_module = BaseBottleneck(self.embedding_dim, self.latent_dim)

        aux_params = {"latent_dim": self.latent_dim, "probs": hparams.probs}
        aux_hparams = argparse.Namespace(**aux_params)

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            # print(auxnetwork)
            self.regressor_module = auxnetwork(aux_hparams)
        except:
            auxnetwork = str2auxnetwork("spectral")
            self.regressor_module = auxnetwork(aux_hparams)

    def _generate_square_subsequent_mask(self, sz):
        """create mask for transformer
        Args:
            sz (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S )
        """
        mask = torch.ones((sz, sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _build_decoder(self, hparams):

        dec_layers = [
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            Block(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1),
        ]
        self.dec_conv_module = nn.ModuleList(dec_layers)

    def transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            self.src_mask = self._generate_square_subsequent_mask(
                embedded_batch.size(1)
            ).to(self.device)
        pos_encoded_batch = self.pos_encoder(embedded_batch)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)
        return output_embed

    def encode(self, batch):

        embedded_batch = self.embed(batch)
        output_embed = self.transformer_encoding(embedded_batch)
        glob_attn = self.glob_attn_module(output_embed)  # output should be B x S x 1
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # to regain the batch dimension
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)
        return z_rep

    def decode(self, z_rep):

        h_rep = z_rep  # B x 1 X L
        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.seq_len)
            h_rep = layer(h_rep)
        return h_rep

    def diff_train(self, z_rep, labels):

        b = z_rep.shape[0]
        if np.random.rand() < 0.1:
            labels = torch.zeros_like(labels)
        labels = labels.long()
        loss = self.Gaussian_Diffusion_Trainer(z_rep, labels)
        loss = loss.sum() / b ** 2.
        return loss

    def forward(self, batch):
        batch, *targets = batch
        batch = batch.cuda()
        _, _, labels = targets
        z_rep = self.encode(batch)
        z_rep = torch.unsqueeze(z_rep, 1)
        diff_loss = self.diff_train(z_rep, labels)
        z_rep = torch.squeeze(z_rep, 1).to(torch.float32)
        x_hat = self.decode(z_rep)
        y_hat = self.regressor_module(z_rep)
        return [x_hat, y_hat], z_rep, diff_loss
