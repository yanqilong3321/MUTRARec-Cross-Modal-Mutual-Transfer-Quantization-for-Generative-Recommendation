import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x,use_sk=use_sk)
        # print(indices.shape)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon

class FusionRQVAE(nn.Module):
    def __init__(self,
                 text_in_dim,
                 vis_in_dim,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 lora_rank=32,
                 lora_alpha=1.0
        ):
        super(FusionRQVAE, self).__init__()
        
        self.text_in_dim = text_in_dim
        self.vis_in_dim = vis_in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.lora_alpha = lora_alpha
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # --- Text Pathway ---
        self.text_encode_layer_dims = [self.text_in_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims, dropout=self.dropout_prob, bn=self.bn)
        
        self.text_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = kmeans_init,
                                          kmeans_iters = kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters,
                                          use_linear=use_linear)
                                          
        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

        # --- Vis Pathway ---
        self.vis_encode_layer_dims = [self.vis_in_dim] + self.layers + [self.e_dim]
        self.vis_encoder = MLPLayers(layers=self.vis_encode_layer_dims, dropout=self.dropout_prob, bn=self.bn)
        
        self.vis_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = kmeans_init,
                                          kmeans_iters = kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters,
                                          use_linear=use_linear)

        self.vis_decode_layer_dims = self.vis_encode_layer_dims[::-1]
        self.vis_decoder = MLPLayers(layers=self.vis_decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

        # --- LoRA Interaction ---
        # A (Up): r -> dim
        # B (Down): dim -> r
        self.text_lora_A = nn.Linear(lora_rank, e_dim, bias=False)
        self.text_lora_B = nn.Linear(e_dim, lora_rank, bias=False)
        
        self.vis_lora_A = nn.Linear(lora_rank, e_dim, bias=False)
        self.vis_lora_B = nn.Linear(e_dim, lora_rank, bias=False)
        
        # Init LoRA
        nn.init.zeros_(self.text_lora_B.weight)
        nn.init.normal_(self.text_lora_A.weight, std=0.02)
        nn.init.zeros_(self.vis_lora_B.weight)
        nn.init.normal_(self.vis_lora_A.weight, std=0.02)
        
    def forward(self, x_text, x_vis, use_sk=True):
        # 1. Encode
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        # 2. Interaction
        # Text fused = Text + alpha * (Vis -> B_vis -> A_text)
        # Vis enters Vis LoRA B (Down) -> Text LoRA A (Up) -> Add to Text
        delta_text = self.text_lora_A(self.vis_lora_B(z_vis))
        z_text_fused = z_text + self.lora_alpha * delta_text
        
        # Vis fused = Vis + alpha * (Text -> B_text -> A_vis)
        delta_vis = self.vis_lora_A(self.text_lora_B(z_text))
        z_vis_fused = z_vis + self.lora_alpha * delta_vis
        
        # 3. Quantize
        x_q_text, rq_loss_text, indices_text, dist_text = self.text_rq(z_text_fused, use_sk=use_sk)
        x_q_vis, rq_loss_vis, indices_vis, dist_vis = self.vis_rq(z_vis_fused, use_sk=use_sk)
        
        # 4. Decode
        out_text = self.text_decoder(x_q_text)
        out_vis = self.vis_decoder(x_q_vis)
        
        return out_text, out_vis, rq_loss_text, rq_loss_vis, indices_text, indices_vis

    def compute_loss(self, out_text, out_vis, quant_loss_text, quant_loss_vis, x_text=None, x_vis=None):
        # Text Loss
        if self.loss_type == 'mse':
            loss_recon_text = F.mse_loss(out_text, x_text, reduction='mean')
            loss_recon_vis = F.mse_loss(out_vis, x_vis, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon_text = F.l1_loss(out_text, x_text, reduction='mean')
            loss_recon_vis = F.l1_loss(out_vis, x_vis, reduction='mean')
        else:
             raise ValueError('incompatible loss type')
             
        loss_total_text = loss_recon_text + self.quant_loss_weight * quant_loss_text
        loss_total_vis = loss_recon_vis + self.quant_loss_weight * quant_loss_vis
        
        loss_total = loss_total_text + loss_total_vis
        
        return loss_total, loss_recon_text, loss_recon_vis
        
    @torch.no_grad()
    def get_indices(self, x_text, x_vis, use_sk=False):
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        delta_text = self.text_lora_A(self.vis_lora_B(z_vis))
        z_text_fused = z_text + self.lora_alpha * delta_text
        
        delta_vis = self.vis_lora_A(self.text_lora_B(z_text))
        z_vis_fused = z_vis + self.lora_alpha * delta_vis
        
        _, _, indices_text, _ = self.text_rq(z_text_fused, use_sk=use_sk)
        _, _, indices_vis, _ = self.vis_rq(z_vis_fused, use_sk=use_sk)
        
        return indices_text, indices_vis
