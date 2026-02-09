import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class FusionRQVAE_v2(nn.Module):
    """
    FusionRQVAE Version 2: LoRA融合发生在量化之后
    
    架构流程:
    1. Encode: x_text/x_vis -> z_text/z_vis
    2. Quantize: z_text/z_vis -> x_q_text/x_q_vis (先量化)
    3. LoRA Fusion: x_q_text/x_q_vis -> x_q_text_fused/x_q_vis_fused (后融合)
    4. Decode: x_q_fused -> out_text/out_vis
    
    关键改进:
    - 码本学习纯净的、模态独立的特征分布
    - LoRA在离散的码本向量空间中进行跨模态交互
    - 分工明确: 码本负责各自模态的语义表示, LoRA负责跨模态对齐
    """
    
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
        super(FusionRQVAE_v2, self).__init__()
        
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
                                          kmeans_init=kmeans_init,
                                          kmeans_iters=kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters,
                                          use_linear=use_linear)
                                          
        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

        # --- Vis Pathway ---
        self.vis_encode_layer_dims = [self.vis_in_dim] + self.layers + [self.e_dim]
        self.vis_encoder = MLPLayers(layers=self.vis_encode_layer_dims, dropout=self.dropout_prob, bn=self.bn)
        
        self.vis_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init=kmeans_init,
                                          kmeans_iters=kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters,
                                          use_linear=use_linear)

        self.vis_decode_layer_dims = self.vis_encode_layer_dims[::-1]
        self.vis_decoder = MLPLayers(layers=self.vis_decode_layer_dims, dropout=self.dropout_prob, bn=self.bn)

        # --- LoRA Interaction (在量化后应用) ---
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
        """
        前向传播: 量化后融合
        
        流程:
        1. 编码: 各自独立编码
        2. 量化: 各自独立量化 (码本只看到纯净的模态特征)
        3. LoRA融合: 在量化后的码本向量上进行跨模态交互
        4. 解码: 从融合后的特征重建
        """
        # 1. Encode (独立编码)
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        # 2. Quantize (先量化 - 关键改动!)
        # 码本在这里学习纯净的、模态独立的特征分布
        x_q_text, rq_loss_text, indices_text, dist_text = self.text_rq(z_text, use_sk=use_sk)
        x_q_vis, rq_loss_vis, indices_vis, dist_vis = self.vis_rq(z_vis, use_sk=use_sk)
        
        # 3. LoRA Interaction (在量化后的码本向量上融合)
        # Text fused = x_q_text + alpha * (x_q_vis -> B_vis -> A_text)
        delta_text = self.text_lora_A(self.vis_lora_B(x_q_vis))
        x_q_text_fused = x_q_text + self.lora_alpha * delta_text
        
        # Vis fused = x_q_vis + alpha * (x_q_text -> B_text -> A_vis)
        delta_vis = self.vis_lora_A(self.text_lora_B(x_q_text))
        x_q_vis_fused = x_q_vis + self.lora_alpha * delta_vis
        
        # 4. Decode (从融合后的特征解码)
        out_text = self.text_decoder(x_q_text_fused)
        out_vis = self.vis_decoder(x_q_vis_fused)
        
        return out_text, out_vis, rq_loss_text, rq_loss_vis, indices_text, indices_vis

    def compute_loss(self, out_text, out_vis, quant_loss_text, quant_loss_vis, x_text=None, x_vis=None):
        """计算总损失"""
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
        """
        获取量化索引
        
        注意: V2版本中，indices是在融合之前生成的
        这意味着可以单独使用每个模态的码进行检索
        """
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        # V2版本: 直接从纯净特征获取索引
        _, _, indices_text, _ = self.text_rq(z_text, use_sk=use_sk)
        _, _, indices_vis, _ = self.vis_rq(z_vis, use_sk=use_sk)
        
        return indices_text, indices_vis
    
    @torch.no_grad()
    def get_fused_embeddings(self, x_text, x_vis, use_sk=False):
        """
        获取融合后的嵌入 (用于分析)
        """
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        x_q_text, _, _, _ = self.text_rq(z_text, use_sk=use_sk)
        x_q_vis, _, _, _ = self.vis_rq(z_vis, use_sk=use_sk)
        
        # 融合
        delta_text = self.text_lora_A(self.vis_lora_B(x_q_vis))
        x_q_text_fused = x_q_text + self.lora_alpha * delta_text
        
        delta_vis = self.vis_lora_A(self.text_lora_B(x_q_text))
        x_q_vis_fused = x_q_vis + self.lora_alpha * delta_vis
        
        return x_q_text_fused, x_q_vis_fused, x_q_text, x_q_vis

