"""
FusionRQVAE V3: 量化前LoRA融合 + 低秩共享码本

核心创新: 双重跨模态对齐
1. 特征层面: LoRA在量化前融合（像baseline）
2. 码本层面: 低秩共享码本A矩阵（像V1）

架构: Encode → [LoRA融合] → Quantize(低秩共享码本) → Decode
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .vq_lowrank import ResidualVectorQuantizerLowRank


class FusionRQVAE_v3(nn.Module):
    """
    FusionRQVAE V3: 量化前LoRA融合 + 低秩共享码本
    
    双重跨模态机制:
    1. 特征LoRA: 在编码后进行跨模态融合
    2. 码本共享: 两个模态共享基础码本矩阵A
    
    优势:
    - 强跨模态对齐（特征+码本双重对齐）
    - 参数效率高（低秩码本）
    - 信息保留完整（量化前融合）
    
    风险:
    - 码本学习困难（融合后的动态分布 + 低秩约束）
    - 可能需要更大的rank（建议r>=16）
    """
    
    def __init__(self,
                 text_in_dim,
                 vis_in_dim,
                 num_emb_list=None,
                 e_dim=64,
                 codebook_lora_rank=16,  # 码本的低秩维度（建议>=16）
                 fusion_lora_rank=32,    # 特征融合的LoRA维度
                 fusion_lora_alpha=0.1,  # 特征融合强度（建议<=0.1）
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 codebook_init_method='random'
        ):
        super(FusionRQVAE_v3, self).__init__()
        
        self.text_in_dim = text_in_dim
        self.vis_in_dim = vis_in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.codebook_lora_rank = codebook_lora_rank
        self.fusion_lora_rank = fusion_lora_rank
        self.fusion_lora_alpha = fusion_lora_alpha
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.codebook_init_method = codebook_init_method
        
        self.num_rq_layers = len(num_emb_list)

        # --- Text Pathway ---
        self.text_encode_layer_dims = [self.text_in_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims, 
                                      dropout=self.dropout_prob, bn=self.bn)
        
        # 文本RQ使用低秩码本
        self.text_rq = ResidualVectorQuantizerLowRank(
            num_emb_list, e_dim, lora_rank=codebook_lora_rank,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            init_method=codebook_init_method
        )
                                          
        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims, 
                                      dropout=self.dropout_prob, bn=self.bn)

        # --- Vis Pathway ---
        self.vis_encode_layer_dims = [self.vis_in_dim] + self.layers + [self.e_dim]
        self.vis_encoder = MLPLayers(layers=self.vis_encode_layer_dims, 
                                     dropout=self.dropout_prob, bn=self.bn)
        
        # 视觉RQ使用低秩码本
        self.vis_rq = ResidualVectorQuantizerLowRank(
            num_emb_list, e_dim, lora_rank=codebook_lora_rank,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            init_method=codebook_init_method
        )

        self.vis_decode_layer_dims = self.vis_encode_layer_dims[::-1]
        self.vis_decoder = MLPLayers(layers=self.vis_decode_layer_dims, 
                                     dropout=self.dropout_prob, bn=self.bn)

        # --- 共享的码本矩阵 A (跨模态共享) ---
        self.codebook_lora_a_list = nn.ParameterList([
            nn.Parameter(torch.zeros(n_e, codebook_lora_rank))
            for n_e in num_emb_list
        ])
        
        # 初始化A矩阵
        for lora_a in self.codebook_lora_a_list:
            std_a = 1.0 / np.sqrt(lora_a.shape[0] * codebook_lora_rank)
            nn.init.normal_(lora_a, mean=0, std=std_a)
        
        # --- 特征层面的LoRA融合 (像baseline) ---
        self.text_fusion_lora_A = nn.Linear(fusion_lora_rank, e_dim, bias=False)
        self.text_fusion_lora_B = nn.Linear(e_dim, fusion_lora_rank, bias=False)
        
        self.vis_fusion_lora_A = nn.Linear(fusion_lora_rank, e_dim, bias=False)
        self.vis_fusion_lora_B = nn.Linear(e_dim, fusion_lora_rank, bias=False)
        
        # Init 特征LoRA
        nn.init.zeros_(self.text_fusion_lora_B.weight)
        nn.init.normal_(self.text_fusion_lora_A.weight, std=0.02)
        nn.init.zeros_(self.vis_fusion_lora_B.weight)
        nn.init.normal_(self.vis_fusion_lora_A.weight, std=0.02)
        
    def forward(self, x_text, x_vis, use_sk=True):
        """
        前向传播: 量化前融合 + 低秩码本
        
        流程:
        1. Encode
        2. LoRA Fusion (特征层面融合)
        3. Quantize (使用低秩共享码本)
        4. Decode
        """
        # 1. Encode
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        # 2. Feature-level LoRA Fusion (量化前融合)
        delta_text = self.text_fusion_lora_A(self.vis_fusion_lora_B(z_vis))
        z_text_fused = z_text + self.fusion_lora_alpha * delta_text
        
        delta_vis = self.vis_fusion_lora_A(self.text_fusion_lora_B(z_text))
        z_vis_fused = z_vis + self.fusion_lora_alpha * delta_vis
        
        # 3. Quantize (使用低秩共享码本)
        x_q_text, rq_loss_text, indices_text, dist_text = self.text_rq(
            z_text_fused, self.codebook_lora_a_list, use_sk=use_sk
        )
        x_q_vis, rq_loss_vis, indices_vis, dist_vis = self.vis_rq(
            z_vis_fused, self.codebook_lora_a_list, use_sk=use_sk
        )
        
        # 4. Decode
        out_text = self.text_decoder(x_q_text)
        out_vis = self.vis_decoder(x_q_vis)
        
        return out_text, out_vis, rq_loss_text, rq_loss_vis, indices_text, indices_vis

    def compute_loss(self, out_text, out_vis, quant_loss_text, quant_loss_vis, 
                     x_text=None, x_vis=None):
        """计算总损失"""
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
        """获取量化索引"""
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        # 特征融合
        delta_text = self.text_fusion_lora_A(self.vis_fusion_lora_B(z_vis))
        z_text_fused = z_text + self.fusion_lora_alpha * delta_text
        
        delta_vis = self.vis_fusion_lora_A(self.text_fusion_lora_B(z_text))
        z_vis_fused = z_vis + self.fusion_lora_alpha * delta_vis
        
        # 量化
        _, _, indices_text, _ = self.text_rq(z_text_fused, self.codebook_lora_a_list, use_sk=use_sk)
        _, _, indices_vis, _ = self.vis_rq(z_vis_fused, self.codebook_lora_a_list, use_sk=use_sk)
        
        return indices_text, indices_vis
    
    @torch.no_grad()
    def get_codebooks(self):
        """获取所有码本"""
        text_codebooks = self.text_rq.get_codebook(self.codebook_lora_a_list)
        vis_codebooks = self.vis_rq.get_codebook(self.codebook_lora_a_list)
        return text_codebooks, vis_codebooks
    
    def print_codebook_stats(self):
        """打印码本和LoRA统计"""
        print("\n" + "="*60)
        print("V3 双重跨模态对齐统计")
        print("="*60)
        
        # 码本参数量
        params_a = sum(a.numel() for a in self.codebook_lora_a_list)
        params_b_text = sum(p.numel() for p in self.text_rq.parameters())
        params_b_vis = sum(p.numel() for p in self.vis_rq.parameters())
        
        # LoRA参数量
        params_fusion_lora = (self.text_fusion_lora_A.weight.numel() + 
                             self.text_fusion_lora_B.weight.numel() +
                             self.vis_fusion_lora_A.weight.numel() + 
                             self.vis_fusion_lora_B.weight.numel())
        
        print(f"码本参数:")
        print(f"  共享码本A: {params_a:,}")
        print(f"  文本码本B: {params_b_text:,}")
        print(f"  视觉码本B: {params_b_vis:,}")
        print(f"  码本总计: {params_a + params_b_text + params_b_vis:,}")
        
        print(f"\n特征融合LoRA参数:")
        print(f"  LoRA总计: {params_fusion_lora:,}")
        
        print(f"\n总跨模态参数: {params_a + params_b_text + params_b_vis + params_fusion_lora:,}")
        
        # 对比原版
        original_params = 2 * sum(self.num_emb_list) * self.e_dim
        reduction = (1 - (params_a + params_b_text + params_b_vis) / original_params) * 100
        print(f"\n码本参数减少: {reduction:.1f}%")
        print("="*60 + "\n")

