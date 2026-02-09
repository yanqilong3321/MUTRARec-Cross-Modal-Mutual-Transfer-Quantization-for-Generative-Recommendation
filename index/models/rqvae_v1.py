"""
FusionRQVAE V1: 低秩共享码本版本

核心创新: 
- 文本和视觉共享基础码本矩阵 A [num_codes, r]
- 每个模态有自己的投影矩阵 B [r, e_dim]
- 实际码本 = A @ B
- 这种参数化本身就是一种"码本级别的跨模态交互"
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .vq_lowrank import ResidualVectorQuantizerLowRank


class FusionRQVAE_v1(nn.Module):
    """
    FusionRQVAE V1: 低秩共享码本
    
    架构:
    1. Encode: x_text/x_vis -> z_text/z_vis
    2. Quantize (使用低秩码本): 
       - 共享 A [num_codes, r]
       - 文本 B_text, 视觉 B_vis
       - 文本码本 = A @ B_text, 视觉码本 = A @ B_vis
    3. Decode: x_q -> out
    
    参数量对比:
    - 原版: 2 × num_codes × e_dim
    - V1版: num_codes × r + 2 × r × e_dim (如果r << e_dim，大幅减少)
    """
    
    def __init__(self,
                 text_in_dim,
                 vis_in_dim,
                 num_emb_list=None,
                 e_dim=64,
                 lora_rank=8,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 codebook_init_method='random'  # 'random' 或 'kmeans_svd'
        ):
        super(FusionRQVAE_v1, self).__init__()
        
        self.text_in_dim = text_in_dim
        self.vis_in_dim = vis_in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.lora_rank = lora_rank
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
        
        # 文本RQ使用低秩码本（只有B矩阵）
        self.text_rq = ResidualVectorQuantizerLowRank(
            num_emb_list, e_dim, lora_rank=lora_rank,
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
        
        # 视觉RQ使用低秩码本（只有B矩阵）
        self.vis_rq = ResidualVectorQuantizerLowRank(
            num_emb_list, e_dim, lora_rank=lora_rank,
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
        # 每个RQ层一个A矩阵
        self.codebook_lora_a_list = nn.ParameterList([
            nn.Parameter(torch.zeros(n_e, lora_rank))
            for n_e in num_emb_list
        ])
        
        # 初始化A矩阵
        for lora_a in self.codebook_lora_a_list:
            # 方案1: Xavier风格初始化
            std_a = 1.0 / np.sqrt(lora_a.shape[0] * lora_rank)
            nn.init.normal_(lora_a, mean=0, std=std_a)
        
    def forward(self, x_text, x_vis, use_sk=True):
        """
        前向传播
        
        流程:
        1. 编码
        2. 量化（使用低秩码本：共享A + 各自的B）
        3. 解码
        """
        # 1. Encode
        z_text = self.text_encoder(x_text)
        z_vis = self.vis_encoder(x_vis)
        
        # 2. Quantize (使用低秩码本)
        x_q_text, rq_loss_text, indices_text, dist_text = self.text_rq(
            z_text, self.codebook_lora_a_list, use_sk=use_sk
        )
        x_q_vis, rq_loss_vis, indices_vis, dist_vis = self.vis_rq(
            z_vis, self.codebook_lora_a_list, use_sk=use_sk
        )
        
        # 3. Decode
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
        
        _, _, indices_text, _ = self.text_rq(z_text, self.codebook_lora_a_list, use_sk=use_sk)
        _, _, indices_vis, _ = self.vis_rq(z_vis, self.codebook_lora_a_list, use_sk=use_sk)
        
        return indices_text, indices_vis
    
    @torch.no_grad()
    def get_codebooks(self):
        """
        获取所有码本（用于分析）
        
        Returns:
            text_codebooks: [num_layers, num_codes, e_dim]
            vis_codebooks: [num_layers, num_codes, e_dim]
        """
        text_codebooks = self.text_rq.get_codebook(self.codebook_lora_a_list)
        vis_codebooks = self.vis_rq.get_codebook(self.codebook_lora_a_list)
        return text_codebooks, vis_codebooks
    
    def print_codebook_stats(self):
        """打印码本统计信息"""
        print("\n" + "="*60)
        print("V1 低秩码本统计")
        print("="*60)
        
        # 参数量
        params_a = sum(a.numel() for a in self.codebook_lora_a_list)
        params_b_text = sum(p.numel() for p in self.text_rq.parameters())
        params_b_vis = sum(p.numel() for p in self.vis_rq.parameters())
        
        print(f"共享码本A参数量: {params_a:,}")
        print(f"文本码本B参数量: {params_b_text:,}")
        print(f"视觉码本B参数量: {params_b_vis:,}")
        print(f"总码本参数量: {params_a + params_b_text + params_b_vis:,}")
        
        # 对比原版
        original_params = 2 * sum(self.num_emb_list) * self.e_dim
        print(f"\n原版码本参数量: {original_params:,}")
        reduction = (1 - (params_a + params_b_text + params_b_vis) / original_params) * 100
        print(f"参数量减少: {reduction:.1f}%")
        
        # 码本相似度
        with torch.no_grad():
            text_codebooks, vis_codebooks = self.get_codebooks()
            for i in range(len(text_codebooks)):
                text_cb = text_codebooks[i]  # [n_e, e_dim]
                vis_cb = vis_codebooks[i]
                
                # 计算余弦相似度
                text_norm = F.normalize(text_cb, dim=1)
                vis_norm = F.normalize(vis_cb, dim=1)
                similarity = (text_norm * vis_norm).sum(dim=1).mean()
                
                print(f"\nRQ Layer {i}: 文本-视觉码本平均余弦相似度 = {similarity:.4f}")
        
        print("="*60 + "\n")


