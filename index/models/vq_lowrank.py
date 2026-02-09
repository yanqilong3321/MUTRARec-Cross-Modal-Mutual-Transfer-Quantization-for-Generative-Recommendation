"""
低秩码本向量量化器 (Low-rank VectorQuantizer)

核心思想: 码本通过低秩分解参数化
- 原版: embedding [num_codes, e_dim]
- 低秩: A [num_codes, r] @ B [r, e_dim] → [num_codes, e_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizerLowRank(nn.Module):
    """
    低秩码本的向量量化器
    
    Args:
        n_e: 码本大小（码向量数量）
        e_dim: 码本向量维度
        lora_rank: 低秩分解的秩 r
        beta: commitment loss 权重
        kmeans_init: 是否使用K-means初始化
        kmeans_iters: K-means迭代次数
        sk_epsilon: Sinkhorn算法的epsilon
        sk_iters: Sinkhorn算法迭代次数
        init_method: 初始化方法 ('random' 或 'kmeans_svd')
    """
    
    def __init__(self, n_e, e_dim, lora_rank=8,
                 beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.01, sk_iters=100, 
                 init_method='random'):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.lora_rank = lora_rank
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.init_method = init_method
        
        # 低秩参数: 这是模态特定的B矩阵
        # A矩阵在FusionRQVAE中是共享的
        self.codebook_lora_b = nn.Parameter(torch.zeros(lora_rank, e_dim))
        
        # 初始化
        if not kmeans_init or init_method == 'random':
            self.initted = True
            self._init_random()
        else:
            self.initted = False
            # K-means初始化会在第一次前向传播时完成
    
    def _init_random(self):
        """方案1: 对称随机初始化（Xavier风格）"""
        # B矩阵使用Xavier初始化
        std_b = 1.0 / np.sqrt(self.lora_rank * self.e_dim)
        nn.init.normal_(self.codebook_lora_b, mean=0, std=std_b)
    
    def init_emb_from_data(self, data, codebook_lora_a):
        """
        方案3: K-means + SVD初始化
        
        Args:
            data: 输入数据 [B, e_dim]
            codebook_lora_a: 共享的A矩阵 [n_e, r]
        """
        # 1. 对数据做K-means得到聚类中心
        centers = kmeans(data, self.n_e, self.kmeans_iters)  # [n_e, e_dim]
        
        # 2. 给定A，求最佳的B使得 A @ B ≈ centers
        # 使用最小二乘法: B = (A^T A)^{-1} A^T centers
        # 或使用伪逆: B = pinv(A) @ centers
        A = codebook_lora_a.detach()  # [n_e, r]
        
        # 方法1: 伪逆（更稳定）
        A_pinv = torch.linalg.pinv(A)  # [r, n_e]
        B_optimal = A_pinv @ centers  # [r, e_dim]
        
        # 方法2: 最小二乘（备选）
        # B_optimal, _ = torch.lstsq(centers.t(), A.t())
        # B_optimal = B_optimal[:self.lora_rank].t()
        
        self.codebook_lora_b.data.copy_(B_optimal)
        self.initted = True
        
        # 验证拟合质量
        reconstructed = A @ B_optimal
        reconstruction_error = F.mse_loss(reconstructed, centers)
        print(f"  K-means+SVD init: reconstruction error = {reconstruction_error.item():.6f}")
    
    def get_codebook(self, codebook_lora_a):
        """
        获取完整码本
        
        Args:
            codebook_lora_a: 共享的A矩阵 [n_e, r]
        
        Returns:
            codebook: [n_e, e_dim]
        """
        return codebook_lora_a @ self.codebook_lora_b
    
    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def forward(self, x, codebook_lora_a, use_sk=True):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, e_dim]
            codebook_lora_a: 共享的A矩阵 [n_e, r]
            use_sk: 是否使用Sinkhorn算法
        
        Returns:
            x_q: 量化后的特征
            loss: 量化损失
            indices: 码本索引
            distances: 到各个码的距离
        """
        # Flatten input
        latent = x.view(-1, self.e_dim)
        
        # K-means初始化（仅第一次）
        if not self.initted and self.training:
            if self.init_method == 'kmeans_svd':
                print("Initializing codebook with K-means+SVD...")
                self.init_emb_from_data(latent, codebook_lora_a)
            else:
                self._init_random()
                self.initted = True
        
        # 生成完整码本
        embeddings_weight = self.get_codebook(codebook_lora_a)  # [n_e, e_dim]
        
        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, embeddings_weight.t())
        
        # 选择最近的码
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        
        # 查表获取量化向量
        x_q = F.embedding(indices, embeddings_weight).view(x.shape)
        
        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        
        # preserve gradients (straight-through estimator)
        x_q = x + (x_q - x).detach()
        
        indices = indices.view(x.shape[:-1])
        
        return x_q, loss, indices, d


class ResidualVectorQuantizerLowRank(nn.Module):
    """
    低秩码本的残差向量量化器
    
    每个RQ层共享同一个codebook_lora_a（跨模态共享）
    每个RQ层有独立的codebook_lora_b（模态特定）
    """
    
    def __init__(self, n_e_list, e_dim, lora_rank=8,
                 kmeans_init=False, kmeans_iters=100, 
                 sk_epsilons=None, sk_iters=100,
                 init_method='random'):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.lora_rank = lora_rank
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons if sk_epsilons else [0.0] * self.num_quantizers
        self.sk_iters = sk_iters
        self.init_method = init_method
        
        # 注意: 这里只创建B矩阵，A矩阵在外部（FusionRQVAE）创建并共享
        self.vq_layers = nn.ModuleList([
            VectorQuantizerLowRank(
                n_e, e_dim, lora_rank=lora_rank,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
                init_method=init_method
            )
            for n_e, sk_epsilon in zip(n_e_list, self.sk_epsilons)
        ])
    
    def get_codebook(self, codebook_lora_a_list):
        """
        获取所有层的码本
        
        Args:
            codebook_lora_a_list: 每层的A矩阵列表
        
        Returns:
            all_codebook: [num_layers, n_e, e_dim]
        """
        all_codebook = []
        for quantizer, lora_a in zip(self.vq_layers, codebook_lora_a_list):
            codebook = quantizer.get_codebook(lora_a)
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def forward(self, x, codebook_lora_a_list, use_sk=True):
        """
        前向传播
        
        Args:
            x: 输入特征
            codebook_lora_a_list: 每层的A矩阵列表 [层数个 [n_e, r]]
            use_sk: 是否使用Sinkhorn
        """
        all_losses = []
        all_indices = []
        all_distances = []
        
        x_q = 0
        residual = x
        for quantizer, lora_a in zip(self.vq_layers, codebook_lora_a_list):
            x_res, loss, indices, distance = quantizer(residual, lora_a, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            
            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)
        
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)
        
        return x_q, mean_losses, all_indices, all_distances


