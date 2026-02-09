import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os
import sys
from collections import defaultdict
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import PairedEmbDatasetAll
from models.rqvae_v3 import FusionRQVAE_v3
from utils import ensure_dir, set_color

class FusionTrainer_v3(object):
    """FusionRQVAE V3 训练器（量化前融合 + 低秩共享码本）"""
    
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)
        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_collision_ckpt = "best_collision_model_v3.pth"
        
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        params = self.model.parameters()
        if self.learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon_loss_text = 0
        total_recon_loss_vis = 0
        
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, 
                        desc=set_color(f"Train {epoch_idx} [V3]", "pink"),
                        disable=not sys.stdout.isatty())
        
        for batch_idx, (x_text, x_vis) in enumerate(iter_data):
            x_text = x_text.to(self.device)
            x_vis = x_vis.to(self.device)
            
            self.optimizer.zero_grad()
            
            out_text, out_vis, rq_loss_text, rq_loss_vis, _, _ = self.model(x_text, x_vis)
            
            loss, recon_text, recon_vis = self.model.compute_loss(
                out_text, out_vis, rq_loss_text, rq_loss_vis, x_text, x_vis
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss_text += recon_text.item()
            total_recon_loss_vis += recon_vis.item()
            
        return total_loss, total_recon_loss_text, total_recon_loss_vis

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()
        indices_list_text = []
        indices_list_vis = []
        num_sample = 0
        
        iter_data = tqdm(valid_data, total=len(valid_data), ncols=100,
                        desc=set_color(f"Evaluate [V3]", "pink"),
                        disable=not sys.stdout.isatty())
        
        for batch_idx, (x_text, x_vis) in enumerate(iter_data):
            num_sample += len(x_text)
            x_text = x_text.to(self.device)
            x_vis = x_vis.to(self.device)
            
            idx_text, idx_vis = self.model.get_indices(x_text, x_vis)
            
            idx_text = idx_text.view(-1, idx_text.shape[-1]).cpu().numpy()
            idx_vis = idx_vis.view(-1, idx_vis.shape[-1]).cpu().numpy()
            
            for index in idx_text:
                indices_list_text.append("-".join([str(int(_)) for _ in index]))
            for index in idx_vis:
                indices_list_vis.append("-".join([str(int(_)) for _ in index]))
                
        # Calculate Collision Rate
        def get_collision(indices_list):
            indices_set = set(indices_list)
            collision_rate = (num_sample - len(indices_set)) / num_sample
            return collision_rate, len(indices_set)
            
        col_rate_text, unique_codes_text = get_collision(indices_list_text)
        col_rate_vis, unique_codes_vis = get_collision(indices_list_vis)
        
        avg_col_rate = (col_rate_text + col_rate_vis) / 2
        
        return avg_col_rate, col_rate_text, col_rate_vis, unique_codes_text, unique_codes_vis

    def _save_checkpoint(self, epoch, collision_rate, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file else \
            os.path.join(self.ckpt_dir, f'epoch_{epoch}_col_{collision_rate:.4f}_model_v3.pth')
            
        state = {
            "args": self.args,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_collision_rate": self.best_collision_rate,
        }
        torch.save(state, ckpt_path)
        self.logger.info(set_color("Saved V3 checkpoint", "blue") + f": {ckpt_path}")

    def fit(self, data_loader):
        self.logger.info(set_color("="*60, "yellow"))
        self.logger.info(set_color("Training FusionRQVAE V3 (Pre-Quant Fusion + Low-Rank Codebook)", "yellow"))
        self.logger.info(set_color(f"Codebook Init: {self.args.codebook_init_method}", "yellow"))
        self.logger.info(set_color(f"Codebook Rank: {self.args.codebook_lora_rank}", "yellow"))
        self.logger.info(set_color(f"Fusion Alpha: {self.args.fusion_lora_alpha}", "yellow"))
        self.logger.info(set_color("="*60, "yellow"))
        
        # 打印统计
        self.model.print_codebook_stats()
        
        for epoch_idx in range(self.epochs):
            # Train
            t_start = time()
            loss, recon_t, recon_v = self._train_epoch(data_loader, epoch_idx)
            t_end = time()
            
            train_info = (f"Epoch {epoch_idx}: Time {t_end-t_start:.2f}s, "
                         f"Loss {loss:.4f}, ReconT {recon_t:.4f}, ReconV {recon_v:.4f}")
            self.logger.info(set_color(train_info, "green"))
            
            if loss < self.best_loss:
                self.best_loss = loss
            
            # Eval
            if (epoch_idx + 1) % self.eval_step == 0:
                eval_start = time()
                col_rate, col_t, col_v, unique_t, unique_v = self._valid_epoch(data_loader)
                eval_end = time()
                
                eval_info = (f"Eval (Epoch {epoch_idx}): Time {eval_end-eval_start:.2f}s\n"
                           f"  Collision: Avg={col_rate:.4f}, Text={col_t:.4f}, Vis={col_v:.4f}\n"
                           f"  Unique: Text={unique_t}, Vis={unique_v}\n"
                           f"  Best Collision: {self.best_collision_rate:.4f}")
                self.logger.info(set_color(eval_info, "cyan"))
                
                if col_rate < self.best_collision_rate:
                    self.best_collision_rate = col_rate
                    self._save_checkpoint(epoch_idx, col_rate, self.best_collision_ckpt)
                    
        return self.best_loss, self.best_collision_rate

def parse_args():
    parser = argparse.ArgumentParser(description="Fusion Index V3")
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--learner', type=str, default="AdamW")
    
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument('--datasets', type=str, default='Scientific')
    parser.add_argument('--text_embedding_file', type=str, default=".emb-llama-td.npy")
    parser.add_argument('--vis_embedding_file', type=str, default=".emb-ViT-L-14.npy")
    
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mse")
    
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument("--sk_iters", type=int, default=50)
    
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64])
    
    # V3特有参数
    parser.add_argument('--codebook_lora_rank', type=int, default=16,
                       help='Low-rank for codebook (建议>=16)')
    parser.add_argument('--fusion_lora_rank', type=int, default=32,
                       help='LoRA rank for feature fusion')
    parser.add_argument('--fusion_lora_alpha', type=float, default=0.1,
                       help='Feature fusion strength (建议<=0.1)')
    parser.add_argument('--codebook_init_method', type=str, default='random',
                       choices=['random', 'kmeans_svd'])
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_dir", type=str, default="./log/fusion_v3")
    parser.add_argument("--log_file", type=str, default="train_v3.log",
                       help='Log file name (relative to ckpt_dir)')

    return parser.parse_args()

if __name__ == '__main__':
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.ckpt_dir, args.log_file)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    logger.info("="*60)
    logger.info("FusionRQVAE V3: Pre-Quant Fusion + Low-Rank Codebook")
    logger.info("="*60)
    
    # 加载数据
    logger.info("Loading data...")
    data = PairedEmbDatasetAll(args)
    
    logger.info(f"Initializing FusionRQVAE_v3: Text={data.text_dim}, Vis={data.vis_dim}")
    logger.info(f"Codebook Rank={args.codebook_lora_rank}, Fusion Alpha={args.fusion_lora_alpha}")
    
    model = FusionRQVAE_v3(
        text_in_dim=data.text_dim,
        vis_in_dim=data.vis_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        codebook_lora_rank=args.codebook_lora_rank,
        fusion_lora_rank=args.fusion_lora_rank,
        fusion_lora_alpha=args.fusion_lora_alpha,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        codebook_init_method=args.codebook_init_method
    )
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    data_loader = DataLoader(
        data, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    
    trainer = FusionTrainer_v3(args, model)
    best_loss, best_collision = trainer.fit(data_loader)
    
    logger.info("="*60)
    logger.info(f"Training Complete!")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info(f"Best Collision Rate: {best_collision:.4f}")
    logger.info("="*60)

