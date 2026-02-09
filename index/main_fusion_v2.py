import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os
from collections import defaultdict
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import PairedEmbDatasetAll
from models.rqvae_v2 import FusionRQVAE_v2
from utils import ensure_dir, set_color

class FusionTrainer_v2(object):
    """
    FusionRQVAE V2版本的训练器
    
    与原版的唯一区别: 使用 FusionRQVAE_v2 模型
    训练流程、损失计算、评估方式完全一致
    """
    
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
        self.best_collision_rate = np.inf  # Avg of text and vis
        self.best_collision_ckpt = "best_collision_model_v2.pth"
        
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        params = self.model.parameters()
        if self.learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:  # Fallback to Adam
            optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon_loss_text = 0
        total_recon_loss_vis = 0
        
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, 
                        desc=set_color(f"Train {epoch_idx} [V2]", "pink"))
        
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
                        desc=set_color(f"Evaluate [V2]", "pink"))
        
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
            usage_rate = len(indices_set) / num_sample  # 码本使用率
            return collision_rate, len(indices_set), usage_rate
            
        col_rate_text, unique_codes_text, usage_text = get_collision(indices_list_text)
        col_rate_vis, unique_codes_vis, usage_vis = get_collision(indices_list_vis)
        
        avg_col_rate = (col_rate_text + col_rate_vis) / 2
        
        return avg_col_rate, col_rate_text, col_rate_vis, unique_codes_text, unique_codes_vis

    def _save_checkpoint(self, epoch, collision_rate, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file else \
            os.path.join(self.ckpt_dir, f'epoch_{epoch}_col_{collision_rate:.4f}_model_v2.pth')
            
        state = {
            "args": self.args,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_collision_rate": self.best_collision_rate,
        }
        torch.save(state, ckpt_path)
        self.logger.info(set_color("Saved V2 checkpoint", "blue") + f": {ckpt_path}")

    def fit(self, data_loader):
        self.logger.info(set_color("="*60, "yellow"))
        self.logger.info(set_color("Training FusionRQVAE V2 (Post-Quantization Fusion)", "yellow"))
        self.logger.info(set_color("="*60, "yellow"))
        
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
                           f"  Collision Rate: Avg={col_rate:.4f}, Text={col_t:.4f}, Vis={col_v:.4f}\n"
                           f"  Unique Codes: Text={unique_t}, Vis={unique_v}\n"
                           f"  Best Collision Rate: {self.best_collision_rate:.4f}")
                self.logger.info(set_color(eval_info, "cyan"))
                
                if col_rate < self.best_collision_rate:
                    self.best_collision_rate = col_rate
                    self._save_checkpoint(epoch_idx, col_rate, self.best_collision_ckpt)
                    
        return self.best_loss, self.best_collision_rate

def parse_args():
    parser = argparse.ArgumentParser(description="Fusion Index V2 - Post-Quantization Fusion")
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--eval_step', type=int, default=50, help='evaluation frequency')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer type')
    
    # 数据参数
    parser.add_argument("--data_root", type=str, default="", help="data root directory")
    parser.add_argument('--datasets', type=str, default='Scientific', help='dataset name')
    parser.add_argument('--text_embedding_file', type=str, default=".emb-llama-td.npy", 
                       help='text embedding file suffix')
    parser.add_argument('--vis_embedding_file', type=str, default=".emb-ViT-L-14.npy",
                       help='visual embedding file suffix')
    
    # 模型参数
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout probability")
    parser.add_argument("--bn", type=bool, default=False, help="use batch normalization")
    parser.add_argument("--loss_type", type=str, default="mse", help="reconstruction loss type")
    
    # 量化参数
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans initialization")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="kmeans iterations")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0],
                       help="sinkhorn epsilon values")
    parser.add_argument("--sk_iters", type=int, default=50, help="sinkhorn iterations")
    
    # RQVAE参数
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256],
                       help='codebook sizes for each RQ layer')
    parser.add_argument('--e_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='quantization loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64],
                       help='hidden layer dimensions')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank (bottleneck dimension)')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA fusion strength')
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--ckpt_dir", type=str, default="./log/fusion_v2", help="checkpoint directory")

    return parser.parse_args()

if __name__ == '__main__':
    # 设置随机种子
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.ckpt_dir, 'train_v2.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    logger.info("="*60)
    logger.info("FusionRQVAE V2: Post-Quantization Fusion")
    logger.info("="*60)
    logger.info(f"Arguments: {args}")
    
    # 加载数据
    logger.info("Loading data...")
    data = PairedEmbDatasetAll(args)
    
    # 初始化模型
    logger.info(f"Initializing FusionRQVAE_v2 with Text Dim={data.text_dim}, Vis Dim={data.vis_dim}")
    model = FusionRQVAE_v2(
        text_in_dim=data.text_dim,
        vis_in_dim=data.vis_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 数据加载器
    data_loader = DataLoader(
        data, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    
    # 训练
    trainer = FusionTrainer_v2(args, model)
    best_loss, best_collision = trainer.fit(data_loader)
    
    logger.info("="*60)
    logger.info(f"Training Complete!")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info(f"Best Collision Rate: {best_collision:.4f}")
    logger.info("="*60)

