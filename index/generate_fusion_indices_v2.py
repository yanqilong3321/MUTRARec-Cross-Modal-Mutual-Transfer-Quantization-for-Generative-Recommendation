import collections
import json
import logging
import numpy as np
import torch
import copy
from tqdm import tqdm
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
import os

from datasets import PairedEmbDatasetAll
from models.rqvae_v2 import FusionRQVAE_v2

def parse_args():
    parser = argparse.ArgumentParser(description="Fusion Index Generation V2")
    
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_root', type=str, default="/data/yql/workspace/MQL4GRec/data/")
    parser.add_argument('--device', type=str, default="cuda:0")
    
    # Optional overrides if not loading from ckpt args completely
    parser.add_argument('--text_embedding_file', type=str, default=".emb-llama-td.npy")
    parser.add_argument('--vis_embedding_file', type=str, default=".emb-ViT-L-14.npy")

    return parser.parse_args()

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups

def resolve_collisions(all_indices, all_distances, num_emb_list):
    # 复用原有的冲突解决逻辑
    # all_indices: list of list of ints
    # all_distances: (N, levels, codebook_size)
    
    all_indices_str = [str(x) for x in all_indices]
    all_indices_str_set = set(all_indices_str)
    
    sort_distances_index = np.argsort(all_distances, axis=2)
    
    # Pre-calculate min distances for collision resolution
    item_min_dis = defaultdict(list)
    for item, distances in tqdm(enumerate(all_distances), desc='cal distances', total=len(all_distances)):
        for dis in distances:
            item_min_dis[item].append(np.min(dis))
            
    tt = 0
    level = len(num_emb_list) - 1
    max_num = num_emb_list[0] # assuming all levels have same size or using first
    
    while True:
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str))
        print(f'tot_item: {tot_item}, tot_indice: {tot_indice}')
        print("Collision Rate",(tot_item-tot_indice)/tot_item)
        
        if check_collision(all_indices_str) or tt == 2:
            print('tt', tt)
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"Collision Groups: {len(collision_item_groups)}")
        
        for collision_items in collision_item_groups:
            min_distances = []
            for i, item in enumerate(collision_items):
                min_distances.append(item_min_dis[item][level])

            min_index = np.argsort(np.array(min_distances))
            
            for i, m_index in enumerate(min_index):
                if i == 0: continue # Keep the best one
                
                item = collision_items[m_index]
                ori_code = copy.deepcopy(all_indices[item])
                
                # Try to change the last code
                num = i
                while str(ori_code) in all_indices_str_set and num < max_num:
                    ori_code[level] = sort_distances_index[item][level][num]
                    num += 1
                
                # If still collision, try second to last
                for k in range(1, max_num):
                    if str(ori_code) in all_indices_str_set:
                        ori_code = copy.deepcopy(all_indices[item])
                        ori_code[level-1] = sort_distances_index[item][level-1][k]
                    
                    num = 0
                    while str(ori_code) in all_indices_str_set and num < max_num:
                        ori_code[level] = sort_distances_index[item][level][num]
                        num += 1
                    
                    if str(ori_code) not in all_indices_str_set:
                        break
                
                all_indices[item] = ori_code
                all_indices_str[item] = str(ori_code)
                all_indices_str_set.add(str(ori_code))
        
        tt += 1
        
    return all_indices

def save_json(all_indices, output_file, prefix_fmt):
    all_indices_dict = {}
    for item, indices in enumerate(all_indices):
        code = []
        for i, ind in enumerate(indices):
            code.append(prefix_fmt[i].format(int(ind)))
        all_indices_dict[item] = code
        
    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp, indent=4)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    cmd_args = parse_args()
    
    device = torch.device(cmd_args.device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {cmd_args.ckpt_path}")
    ckpt = torch.load(cmd_args.ckpt_path, map_location='cpu')
    train_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    
    # Override dataset for loading specific one
    train_args.datasets = cmd_args.dataset 
    train_args.data_root = cmd_args.data_root
    
    # Load Data (Paired)
    print(f"Loading data for {cmd_args.dataset}...")
    dataset = PairedEmbDatasetAll(train_args) 
    
    # Reconstruct Model (V2)
    print("Reconstructing FusionRQVAE_v2...")
    model = FusionRQVAE_v2(
        text_in_dim=dataset.text_dim,
        vis_in_dim=dataset.vis_dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_epsilons=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
        lora_rank=train_args.lora_rank,
        lora_alpha=train_args.lora_alpha
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    data_loader = DataLoader(dataset, num_workers=4, batch_size=64, shuffle=False)
    
    # Collect Indices and Distances
    print("Generating indices (V2: quantization before fusion)...")
    text_indices = []
    vis_indices = []
    text_distances = []
    vis_distances = []
    
    for x_text, x_vis in tqdm(data_loader):
        x_text = x_text.to(device)
        x_vis = x_vis.to(device)
        
        # V2版本: 量化在融合之前
        # 我们需要手动调用来获取distances
        with torch.no_grad():
            z_text = model.text_encoder(x_text)
            z_vis = model.vis_encoder(x_vis)
            
            # V2: 先量化，再融合
            _, _, idx_text, dist_text = model.text_rq(z_text, use_sk=False)
            _, _, idx_vis, dist_vis = model.vis_rq(z_vis, use_sk=False)
            
            idx_text = idx_text.view(-1, idx_text.shape[-1]).cpu().numpy()
            idx_vis = idx_vis.view(-1, idx_vis.shape[-1]).cpu().numpy()
            dist_text = dist_text.cpu().numpy() # (B, levels, K)
            dist_vis = dist_vis.cpu().numpy()
            
            for i in idx_text: text_indices.append(i.tolist())
            for i in idx_vis: vis_indices.append(i.tolist())
            
            text_distances.append(dist_text)
            vis_distances.append(dist_vis)

    text_distances = np.concatenate(text_distances, axis=0)
    vis_distances = np.concatenate(vis_distances, axis=0)
    
    print(f"Text Distances Shape: {text_distances.shape}")
    print(f"Vis Distances Shape: {vis_distances.shape}")
    
    # Resolve Collisions and Save Text (V2版本带后缀)
    print("\nProcessing Text Indices...")
    text_indices = resolve_collisions(text_indices, text_distances, train_args.num_emb_list)
    prefix_text = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]
    save_json(text_indices, os.path.join(cmd_args.output_dir, f"{cmd_args.dataset}.index_lemb_v2.json"), prefix_text)
    
    # Resolve Collisions and Save Vis (V2版本带后缀)
    print("\nProcessing Vis Indices...")
    vis_indices = resolve_collisions(vis_indices, vis_distances, train_args.num_emb_list)
    prefix_vis = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>","<E_{}>"]
    save_json(vis_indices, os.path.join(cmd_args.output_dir, f"{cmd_args.dataset}.index_vitemb_v2.json"), prefix_vis)
    
    print("\nDone!")

