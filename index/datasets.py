import numpy as np
import torch
import torch.utils.data as data
import os

class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetAll(data.Dataset):

    def __init__(self, args):

        self.datasets = args.datasets.split(',')
        embeddings = []
        self.dataset_count = []
        for dataset in self.datasets:
            print(dataset)
            embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            self.dataset_count.append(embedding.shape[0])
            
        self.embeddings = np.concatenate(embeddings)
        self.dim = self.embeddings.shape[-1]
        
        print(self.dataset_count)
        print(self.embeddings.shape[0])

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetOne(data.Dataset):

    def __init__(self, args, dataset):


        print(dataset)
        embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
        self.embedding = np.load(embedding_path)

        self.dim = self.embedding.shape[-1]
        
        self.data_count = self.embedding.shape[0]

        print(self.embedding.shape)

    def __getitem__(self, index):
        emb = self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embedding)

class PairedEmbDatasetAll(data.Dataset):
    def __init__(self, args):
        self.datasets = args.datasets.split(',')
        text_embeddings = []
        vis_embeddings = []
        self.dataset_count = []
        
        # Default file suffixes if not provided in args
        text_suffix = getattr(args, 'text_embedding_file', '.emb-llama-td.npy')
        vis_suffix = getattr(args, 'vis_embedding_file', '.emb-ViT-L-14.npy')

        for dataset in self.datasets:
            print(f"Loading {dataset}...")
            # Load Text
            text_path = os.path.join(args.data_root, dataset, f'{dataset}{text_suffix}')
            t_emb = np.load(text_path)
            
            # Load Vis
            vis_path = os.path.join(args.data_root, dataset, f'{dataset}{vis_suffix}')
            v_emb = np.load(vis_path)
            
            if t_emb.shape[0] != v_emb.shape[0]:
                print(f"Warning: Mismatch in {dataset}: Text {t_emb.shape}, Vis {v_emb.shape}. Truncating to min.")
                min_len = min(t_emb.shape[0], v_emb.shape[0])
                t_emb = t_emb[:min_len]
                v_emb = v_emb[:min_len]
            
            text_embeddings.append(t_emb)
            vis_embeddings.append(v_emb)
            self.dataset_count.append(t_emb.shape[0])
            
        self.text_embeddings = np.concatenate(text_embeddings)
        self.vis_embeddings = np.concatenate(vis_embeddings)
        
        self.text_dim = self.text_embeddings.shape[-1]
        self.vis_dim = self.vis_embeddings.shape[-1]
        
        print(f"Total Data: {self.text_embeddings.shape[0]}")
        print(f"Text Dim: {self.text_dim}, Vis Dim: {self.vis_dim}")

    def __getitem__(self, index):
        t_emb = torch.FloatTensor(self.text_embeddings[index])
        v_emb = torch.FloatTensor(self.vis_embeddings[index])
        return t_emb, v_emb

    def __len__(self):
        return len(self.text_embeddings)
