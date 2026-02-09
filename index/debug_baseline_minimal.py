import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.rqvae import RQVAE

def test_rqvae():
    print("Testing RQVAE minimal instantiation...")
    
    # Mock parameters based on Baseline Config
    # Input=4096, Layers=[2048, 1024, 512, 256, 128, 64], EmbList=[256, 256, 256, 256]
    input_dim = 4096
    num_emb_list = [256, 256, 256, 256]
    e_dim = 32
    layers = [2048, 1024, 512, 256, 128, 64]
    
    try:
        model = RQVAE(
            in_dim=input_dim,
            num_emb_list=num_emb_list,
            e_dim=e_dim,
            layers=layers,
            dropout_prob=0.0,
            bn=False
        )
        print("Model instantiated successfully.")
        
        # Mock input
        dummy_input = torch.randn(2, input_dim)
        print(f"Dummy input shape: {dummy_input.shape}")
        
        # Test get_indices
        print("Calling get_indices...")
        ret = model.get_indices(dummy_input)
        print(f"Return type: {type(ret)}")
        
        if isinstance(ret, tuple):
            print(f"Return length: {len(ret)}")
            print(f"Indices shape: {ret[0].shape}")
            print(f"Distances shape: {ret[1].shape}")
        else:
            print(f"Return value: {ret}")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rqvae()
