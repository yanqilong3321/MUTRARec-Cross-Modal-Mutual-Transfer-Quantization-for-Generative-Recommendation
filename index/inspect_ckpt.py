import torch
import sys

ckpt_path = "log/Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports/llama_256/best_collision_model.pth"

try:
    print(f"Inspecting {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Type: {type(ckpt)}")
    
    if isinstance(ckpt, dict):
        print(f"Keys: {ckpt.keys()}")
        if 'state_dict' in ckpt:
            print("state_dict keys sample:")
            for k in list(ckpt['state_dict'].keys())[:5]:
                print(f"  {k}")
        if 'args' in ckpt:
            print(f"Args: {ckpt['args']}")
    else:
        print("Checkpoint is not a dict!")

except Exception as e:
    print(f"Error: {e}")
