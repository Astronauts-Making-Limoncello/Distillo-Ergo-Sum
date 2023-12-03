from rich import print

import torch

### --- device --- ###

def print_device_name(device: torch.cuda.device):
    
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        
        if "NVIDIA" in device_name:
            device_name = f"[#008000][b]{device_name}[/b][/#008000]"
        
        print(f"Device: {device_name}\n")

### --- device --- ###

################################################################################

### --- model --- ###

def print_num_trainable_parameters(model: torch.nn.Module, num_color: str = "#000000"):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Number of trainable parameters: [{num_color}][b]{n}[/b][/{num_color}]\n")

### --- model --- ###

################################################################################