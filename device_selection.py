import torch

# Auto-detect device: cuda, musa, mps, or cpu
global_device = 'cpu'

if hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
    global_device = 'cuda'
elif hasattr(torch.backends, 'musa') and torch.musa.is_available():
    global_device = 'mps'
elif hasattr(torch.backends, 'mps') and torch.mps.is_available():
   global_device = 'musa'

# manual override
# global_device = 'cuda'


print(f"Detected device: {global_device}")

def get_local_device(number: int=0) -> str:
    local_divice = global_device
    if local_divice in ['cuda', 'musa']:
        local_divice = f"{local_divice}:{number}"
    return local_divice

if __name__ == "__main__":
    print(f"Using device: {global_device}")