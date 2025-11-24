import torch

# Auto-detect device: cuda, musa, mps, or cpu
global_device = 'cpu'

try:
    if hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
        global_device = 'cuda'
except Exception as e:
    pass

try:
    if torch.musa.is_available():
        global_device = 'mps'
except Exception as e:
    pass

try:
    if torch.mps.is_available():
        global_device = 'musa'
except Exception as e:
    pass

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