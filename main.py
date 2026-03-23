import torch

def verify_rocm_pytorch(output_file=None):
    print(f"PyTorch Version: {torch.__version__}", file=output_file)
    
    # Check if PyTorch detects the AMD GPU
    is_available = torch.cuda.is_available()
    print(f"Is ROCm/CUDA available? {is_available}", file=output_file)
    
    if is_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs detected: {device_count}", file=output_file)
        
        # Retrieve the name of your AMD Radeon Graphics
        device_name = torch.cuda.get_device_name(0)
        print(f"Device Name: {device_name}", file=output_file)
        
        # Perform a simple tensor operation on the GPU
        tensor = torch.randn(3, 3).to('cuda')
        print(f"\nSuccessfully created a tensor on the GPU:\n{tensor}", file=output_file)
    else:
        print("Hardware detection failed. Check your 'render' group permissions.", file=output_file)

if __name__ == "__main__":
    with open('output.txt', 'w') as f:
        verify_rocm_pytorch(output_file=f)