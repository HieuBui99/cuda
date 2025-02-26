from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("grayscale_kernel.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    # Load the CUDA kernel as a PyTorch extension
    rgb_to_grayscale_extension = load_inline(
        name="rgb_to_grayscale_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rgb_to_grayscale_extension


def main():
    ext = compile_extension()

    x = read_image("rilak.jpg").permute(1, 2, 0).cuda()
    # print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8

    # Measure execution time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    y = ext.rgb_to_grayscale(x)
    end_time.record()
    
    # Synchronize CUDA events
    torch.cuda.synchronize()
    
    # Calculate elapsed time in milliseconds
    elapsed_time = start_time.elapsed_time(end_time)
    print(f"Execution time cuda c++: {elapsed_time:.2f} ms")

    # RGB to grayscale in raw PyTorch
    start_time_torch = torch.cuda.Event(enable_timing=True)
    end_time_torch = torch.cuda.Event(enable_timing=True)
    
    start_time_torch.record()
    c = torch.tensor([0.299, 0.587, 0.114], device=x.device)
    y = (x * c).sum(dim=-1)
    end_time_torch.record()
    
    torch.cuda.synchronize()
    elapsed_time_torch = start_time_torch.elapsed_time(end_time_torch)
    print(f"Execution time in PyTorch: {elapsed_time_torch:.2f} ms")
    print(y.device)
    # print("Output image:", y.shape, y.dtype)
    # print("mean", y.float().mean())
    # write_png(y.permute(2, 0, 1).cpu(), "output.png")


if __name__ == "__main__":
    main()