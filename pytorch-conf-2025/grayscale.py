
import torch

import triton
import triton.language as tl

## MAX Imports

import max.torch
import numpy as np
from max.torch import CustomOpLibrary
from max.dtype import DType
from max.graph import ops, DeviceRef
from pathlib import Path

# Load the Mojo custom operations from the `operations` directory.
mojo_kernels = Path(__file__).parent / "custom_ops"
custom_ops = CustomOpLibrary(mojo_kernels)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# torch.compile does not work on HIP devices?
# @torch.compile
def torch_grayscale(img: torch.Tensor) -> torch.Tensor:
    rgb_mask = torch.as_tensor(
        [0.21, 0.71, 0.07], dtype=torch.float32, device=img.device
    )

    C, H, W = img.shape

    # Expand relevant dims
    rgb_mask = rgb_mask[:, None, None].expand(C, H, W)

    img = img.to(torch.float32) * torch.broadcast_to(rgb_mask, img.shape)

    result = img.sum(dim=0, dtype=torch.float32)
    return result

@triton.jit
def grayscale_kernel(x_ptr, out_ptr, h, w, bs0: tl.constexpr, bs1: tl.constexpr):
    """
    GPU kernel for converting RGB image to grayscale

    Args:
        x_ptr: Pointer to input RGB image data
        out_ptr: Pointer to output grayscale image data
        h: Image height
        w: Image width
        bs0: Block size for height dimension
        bs1: Block size for width dimension
    """
    # Get program IDs for parallel processing
    pid_0 = tl.program_id(0)  # Block ID in height dimension
    pid_1 = tl.program_id(1)  # Block ID in width dimension

    # Calculate offsets for this block
    offs_0 = pid_0 * bs0 + tl.arange(0, bs0)  # Offsets in height dimension
    offs_1 = pid_1 * bs1 + tl.arange(0, bs1)  # Offsets in width dimension

    # Calculate 2D offset matrix
    offs = w * offs_0[:,None] + offs_1[None, :]

    # Create masks to handle image boundaries
    mask_0 = offs_0 < h
    mask_1 = offs_1 < w
    mask = mask_0[:,None] & mask_1[None,:]

    # Load RGB channels
    r = tl.load(x_ptr + 0*h*w + offs, mask=mask)
    g = tl.load(x_ptr + 1*h*w + offs, mask=mask)
    b = tl.load(x_ptr + 2*h*w + offs, mask=mask)

    out = 0.21*r + 0.71*g + 0.07*b

    # Store the result
    tl.store(out_ptr + offs, out, mask=mask)


def triton_grayscale(x, bs):
    """
    Convert RGB image to grayscale using GPU acceleration

    Args:
        x: Input RGB image tensor (channels, height, width)
        bs: Tuple of block sizes (height, width) for GPU processing

    Returns:
        Grayscale image tensor (height, width)
    """
    c, h, w = x.shape
    # Create output tensor
    out = torch.empty((h, w), dtype=x.dtype, device=x.device)

    # Define processing grid based on block sizes
    grid = lambda meta: (triton.cdiv(h, meta['bs0']), triton.cdiv(w, meta['bs1']))

    # Launch GPU kernel
    grayscale_kernel[grid](x, out, h, w, bs0=bs[0], bs1=bs[1])
    return out.view(h, w)


max_grayscale_kernel = custom_ops.grayscale


def max_grayscale(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty(x.shape[1:], dtype=x.dtype, device=x.device)
    max_grayscale_kernel(result, x)
    return result


@max.torch.graph_op
def max_grayscale_graph(pic: max.graph.TensorValue):
    c, h, w = pic.shape

    scale = ops.constant(np.array([0.21, 0.71, 0.07]), dtype=DType.float32, device=DeviceRef.GPU())
    scale = ops.unsqueeze(scale, axis=-1)
    scale = ops.unsqueeze(scale, axis=-1)

    scaled = pic * ops.broadcast_to(scale, (c, h, w))
    grayscaled = ops.sum(scaled, axis=0)
    # max reductions don't remove the dimension, need to squeeze
    return ops.squeeze(grayscaled, axis=0)


@torch.compile
def grayscale_graph(pic: torch.Tensor):
    output = pic.new_empty(pic.shape[1:])
    max_grayscale_graph(output, pic)
    return output


torch.manual_seed(0)
size = (3, 1024, 1024)
x = torch.rand(size, device=DEVICE)
output_torch = torch_grayscale(x)
output_triton = triton_grayscale(x, bs=(32, 32))
output_max = max_grayscale(x)
# output_max_graph = grayscale_graph(x)
print(output_torch)
print(output_triton)
print(output_max)
print(f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch[:] - output_triton[:]))}')
print(f'The maximum difference between torch and MAX is '
    f'{torch.max(torch.abs(output_torch[:] - output_max[:]))}')


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 15)],
        x_log=True,
        line_arg='provider',
        line_vals=['max', 'triton', 'torch'],
        line_names=['MAX', 'Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='gbps',
        plot_name='grayscale performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand((3, size, size), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_grayscale(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_grayscale(x, bs=(32, 32)), quantiles=quantiles)
    if provider == 'max':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: max_grayscale(x), quantiles=quantiles)

    gbps = lambda ms: (4.0 / 3.0) * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

import matplotlib.pyplot as plt

# Using rcParams to change DPI of the figure
plt.rcParams['figure.dpi'] = 3200
benchmark.run(show_plots=True, print_data=True)
