# PyTorch MAX Bridge Workshop

## Overview

This workshop demonstrates bridging PyTorch with Modular MAX, showcasing performance optimizations and GPU acceleration capabilities. The workshop includes interactive slides and hands-on notebooks with benchmarking examples.

## Prerequisites

### Install Pixi

Follow the [official installation guide](https://pixi.sh/latest/installation/).

## Running the Workshop

### 1. Presentation Slides (Multiplatform)

The slides work on all supported platforms (Linux, macOS, Windows):

```bash
# Install slide dependencies
pixi run slide-install

# Run development server with hot reload
pixi run dev-slides

# Or run production slides
pixi run slides

# Generate PDF export
pixi run pdf
```

### 2. Interactive Notebooks (Linux only)

The notebooks require GPU support and are currently available on Linux only.

#### For NVIDIA GPUs:

```bash
# Enter CUDA environment
pixi shell -e cuda

# Start Jupyter Lab
jupyter lab

# Follow the main.notebook for hands-on exercises
# Run benchmarks when prompted:
python rope_benchmark.py

# Exit the pixi environment when done
exit
```

#### For AMD GPUs:

```bash
# Enter ROCm environment
pixi shell -e rocm

# Start Jupyter Lab
jupyter lab

# Follow the main.notebook for hands-on exercises
# Run benchmarks when prompted:
python rope_benchmark.py

# Exit the pixi environment when done
exit
```

## Supported Platforms

- **Slides**: Linux (x64), macOS (x64/ARM64), Windows (x64)
- **Notebooks**: Linux (x64) only - requires NVIDIA CUDA 12+ or AMD ROCm support

## Workshop Contents

1. **Slides**: Introduction to PyTorch-MAX integration and performance benefits
2. **main.notebook**: Interactive Jupyter notebook with step-by-step examples
3. **rope_benchmark.py**: Performance benchmarking script comparing PyTorch vs MAX

## System Requirements

### For CUDA (NVIDIA):
- Linux x64
- CUDA Toolkit 12.x
- NVIDIA GPU with CUDA compute capability

### For ROCm (AMD):
- Linux x64
- ROCm 6.3 compatible AMD GPU

## Dependencies

The workshop automatically manages dependencies through Pixi, including:
- Python 3.10-3.12
- PyTorch 2.7.1 (CUDA/ROCm variants)
- Jupyter Lab & IPython kernel
- Scientific computing libraries (NumPy, SciPy, Matplotlib)
- Transformers library

## Troubleshooting

- **GPU not detected**: Ensure CUDA/ROCm drivers are properly installed
- **Memory issues**: Close other GPU applications or reduce batch sizes
- **Pixi environment issues**: Try `pixi clean` and reinstall

## License

This project is licensed under the LLVM License - see the [LICENSE](../LICENSE) file for details.
