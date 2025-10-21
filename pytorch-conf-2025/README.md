# PyTorch Conference 2025 Workshop
### High-Performance GPU Computing with MAX Engine and Custom Mojo Operations

<p align="center">
  <a href="https://docs.modular.com/mojo">
    <img src="https://img.shields.io/badge/Powered%20by-Mojo-FF5F1F" alt="Powered by Mojo">
  </a>
  <a href="https://docs.modular.com/max/get-started/#stay-in-touch">
    <img src="https://img.shields.io/badge/Subscribe-Updates-00B5AD?logo=mail.ru" alt="Subscribe for Updates">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-Conference%202025-EE4C2C?logo=pytorch" alt="PyTorch Conference 2025">
  </a>
</p>

## Overview

This workshop demonstrates practical GPU acceleration techniques by comparing performance across different compute frameworks:

- **Mojo 🔥**: The best way to programming vendor-agnostic GPU
- **MAX**: Modular's high-performance AI compute and serving engine
- **Custom Mojo Operations**: Hand-optimized GPU kernels using Mojo programming language
- **PyTorch**: Baseline implementations using standard PyTorch operations

The workshop shows how to gradually replace PyTorch operations with MAX implementations for improved performance while maintaining identical model behavior.

## Interactive Demo: Model Surgery with MAX

The `demo.ipynb` notebook demonstrates a practical approach to integrating MAX with existing PyTorch models. Using a GPT-2 conversational model, it shows how to perform "model surgery" by incrementally replacing components:

1. **Stage 1**: Original PyTorch model baseline
2. **Stage 2**: Replace LayerNorm with MAX graph operations
3. **Stage 3**: Replace NewGELU activation with custom Mojo kernels
4. **Stage 4**: Replace entire GPT2MLP layers with fused MAX implementations

Each stage maintains identical model outputs while improving performance. The notebook includes verification tests to ensure numerical correctness and demonstrates how MAX's graph compiler can fuse operations for additional speedups.

## Prerequisites & Installation

### Step 1: Install Pixi Package Manager

Pixi manages all dependencies and environments for this workshop.

```bash
# Install Pixi (cross-platform)
curl -fsSL https://pixi.sh/install.sh | bash
```

**Alternative installation methods:** [Full Pixi Installation Guide](https://pixi.sh/latest/#installation)

### Step 2: GPU Requirements

Ensure your system meets the GPU requirements for optimal performance:

**[MAX GPU Requirements](https://docs.modular.com/max/packages#gpu-compatibility)**

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/modular/workshops
cd pytorch-conf-2025
```

### 2. Activate Environment

Choose your GPU vendor:

#### For NVIDIA GPUs:
```bash
pixi shell -e nvidia
```

#### For AMD GPUs:
```bash
pixi shell -e amd
```

### 3. Run Benchmarks

Execute performance comparisons for operations:

```bash
# Run grayscale conversion benchmark
python benchmark.py --case grayscale --save results
```

### 4. Interactive Demo

Launch Jupyter Lab for hands-on exploration:

```bash
jupyter lab
```

Open `demo.ipynb` to explore:
- Interactive comparisons between frameworks
- Performance visualizations and analysis
- Experiment with different parameters
- Step-by-step explanations of each technique

## Benchmark Framework

Automated performance comparison system:

- **Real-time execution**: Live performance monitoring
- **Multiple runs**: Statistical accuracy with error bars
- **Multi-framework**: Direct comparisons across implementations
- **Export results**: Save data and plots for analysis

## Learning Objectives

After completing this workshop, you'll understand:

1. **GPU Acceleration Techniques**
   - Memory coalescing and bandwidth optimization
   - Compute vs memory-bound operations
   - Cache hierarchy utilization

2. **Framework Comparison**
   - When to use MAX Engine vs PyTorch
   - Performance trade-offs and optimization strategies
   - Custom kernel development workflows

3. **Performance Analysis**
   - Benchmarking methodologies
   - Interpreting throughput metrics (GB/s)
   - Identifying performance bottlenecks

4. **Practical Implementation**
   - Writing efficient GPU kernels in Mojo
   - Integrating custom operations with PyTorch
   - Production deployment considerations

## License

This project is licensed under the LLVM License - see the [LICENSE](../LICENSE) file for details.

<p align="center">
  <sub>Built with ❤️ for PyTorch Conference 2025 by the Modular team</sub>
</p>
