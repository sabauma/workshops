"""
PyTorch-MAX Bridge: RoPE Benchmarking

This module benchmarks different RoPE (Rotary Position Embedding) implementations:
- MAX-optimized implementation
- PyTorch native implementation
- PyTorch with torch.compile

Features:
- Comprehensive performance benchmarking with proper CUDA timing
- Extensive warmup for torch.compile
- Statistical outlier removal and multiple timing metrics
- Rich visualization with matplotlib plots
- Scaling analysis across different tensor sizes

Requirements:
- PyTorch with CUDA support
- MAX framework
- matplotlib for plotting
- numpy

Usage:
    python rope_benchmark.py              # Full benchmark suite with plots
    python rope_benchmark.py --quick      # Quick benchmark with plots
"""

import torch
import torch.nn as nn
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional
import gc
import scipy.stats as stats

# MAX framework imports
import max
from max import torch as mtorch
from max.graph import ops
from max.dtype import DType
from builtins import max as builtin_max

# Define MAX operations at module level
@mtorch.graph_op
def max_rope_forward_query(query, cos, sin):
    """
    MAX-optimized RoPE implementation for query tensor only.
    Processes query tensor separately to handle different shapes in GQA.
    """
    # Apply rotary position embedding using MAX kernels
    q_dtype = query.dtype

    # Convert to float32 for computation
    q_float = query.cast(DType.float32)
    cos_float = cos.cast(DType.float32)
    sin_float = sin.cast(DType.float32)

    # Get dimensions from query tensor
    batch_size, seq_len, num_heads, head_dim = q_float.shape

    # Handle cos/sin tensor shapes with robust logic
    cos_shape = cos_float.shape
    sin_shape = sin_float.shape

    # Calculate expected elements for target shape
    target_shape = (1, seq_len, 1, head_dim)
    target_elements = seq_len * head_dim

    # Calculate actual elements in cos/sin tensors
    cos_elements = 1
    for dim in cos_shape:
        cos_elements *= dim

    # Strategy 1: If cos/sin are already in broadcast-compatible shape, use them
    if len(cos_shape) == 4 and cos_shape[0] == 1 and cos_shape[2] == 1:
        cos_reshaped = cos_float
        sin_reshaped = sin_float
    # Strategy 2: If cos/sin are standard 2D tensors [seq_len, head_dim], reshape them
    elif len(cos_shape) == 2 and cos_shape[0] == seq_len and cos_shape[1] == head_dim:
        cos_reshaped = ops.reshape(cos_float, target_shape)
        sin_reshaped = ops.reshape(sin_float, target_shape)
    # Strategy 3: If cos/sin have the right number of elements, try to reshape
    elif cos_elements == target_elements:
        cos_reshaped = ops.reshape(cos_float, target_shape)
        sin_reshaped = ops.reshape(sin_float, target_shape)
    # Strategy 4: If cos/sin have fewer elements, use broadcasting-friendly shapes
    elif cos_elements < target_elements:
        if len(cos_shape) >= 1 and cos_shape[-1] == head_dim:
            broadcast_shape = [1] * (4 - len(cos_shape)) + list(cos_shape)
            cos_reshaped = ops.reshape(cos_float, broadcast_shape)
            sin_reshaped = ops.reshape(sin_float, broadcast_shape)
        else:
            cos_reshaped = ops.reshape(cos_float, (1, 1, 1, head_dim))
            sin_reshaped = ops.reshape(sin_float, (1, 1, 1, head_dim))
    # Strategy 5: Last resort - slice or repeat as needed
    else:
        if len(cos_shape) == 3 and cos_shape[-1] == head_dim:
            actual_seq_len = cos_shape[1]
            if actual_seq_len >= seq_len:
                cos_sliced = cos_float[:, :seq_len, :]
                sin_sliced = sin_float[:, :seq_len, :]
                cos_reshaped = ops.reshape(cos_sliced, (1, seq_len, 1, head_dim))
                sin_reshaped = ops.reshape(sin_sliced, (1, seq_len, 1, head_dim))
            else:
                cos_reshaped = ops.reshape(cos_float, (1, actual_seq_len, 1, head_dim))
                sin_reshaped = ops.reshape(sin_float, (1, actual_seq_len, 1, head_dim))
        else:
            cos_reshaped = ops.reshape(cos_float, (1, 1, 1, head_dim))
            sin_reshaped = ops.reshape(sin_float, (1, 1, 1, head_dim))

    # Implement rotate_half inline for query
    q_split = ops.split(q_float, split_sizes=[head_dim // 2, head_dim // 2], axis=-1)
    q_x1, q_x2 = q_split[0], q_split[1]
    q_rotated_half = ops.concat([ops.mul(q_x2, -1.0), q_x1], axis=-1)

    # Apply rotation: x * cos + rotate_half(x) * sin
    q_rotated = ops.add(
        ops.mul(q_float, cos_reshaped),
        ops.mul(q_rotated_half, sin_reshaped)
    )

    # Convert back to original dtype - decorator writes to output tensor
    q_out = q_rotated.cast(q_dtype)
    return q_out

@mtorch.graph_op
def max_rope_forward_key(key, cos, sin):
    """
    MAX-optimized RoPE implementation for key tensor only.
    Processes key tensor separately to handle different shapes in GQA.
    """
    # Apply rotary position embedding using MAX kernels
    k_dtype = key.dtype

    # Convert to float32 for computation
    k_float = key.cast(DType.float32)
    cos_float = cos.cast(DType.float32)
    sin_float = sin.cast(DType.float32)

    # Get dimensions from key tensor
    batch_size, seq_len, num_heads, head_dim = k_float.shape

    # Handle cos/sin tensor shapes with robust logic
    cos_shape = cos_float.shape
    sin_shape = sin_float.shape

    # Calculate expected elements for target shape
    target_shape = (1, seq_len, 1, head_dim)
    target_elements = seq_len * head_dim

    # Calculate actual elements in cos/sin tensors
    cos_elements = 1
    for dim in cos_shape:
        cos_elements *= dim

    # Strategy 1: If cos/sin are already in broadcast-compatible shape, use them
    if len(cos_shape) == 4 and cos_shape[0] == 1 and cos_shape[2] == 1:
        cos_reshaped = cos_float
        sin_reshaped = sin_float
    # Strategy 2: If cos/sin are standard 2D tensors [seq_len, head_dim], reshape them
    elif len(cos_shape) == 2 and cos_shape[0] == seq_len and cos_shape[1] == head_dim:
        cos_reshaped = ops.reshape(cos_float, target_shape)
        sin_reshaped = ops.reshape(sin_float, target_shape)
    # Strategy 3: If cos/sin have the right number of elements, try to reshape
    elif cos_elements == target_elements:
        cos_reshaped = ops.reshape(cos_float, target_shape)
        sin_reshaped = ops.reshape(sin_float, target_shape)
    # Strategy 4: If cos/sin have fewer elements, use broadcasting-friendly shapes
    elif cos_elements < target_elements:
        if len(cos_shape) >= 1 and cos_shape[-1] == head_dim:
            broadcast_shape = [1] * (4 - len(cos_shape)) + list(cos_shape)
            cos_reshaped = ops.reshape(cos_float, broadcast_shape)
            sin_reshaped = ops.reshape(sin_float, broadcast_shape)
        else:
            cos_reshaped = ops.reshape(cos_float, (1, 1, 1, head_dim))
            sin_reshaped = ops.reshape(sin_float, (1, 1, 1, head_dim))
    # Strategy 5: Last resort - slice or repeat as needed
    else:
        if len(cos_shape) == 3 and cos_shape[-1] == head_dim:
            actual_seq_len = cos_shape[1]
            if actual_seq_len >= seq_len:
                cos_sliced = cos_float[:, :seq_len, :]
                sin_sliced = sin_float[:, :seq_len, :]
                cos_reshaped = ops.reshape(cos_sliced, (1, seq_len, 1, head_dim))
                sin_reshaped = ops.reshape(sin_sliced, (1, seq_len, 1, head_dim))
            else:
                cos_reshaped = ops.reshape(cos_float, (1, actual_seq_len, 1, head_dim))
                sin_reshaped = ops.reshape(sin_float, (1, actual_seq_len, 1, head_dim))
        else:
            cos_reshaped = ops.reshape(cos_float, (1, 1, 1, head_dim))
            sin_reshaped = ops.reshape(sin_float, (1, 1, 1, head_dim))

    # Implement rotate_half inline for key
    k_split = ops.split(k_float, split_sizes=[head_dim // 2, head_dim // 2], axis=-1)
    k_x1, k_x2 = k_split[0], k_split[1]
    k_rotated_half = ops.concat([ops.mul(k_x2, -1.0), k_x1], axis=-1)

    # Apply rotation: x * cos + rotate_half(x) * sin
    k_rotated = ops.add(
        ops.mul(k_float, cos_reshaped),
        ops.mul(k_rotated_half, sin_reshaped)
    )

    # Convert back to original dtype
    k_out = k_rotated.cast(k_dtype)
    return k_out

def pytorch_rope_forward(query: torch.Tensor, key: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard PyTorch RoPE implementation for comparison.
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos and sin to match query/key dimensions for broadcasting
    # cos/sin: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    batch_size, seq_len, num_heads, head_dim = query.shape
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]

    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)

    return q_embed, k_embed

def pytorch_rope_forward_compiled(query: torch.Tensor, key: torch.Tensor,
                                cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch RoPE implementation with torch.compile optimization.
    """
    @torch.compile
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @torch.compile
    def apply_rope(query, key, cos, sin):
        # Reshape cos and sin to match query/key dimensions for broadcasting
        batch_size, seq_len, num_heads, head_dim = query.shape
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]

        q_embed = (query * cos) + (rotate_half(query) * sin)
        k_embed = (key * cos) + (rotate_half(key) * sin)

        return q_embed, k_embed

    return apply_rope(query, key, cos, sin)

def max_rope_forward_wrapper(q, k, cos, sin):
    """
    Wrapper for MAX RoPE implementation that handles different query/key shapes.
    """
    # Ensure tensors are contiguous for MAX compatibility
    q_contiguous = q.contiguous()
    k_contiguous = k.contiguous()
    cos_contiguous = cos.contiguous()
    sin_contiguous = sin.contiguous()

    # Create output tensors for destination-passing style
    q_output = q_contiguous.new_empty(q_contiguous.shape)
    k_output = k_contiguous.new_empty(k_contiguous.shape)

    # Process query and key separately using MAX implementations
    max_rope_forward_query(q_output, q_contiguous, cos_contiguous, sin_contiguous)
    max_rope_forward_key(k_output, k_contiguous, cos_contiguous, sin_contiguous)

    return q_output, k_output

def benchmark_rope_implementations(batch_size=1, seq_len=512, num_heads=8, head_dim=64):
    """
    Comprehensive benchmark comparing MAX, PyTorch, and torch.compile RoPE implementations.
    Uses proper CUDA events for accurate GPU timing and extensive warmup for torch.compile.
    """
    print("RoPE Implementation Benchmark")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda_events = torch.cuda.is_available()

    # Get GPU information for title
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        gpu_name = "CPU"
        gpu_memory = None
        print("Using CPU")

    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Device: {device}")
    print(f"  Tensor dtype: torch.float16")
    print(f"  Using CUDA events: {use_cuda_events}")
    print()

    # Create multiple tensor sets to avoid cache effects
    num_tensor_sets = 10
    print(f"Creating {num_tensor_sets} tensor sets to avoid cache effects...")
    test_data = []
    for i in range(num_tensor_sets):
        torch.manual_seed(42 + i)  # Different seed for each set
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device=device, dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device=device, dtype=torch.float16)
        test_data.append((query, key, cos, sin))

    # Implementations to benchmark
    implementations = {
        "PyTorch": pytorch_rope_forward,
        "PyTorch + torch.compile": pytorch_rope_forward_compiled,
        "MAX-torch hybrid": max_rope_forward_wrapper
    }

    results = {}
    warmup_runs = 50
    num_runs = 100

    for name, impl in implementations.items():
        print(f"\nBenchmarking {name}...")

        # Use first tensor set for compilation and warmup
        query, key, cos, sin = test_data[0]

        # Separate compilation phase to avoid compilation overhead in timing
        print(f"  Compiling functions...")
        with torch.no_grad():
            for _ in range(5):
                _ = impl(query, key, cos, sin)
                if use_cuda_events:
                    torch.cuda.synchronize()

        # Extended warmup for torch.compile optimization
        print(f"  Warming up ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            query, key, cos, sin = test_data[i % num_tensor_sets]  # Rotate through tensor sets
            _ = impl(query, key, cos, sin)
            if use_cuda_events:
                torch.cuda.synchronize()

        # Clear cache after warmup only
        if use_cuda_events:
            torch.cuda.empty_cache()
        gc.collect()

        # Benchmark iterations
        times = []
        memory_usage = []
        result_q = None  # Initialize result variables
        result_k = None

        print(f"  Running {num_runs} benchmark iterations...")

        for run_idx in range(num_runs):
            # Use different tensor sets to avoid cache effects
            query, key, cos, sin = test_data[run_idx % num_tensor_sets]

            # Reset memory stats for this run
            if use_cuda_events:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                result_q, result_k = impl(query, key, cos, sin)
                end.record()
                torch.cuda.synchronize()

                elapsed_time = start.elapsed_time(end) / 1000.0  # Convert to seconds
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                memory_usage.append(peak_memory)
            else:
                start = time.perf_counter()
                result_q, result_k = impl(query, key, cos, sin)
                end = time.perf_counter()
                elapsed_time = end - start
                memory_usage.append(0)  # No memory tracking for CPU

            times.append(elapsed_time)

            # Minimal cleanup - no cache clearing during benchmark
            # Don't delete result_q, result_k here as we need them for the results dict

        # Calculate robust statistics with improved outlier removal
        times_np = np.array(times)
        # Remove outliers (values beyond 3 std devs from median for less aggressive filtering)
        if len(times_np) > 10:
            median_time = np.median(times_np)
            std_time = np.std(times_np)
            mask = np.abs(times_np - median_time) <= 3 * std_time  # Changed from 2 to 3 std devs
            times_np = times_np[mask]

            # Calculate confidence intervals
            confidence_interval = stats.t.interval(0.95, len(times_np)-1,
                                                     loc=np.mean(times_np),
                                                     scale=stats.sem(times_np))
            print(f"  95% Confidence Interval: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]s")

            if use_cuda_events:
                avg_memory = np.mean(memory_usage)
                print(f"  Average Peak Memory: {avg_memory:.2f} MB")

        avg_time = np.mean(times_np)
        std_time = np.std(times_np)
        min_time = np.min(times_np)
        max_time = np.max(times_np)
        median_time = np.median(times_np)

        results[name] = {
            'times': times_np,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'total_time': np.sum(times_np),
            'result_q': result_q,
            'result_k': result_k,
            'memory_usage': np.mean(memory_usage) if memory_usage else 0
        }

    # Verify numerical accuracy
    print("\nVerifying numerical accuracy...")
    baseline_q = results["PyTorch"]["result_q"]
    baseline_k = results["PyTorch"]["result_k"]

    for name in ["PyTorch + torch.compile", "MAX-torch hybrid"]:
        try:
            torch.testing.assert_close(baseline_q, results[name]["result_q"], rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(baseline_k, results[name]["result_k"], rtol=1e-2, atol=1e-2)
            print(f"  {name}: ✓ Results match baseline")
        except Exception as e:
            print(f"  {name}: ✗ Results differ from baseline: {e}")

    # Print detailed results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Test iterations: {num_runs} (after outlier removal)")
    print()

    baseline_time = results["PyTorch"]["avg_time"]
    baseline_median = results["PyTorch"]["median_time"]

    for name, result in results.items():
        print(f"{name}:")
        print(f"  Average time: {result['avg_time']*1000:.3f} ms")
        print(f"  Median time: {result['median_time']*1000:.3f} ms")
        print(f"  Min time: {result['min_time']*1000:.3f} ms")
        print(f"  Max time: {result['max_time']*1000:.3f} ms")
        print(f"  Standard deviation: {result['std_time']*1000:.3f} ms")
        print(f"  Coefficient of variation: {(result['std_time']/result['avg_time'])*100:.1f}%")
        if use_cuda_events:
            print(f"  Average Peak Memory: {result['memory_usage']:.2f} MB")

        if name != "PyTorch":
            speedup_avg = baseline_time / result['avg_time']
            speedup_median = baseline_median / result['median_time']
            speedup_percent = (baseline_time - result['avg_time']) / baseline_time * 100
            print(f"  Speedup vs PyTorch (avg): {speedup_avg:.2f}x")
            print(f"  Speedup vs PyTorch (median): {speedup_median:.2f}x")
            print(f"  Performance improvement: {speedup_percent:.1f}%")
        print()

    # Find the best implementation by different metrics
    best_avg = min(results.keys(), key=lambda x: results[x]['avg_time'])
    best_median = min(results.keys(), key=lambda x: results[x]['median_time'])
    best_min = min(results.keys(), key=lambda x: results[x]['min_time'])

    print("Performance Summary:")
    print(f"  Best average time: {best_avg} ({results[best_avg]['avg_time']*1000:.3f} ms)")
    print(f"  Best median time: {best_median} ({results[best_median]['median_time']*1000:.3f} ms)")
    print(f"  Best minimum time: {best_min} ({results[best_min]['min_time']*1000:.3f} ms)")

    if best_avg != "PyTorch":
        improvement = baseline_time / results[best_avg]['avg_time']
        print(f"  Overall improvement: {improvement:.2f}x faster than PyTorch")

    print("=" * 50)

    return results, gpu_name

def plot_benchmark_results(results, config_name="", save_dir="plots", gpu_name=""):
    """
    Plot comprehensive benchmark results with multiple visualizations.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))

    # Add GPU name to the title
    if gpu_name and gpu_name != "CPU":
        title = f'RoPE Implementation Benchmark - {gpu_name}\n{config_name}'
    else:
        title = f'RoPE Implementation Benchmark - CPU\n{config_name}'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Performance Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    implementations = list(results.keys())
    avg_times = [results[impl]['avg_time'] * 1000 for impl in implementations]  # Convert to ms
    median_times = [results[impl]['median_time'] * 1000 for impl in implementations]

    x = np.arange(len(implementations))
    width = 0.35

    bars1 = ax1.bar(x - width/2, avg_times, width, label='Average', alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    bars2 = ax1.bar(x + width/2, median_times, width, label='Median', alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    ax1.set_xlabel('Implementation')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(implementations, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # 2. Speedup Comparison
    ax2 = plt.subplot(2, 3, 2)
    baseline_time = results["PyTorch"]["avg_time"]
    speedups = [baseline_time / results[impl]['avg_time'] for impl in implementations]

    x_pos = np.arange(len(implementations))
    bars = ax2.bar(x_pos, speedups, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('Speedup (vs PyTorch)')
    ax2.set_title('Speedup Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(implementations, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add speedup labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        color = 'green' if speedup > 1.0 else 'red'
        ax2.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color=color, weight='bold')

    # 3. Timing Distribution Box Plot
    ax3 = plt.subplot(2, 3, 3)
    timing_data = [np.array(results[impl]['times']) * 1000 for impl in implementations]  # Convert to ms

    box_plot = ax3.boxplot(timing_data, tick_labels=implementations, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Timing Distribution')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Performance Metrics Table
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('tight')
    ax4.axis('off')

    # Create table data
    table_data = []
    headers = ['Implementation', 'Avg (ms)', 'Median (ms)', 'Min (ms)', 'Max (ms)', 'Std (ms)', 'CV (%)']

    for impl in implementations:
        r = results[impl]
        row = [
            impl,
            f"{r['avg_time']*1000:.3f}",
            f"{r['median_time']*1000:.3f}",
            f"{r['min_time']*1000:.3f}",
            f"{r['max_time']*1000:.3f}",
            f"{r['std_time']*1000:.3f}",
            f"{(r['std_time']/r['avg_time'])*100:.1f}"
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Performance Metrics Summary')

    # 5. Coefficient of Variation Comparison
    ax5 = plt.subplot(2, 3, 5)
    cvs = [(results[impl]['std_time']/results[impl]['avg_time'])*100 for impl in implementations]

    x_pos = np.arange(len(implementations))
    bars = ax5.bar(x_pos, cvs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax5.set_ylabel('Coefficient of Variation (%)')
    ax5.set_title('Performance Stability\n(Lower is More Stable)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(implementations, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)

    # Add CV labels
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax5.annotate(f'{cv:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # 6. Timing Histograms
    ax6 = plt.subplot(2, 3, 6)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    alphas = [0.7, 0.6, 0.5]

    for i, impl in enumerate(implementations):
        times_ms = np.array(results[impl]['times']) * 1000
        ax6.hist(times_ms, bins=30, alpha=alphas[i], label=impl, color=colors[i], density=True)

    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Density')
    ax6.set_title('Timing Distribution Histograms')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    if gpu_name and gpu_name != "CPU":
        gpu_name_clean = gpu_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        plot_filename = f"{save_dir}/rope_benchmark_{config_name.replace(' ', '_').lower()}_{gpu_name_clean}.png"
    else:
        plot_filename = f"{save_dir}/rope_benchmark_{config_name.replace(' ', '_').lower()}_CPU.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"  Benchmark plot saved: {plot_filename}")

    # Show the plot
    plt.show()

    return plot_filename

def plot_scaling_analysis(all_results, save_dir="plots"):
    """
    Plot performance scaling across different tensor sizes.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Extract data for plotting
    config_names = list(all_results.keys())
    implementations = list(all_results[config_names[0]].keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Calculate total elements for each config
    total_elements = []
    for config_name in config_names:
        # Parse config name to extract dimensions (this is a simplified approach)
        if "Small" in config_name:
            elements = 1 * 128 * 4 * 32
        elif "Medium" in config_name:
            elements = 1 * 512 * 8 * 64
        elif "Large" in config_name:
            elements = 2 * 1024 * 16 * 128
        elif "Very Large" in config_name:
            elements = 4 * 2048 * 32 * 128
        else:
            elements = 1000  # default
        total_elements.append(elements)

    # 1. Absolute Performance
    for i, impl in enumerate(implementations):
        avg_times = [all_results[config][impl]['avg_time'] * 1000 for config in config_names]
        ax1.plot(total_elements, avg_times, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)

    ax1.set_xlabel('Total Elements')
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Performance Scaling: Absolute Times')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Speedup vs PyTorch
    for i, impl in enumerate(implementations[1:], 1):  # Skip PyTorch baseline
        speedups = []
        for config in config_names:
            baseline = all_results[config]["PyTorch"]['avg_time']
            current = all_results[config][impl]['avg_time']
            speedups.append(baseline / current)

        ax2.plot(total_elements, speedups, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)

    ax2.set_xlabel('Total Elements')
    ax2.set_ylabel('Speedup vs PyTorch')
    ax2.set_title('Performance Scaling: Speedup')
    ax2.set_xscale('log')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Throughput (Elements/sec)
    for i, impl in enumerate(implementations):
        throughputs = []
        for j, config in enumerate(config_names):
            avg_time = all_results[config][impl]['avg_time']
            throughput = total_elements[j] / avg_time
            throughputs.append(throughput)

        ax3.plot(total_elements, throughputs, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)

    ax3.set_xlabel('Total Elements')
    ax3.set_ylabel('Throughput (Elements/sec)')
    ax3.set_title('Performance Scaling: Throughput')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Performance Efficiency (relative to theoretical peak)
    for i, impl in enumerate(implementations):
        efficiencies = []
        for j, config in enumerate(config_names):
            avg_time = all_results[config][impl]['avg_time']
            throughput = total_elements[j] / avg_time
            # Assume theoretical peak throughput (this is a rough estimate)
            theoretical_peak = total_elements[j] * 1000  # elements per second
            efficiency = (throughput / theoretical_peak) * 100
            efficiencies.append(efficiency)

        ax4.plot(total_elements, efficiencies, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)

    ax4.set_xlabel('Total Elements')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('Performance Efficiency')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

        # Save the scaling analysis plot
    plot_filename = f"{save_dir}/rope_scaling_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"  Scaling analysis plot saved: {plot_filename}")

    plt.show()

    return plot_filename

def test_tensor_shapes():
    """Test tensor shape handling in MAX RoPE implementation."""
    print("Testing tensor shape handling...")

    # Test with different tensor shapes that might cause issues
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_cases = [
        {
            "name": "Normal shapes",
            "query": (1, 512, 8, 64),
            "key": (1, 512, 8, 64),
            "cos": (512, 64),
            "sin": (512, 64)
        },
        {
            "name": "Small sequence",
            "query": (1, 32, 8, 64),
            "key": (1, 32, 8, 64),
            "cos": (32, 64),
            "sin": (32, 64)
        },
        {
            "name": "Large sequence",
            "query": (1, 2048, 8, 64),
            "key": (1, 2048, 8, 64),
            "cos": (2048, 64),
            "sin": (2048, 64)
        },
        {
            "name": "Problematic shapes",
            "query": (1, 512, 8, 64),
            "key": (1, 512, 8, 64),
            "cos": (1, 1, 64),
            "sin": (1, 1, 64)
        },
        {
            "name": "Already broadcast shapes",
            "query": (1, 512, 8, 64),
            "key": (1, 512, 8, 64),
            "cos": (1, 512, 1, 64),
            "sin": (1, 512, 1, 64)
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i} - {test_case['name']}:")

        # Create tensors
        query = torch.randn(*test_case["query"], device=device, dtype=torch.float16)
        key = torch.randn(*test_case["key"], device=device, dtype=torch.float16)
        cos = torch.randn(*test_case["cos"], device=device, dtype=torch.float16)
        sin = torch.randn(*test_case["sin"], device=device, dtype=torch.float16)

        print(f"  Query: {query.shape}")
        print(f"  Key: {key.shape}")
        print(f"  Cos: {cos.shape}")
        print(f"  Sin: {sin.shape}")

        try:
            result_q, result_k = max_rope_forward_wrapper(query, key, cos, sin)
            print(f"  SUCCESS - Output shapes: {result_q.shape}, {result_k.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")

def analyze_performance_characteristics():
    """Analyze performance characteristics across different tensor sizes."""
    print("Performance Characteristics Analysis")
    print("=" * 50)

    # Test configurations from small to large
    test_configs = [
        {"name": "Small", "batch_size": 1, "seq_len": 128, "num_heads": 4, "head_dim": 32},
        {"name": "Medium", "batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64},
        {"name": "Large", "batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 128},
        {"name": "Very Large", "batch_size": 4, "seq_len": 2048, "num_heads": 32, "head_dim": 128},
    ]

    all_results = {}

    for config in test_configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  Batch: {config['batch_size']}, Seq: {config['seq_len']}, Heads: {config['num_heads']}, Head dim: {config['head_dim']}")

        # Quick benchmark with fewer iterations for analysis
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Calculate theoretical FLOPS
        total_elements = config['batch_size'] * config['seq_len'] * config['num_heads'] * config['head_dim']
        flops_per_element = 10  # Rough estimate for RoPE operations
        theoretical_flops = total_elements * flops_per_element

        print(f"  Total elements: {total_elements:,}")
        print(f"  Theoretical FLOPS: {theoretical_flops:,}")

        # Store for comparison
        all_results[config['name']] = {
            'config': config,
            'total_elements': total_elements,
            'theoretical_flops': theoretical_flops
        }

    # Show scaling characteristics
    print("\nScaling Analysis:")
    print("=" * 30)
    for name, result in all_results.items():
        elements = result['total_elements']
        flops = result['theoretical_flops']
        print(f"{name:>12}: {elements:>12,} elements, {flops:>15,} FLOPS")

def main():
    """Main function to run RoPE benchmarks with plotting."""
    print("RoPE Implementation Benchmarking Suite")
    print("=" * 40)

    # Show system information
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name()}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test tensor shape handling
    print("\n1. Testing tensor shape handling:")
    test_tensor_shapes()

    # Performance characteristics analysis
    print("\n2. Performance characteristics analysis:")
    analyze_performance_characteristics()

    # Store results for scaling analysis
    all_results = {}

    # Run benchmarks with different sizes and collect results
    print("\n3. Running benchmarks with standard size:")
    results_standard, gpu_name = benchmark_rope_implementations(batch_size=1, seq_len=512, num_heads=8, head_dim=64)
    all_results["Standard"] = results_standard

    print("\nPlotting standard size results...")
    plot_benchmark_results(results_standard, "Standard Size (1×512×8×64)", save_dir="plots", gpu_name=gpu_name)

    print("\n4. Running benchmarks with large size:")
    results_large, gpu_name = benchmark_rope_implementations(batch_size=4, seq_len=2048, num_heads=16, head_dim=128)
    all_results["Large"] = results_large

    print("\nPlotting large size results...")
    plot_benchmark_results(results_large, "Large Size (4×2048×16×128)", save_dir="plots", gpu_name=gpu_name)

    print("\n5. Running benchmarks with very large size:")
    results_very_large, gpu_name = benchmark_rope_implementations(batch_size=8, seq_len=4096, num_heads=32, head_dim=128)
    all_results["Very Large"] = results_very_large

    print("\nPlotting very large size results...")
    plot_benchmark_results(results_very_large, "Very Large Size (8×4096×32×128)", save_dir="plots", gpu_name=gpu_name)

    # Additional smaller benchmark for better scaling analysis
    print("\n6. Running benchmarks with small size:")
    results_small, gpu_name = benchmark_rope_implementations(batch_size=1, seq_len=128, num_heads=4, head_dim=32)
    all_results["Small"] = results_small

    print("\nPlotting small size results...")
    plot_benchmark_results(results_small, "Small Size (1×128×4×32)", save_dir="plots", gpu_name=gpu_name)

    # Plot scaling analysis
    print("\n7. Generating scaling analysis plots...")
    plot_scaling_analysis(all_results, save_dir="plots")

    # Generate summary report
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 60)

    for size_name, results in all_results.items():
        print(f"\n{size_name} Configuration:")
        baseline_time = results["PyTorch"]["avg_time"]

        for impl_name, impl_results in results.items():
            if impl_name != "PyTorch":
                speedup = baseline_time / impl_results['avg_time']
                stability = (impl_results['std_time'] / impl_results['avg_time']) * 100
                print(f"  {impl_name:>20}: {speedup:5.2f}x speedup, {stability:4.1f}% CV")
            else:
                print(f"  {impl_name:>20}: baseline ({impl_results['avg_time']*1000:6.3f} ms)")

    print(f"\nAll plots saved in 'plots/' directory")
    print("\nAll benchmarks completed!")

def quick_benchmark_with_plots():
    """Quick benchmark run focused on plotting results."""
    print("Quick RoPE Benchmark with Plotting")
    print("=" * 40)

    # Run a focused benchmark on medium size for quick results
    print("Running focused benchmark (medium size)...")
    results, gpu_name = benchmark_rope_implementations(batch_size=2, seq_len=1024, num_heads=16, head_dim=64)

    # Plot the results
    print("\nGenerating plots...")
    plot_benchmark_results(results, "Quick Benchmark (2×1024×16×64)", save_dir="plots", gpu_name=gpu_name)

    print("\nQuick benchmark with plots completed!")
    print("Plots saved in 'plots/' directory")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_benchmark_with_plots()
    else:
        main()
