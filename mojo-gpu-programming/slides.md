## Mojo🔥 GPU Programming Workshop

<div style="text-align: center;">
<img src="./image/workshop.png" alt="workshop" width="800" height="400">
</div>

---

# Workshop Agenda

### Part 1: Foundations

**1. Why Mojo🔥?**

**2. Setup Using [Pixi](https://pixi.sh/latest/)**

**3. Introduction to Mojo🔥 for GPU Programming**

### Part 2: Practical Implementation

**4. Hands-on GPU Programming in Mojo**

**5. Create your Own AI Model with MAX🧑‍🚀**

**6. PyTorch Integration**

---

# Resources

- **Introduction to GPU Programming in Mojo🔥**
[https://docs.modular.com/mojo/manual/gpu/architecture](https://docs.modular.com/mojo/manual/gpu/architecture)

- **Mojo🔥 GPU Puzzles** [github.com/modular/mojo-gpu-puzzles](https://github.com/modular/mojo-gpu-puzzles)

- **Optimize custom ops for GPUs with Mojo🔥** [https://docs.modular.com/max/tutorials/custom-ops-matmul](https://docs.modular.com/max/tutorials/custom-ops-matmul)

- **Forum** [forum.modular.com](https://forum.modular.com/)

---

<!-- .slide: class="center-slide" -->
# Part 1: Foundations

---

### Why Mojo🔥?

- 🤝 **Heterogenous _System_ Programming Language**
- 🐍 **Python-like Syntax**
- ⚡ **Zero-cost Abstractions**
- 🛡️ **Strong Type System**
- 🦀 **Memory Safety**

---

### Performance Benefits

- 📊 **Built-in SIMD Support**
- 🔧 **Direct Hardware Access**
- 🔄 **Cross-Hardware Portability**

_Combining the best of Python, C++, and Swift/Rust_

Note: Mojo represents a paradigm shift in systems programming, making GPU programming accessible without sacrificing performance.

---

<!-- .slide: class="center-slide" -->

# GPU Programming

## Traditional vs Mojo🔥

---

### ❌ Traditional CUDA Pain Points

**Complex and Error-Prone**

```cpp
// Manual memory management, verbose boilerplate
int *d_a, *d_b, *d_c;
cudaMalloc(&d_a, size * sizeof(int));
cudaMalloc(&d_b, size * sizeof(int));
cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
vector_add<<<grid, block>>>(d_a, d_b, d_c);
cudaDeviceSynchronize();
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
```

⚠️ **Manual memory management • Verbose syntax • Error-prone**

---

### ✅ Mojo🔥 Elegance

**Python Simplicity, C++ Performance**

```mojo
ctx = DeviceContext()
a = ctx.enqueue_create_buffer[dtype](size)
b = ctx.enqueue_create_buffer[dtype](size)
c = ctx.enqueue_create_buffer[dtype](size)
ctx.enqueue_function[vector_add[dtype, size]](
    a.unsafe_ptr(), b.unsafe_ptr(), c.unsafe_ptr(),
    grid_dim=blocks, block_dim=threads
)
ctx.synchronize()
```

✨ **Automatic memory management • Clean syntax • Type-safe**

---

<!-- .slide: class="center-slide" -->
## Mojo🔥 GPU Advantages

## Built for Modern Development

---

### Development Features

- 🚀 **Direct hardware access** with Python ergonomics
- 🔧 **Built-in GPU types**: `DeviceContext`, `LayoutTensor`
- 💾 **Automatic memory management** with ASAP destruction

---

### Performance Features

- ⚡ **Async operations** between CPU and GPU
- 🎯 **Thread-safe by design** for massive parallelism
- 📈 **Compile-time optimizations** for peak performance

---

## Setup Using Pixi

### Prerequisites

- 📖 **Documentation**: [docs.modular.com/max/get-started](https://docs.modular.com/max/get-started)

- **OS Support:**
  - Linux/Ubuntu 22.04+
  - macOS 13+
  - Windows WSL2

- **Python: >=3.9, < 3.13**

- **[Compatible GPUs](https://docs.modular.com/max/faq/#gpu-requirements)**
  - Tier 1: NVIDIA H100/200, A100/A10, L4/L40 and AMD MI300X and MI325X
  - Tier 2: NVIDIA RTX 40XX/30XX series
  - Tier 3: Check the documentation

---

## 🛠️ Installation Steps

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | sh
```

```bash
# Create a new project
pixi init quickstart \
    -c https://conda.modular.com/max-nightly/ -c conda-forge \
    && cd quickstart
```

```bash
# Add MAX that includes Mojo
pixi add max

# Verify installation
pixi run mojo --version
```

🔍 **Environment Check**: Follow along - we'll troubleshoot together!

Note: Make sure everyone has their environment working before we proceed to coding.

---

<!-- .slide: class="center-slide" -->
# Introduction to Mojo🔥

### Essentials for GPU Programming

---

### 🧠 Core Language Concepts

```mojo
# `main` function is the entry point of the program
fn main():
    print("Hello, World!")

# Basic function
fn add(a: Int32, b: Int32) -> Int32:
    return a + b

# Parametric function: `param` is known at compile-time
fn kernel[size: Int, dtype: DType](data: UnsafePointer[dtype]):
    # Parameters drive GPU performance
```

⚠️ **Key Point**: `fn` functions are strict, essential for GPU kernels (no exceptions allowed)

---

### Python Compatible `def` Function

```mojo
def main():
    try:
        ...
    except:
        raise Error("Exception occured!")

# equivalent to

fn main() raises:
    try:
        ...
    except:
        raise Error("Exception occured!")
```

---

### Variable Declaration

- Python-like

```mojo
a = 42
```

- The use of `var` is mostly optional but is required for type-checking

```mojo
var a: UInt128 = 42
# Same as
a = UInt128(42)
```

---

### Function Types Comparison

- **`fn` Functions:**
  - ✅ **Strict and Safe**
  - No exceptions by default
  - Required for GPU kernels

- **`def` Functions:**
  - 🐍 **Python Compatible**
  - Can raise exceptions
  - More flexible
  - Runtime behavior
  - Interop friendly

🔥 **GPU Rule**: Use `fn` for kernels - GPU hardware cannot handle exceptions!

---

### Type System Highlights

- `Int32/64` is a 32/64-bit integer
- `Float32/64` is a 32/64-bit floating point number
- `Bool` is a boolean
- `String` is a string
- `List[T]` is a list of type `T`
- `Pointer[T]` is a safe (non-null) pointer to type `T`
- `UnsafePointer[T]` is an unsafe pointer to type `T`
- `SIMD[DType, width: Int]` is a SIMD vector

---

<!-- .slide: class="center-slide" -->
`Float32` = `Scalar[DType.float32]` = `SIMD[DType.float32, 1]`

```mojo
# compile-time constant
alias size = 4
# SIMD types for vectorization
vector = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

# Memory management
ptr = UnsafePointer[Float32].alloc(1000)
# Explicit cleanup
ptr.free()
```

⚡ **Performance Tip**: SIMD types unlock vectorization for massive parallelism!

---

### Structs

```mojo
struct Vector3D:
    var x: Float32
    var y: Float32
    var z: Float32

    fn __init__(out self, x: Float32, y: Float32, z: Float32):
        self.x = x
        self.y = y
        self.z = z

    fn magnitude(self) -> Float32:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
```

---

### Conditional Statements

```mojo
fn check_value(x: Int32) -> String:
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
```

---

### Loops with `for` and `while`

```mojo
fn iterate_example():
    for i in range(5):
        print(i)

    count = 0
    while count < 3:
        print("count:", count)
        count += 1
```

---

### Break and Continue

```mojo
fn loop_control():
    for i in range(10):
        if i == 3:
            continue  # Skip iteration
        if i == 7:
            break     # Exit loop
        print(i)
```

💡 **GPU Note**: Control flows in kernels should be minimized for optimal performance!

---

### Key Takeaways

🔥 **`fn` for kernels** (no exceptions)

⚡ **SIMD types** Vectorization

🏗️ **Structs** for Data Structures

> Note: These fundamentals are crucial - they enable the performance characteristics that make Mojo special for CPU/GPU programming.

---

<!-- .slide: class="center-slide" -->
# Part 2: Practical Implementation

---

<!-- .slide: class="center-slide" -->
## Hands-on GPU Programming with Mojo🔥

---

### Clone the Mojo🔥 GPU Puzzles

```bash
git clone https://github.com/modular/mojo-gpu-puzzles
cd mojo-gpu-puzzles
```

Make sure you've Pixi installed. See [https://prefix.dev/](https://prefix.dev/)

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

🌐 **Online Version**: [builds.modular.com/puzzles](https://builds.modular.com/puzzles)

📝 **Note**: Extensive use in Part 2!

---

<!-- .slide: class="center-slide" -->
## GPU Programming Fundamentals

---

<div style="display: flex; align-items: flex-start; gap: 40px;">
<div style="flex: 1;">

### GPU vs CPU Architecture
- **CPUs**: Few powerful cores optimized for sequential processing and complex decision making
- **GPUs**: Thousands of smaller, simpler cores designed for parallel processing
- **Complementary**: CPU handles program flow and complex logic, GPU handles parallel computations

</div>
<div style="flex: 0 0 500px;">
<img src="./image/cpu-gpu-architecture.png" alt="CPU vs GPU Architecture" width="500" height="400" style="border-radius: 5px;">
</div>
</div>

---

### GPU Hardware Components

<img src="./image/sm-architecture.png" alt="SM Architecture" width="1000" height="400" style="border-radius: 5px;">

---

### GPU Hardware Components

- **Streaming Multiprocessors (SMs)**: Self-contained processing factories that operate independently
- **CUDA Cores/Stream Processors**: Basic arithmetic units performing calculations
- **Tensor/Matrix Cores**: Specialized units for matrix multiplication and AI operations
- **Register Files**: Ultra-fast storage for thread-specific data
- **Shared Memory/L1 Cache**: Low-latency memory for data sharing between threads

---

### GPU Execution Model

<img src="./image/grid-hierarchy.png" alt="Grid hierarchy" width="700" height="600" style="border-radius: 5px;">

---

### GPU Execution Model
- **Kernel**: Function that runs on GPU across thousands/millions of threads
- **Grid**: Top-level organization of all threads executing a kernel
- **Thread Blocks**: Groups of threads that can collaborate and share memory
- **Warps**: Subsets of 32-64 threads that execute the same instruction simultaneously
- **SIMT**: Single Instruction, Multiple Threads execution model

---

### Key Performance Concepts
- **Warp Divergence**: Performance penalty when threads in a warp take different paths
- **Memory Coalescing**: Efficient memory access when threads access contiguous locations
- **Thread Block Sizing**: Should be multiples of warp size for optimal resource usage
- **Latency Hiding**: GPU switches between warps to hide memory access delays

---

### Your First Kernel

Print block and thread ids for `grid_dim=2` and `block_dim=4`

In `main.mojo`:

```mojo
from gpu import block_idx, thread_idx
from gpu.host import DeviceContext

fn print_threads():
    print("Block index:", block_idx.x, "\t", "Thread index: ", thread_idx.x)

def main():
    # create device context
    ctx = DeviceContext()
    # GPU kernel launches asynchronously - doesn't block Host (CPU)
    ctx.enqueue_function[print_threads](grid_dim=2, block_dim=4)
    # synchronize Host thread with GPU
    ctx.synchronize()
```

---

### Your First Kernel

Run `pixi run mojo main.mojo`

```output
 Block index: 1          Thread index:  0
 Block index: 1          Thread index:  1
 Block index: 1          Thread index:  2
 Block index: 1          Thread index:  3
 Block index: 0          Thread index:  0
 Block index: 0          Thread index:  1
 Block index: 0          Thread index:  2
 Block index: 0          Thread index:  3
```

👁️ Notice the Parallel execution of GPU threads

---

## [🧩Puzzle 1: Map](https://builds.modular.com/puzzles/puzzle_01/puzzle_01.html)

Adds `10` to each position of vector `a` and stores it in vector `output`.

</div>
<div style="flex: 0 0 800px;">
<img src="./gifs/puzzle_01_viz.gif" alt="Puzzle 01" width="800" height="400" style="border-radius: 5px;">
</div>
</div>

---

## [🧩Puzzle 1: Map](https://builds.modular.com/puzzles/puzzle_01/puzzle_01.html)

Adds `10` to each position of vector `a` and stores it in vector `output`.

```mojo
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE
alias dtype = DType.float32

fn add_10(
    output: UnsafePointer[Scalar[dtype]], a: UnsafePointer[Scalar[dtype]]
):
    i = thread_idx.x
    # FILL ME IN (roughly 1 line)
```

Run `pixi run p01`

---

## Solution

```mojo
fn add_10(
    output: UnsafePointer[Scalar[dtype]], a: UnsafePointer[Scalar[dtype]]
):
    i = thread_idx.x
    output[i] = a[i] + 10
```

---

## [🧩Puzzle 3: Guards](https://builds.modular.com/puzzles/puzzle_03/puzzle_03.html)

Add `10` to each position of vector `a` and stores it in vector `output`. You have **more threads than positions**.

</div>
<div style="flex: 0 0 800px;">
<img src="./gifs/puzzle_03_viz.gif" alt="Puzzle 03" width="800" height="400" style="border-radius: 5px;">
</div>
</div>

---

## [🧩Puzzle 3: Guards](https://builds.modular.com/puzzles/puzzle_03/puzzle_03.html)

Add `10` to each position of vector `a` and stores it in vector `output`. You have **more threads than positions**.

```mojo
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (8, 1)
alias dtype = DType.float32

fn add_10_guard(output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    i = thread_idx.x
    # FILL ME IN (roughly 2 lines)
```

Run `pixi run p03`

---

## Solution

```mojo
fn add_10_guard(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    i = thread_idx.x
    if i < size:
        output[i] = a[i] + 10.0
```

---

## [🧩Puzzle 4: 2D Map](https://builds.modular.com/puzzles/puzzle_04/puzzle_04.html)

Add `10` to each position of 2D square matrix `a` and stores it in 2D square matrix `output`. Note `a` is row-major i.e. rows are stored in memory.

</div>
<div style="flex: 0 0 800px;">
<img src="./gifs/puzzle_04_viz.gif" alt="Puzzle 04" width="800" height="400" style="border-radius: 5px;">
</div>
</div>

---

## Thread Indexing Convention

</div>
<div style="flex: 0 0 800px;">
<img src="./gifs/thread_indexing_viz.gif" alt="thread indexing" width="800" height="400" style="border-radius: 5px;">
</div>
</div>

---

## [🧩Puzzle 4: 2D Map](https://builds.modular.com/puzzles/puzzle_04/puzzle_04.html)

Add `10` to each position of 2D square matrix `a` and stores it in 2D square matrix `output`. Note `a` is row-major i.e. rows are stored in memory.

```mojo
alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32

fn add_10_2d(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    # FILL ME IN (roughly 2 lines)
```

---

## Solution

Indexing convention:
- `thread_idx.y` corresponds to the row index
- `thread_idx.x` corresponds to the column index

```mojo
fn add_10_2d(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0
```

---

## Why `LayoutTensor`?

Growing challenging with raw `UnsafePointer`

```mojo
# 2D indexing coming in later puzzles
idx = row * WIDTH + col

# 3D indexing
idx = (batch * HEIGHT + row) * WIDTH + col

# With padding
idx = (batch * padded_height + row) * padded_width + col
```

---

## Why `LayoutTensor`?

Idiomatic access `LayoutTensor`:

```mojo
output[i, j] = a[i, j] + 10.0  # 2D indexing
output[b, i, j] = a[b, i, j] + 10.0  # 3D indexing
```

and more advanced features preview:

```mojo
# Column-major layout
layout_col = Layout.col_major(HEIGHT, WIDTH)
# Tiled layout (for better cache utilization)
layout_tiled = tensor.tiled[4, 4](HEIGHT, WIDTH)
# Vectorized access
vec_tensor = tensor.vectorize[1, simd_width]()
# Asynchronous transfers
copy_dram_to_sram_async[thread_layout=layout](dst, src)
# Tensor Core operations (coming in later puzzles)
mma_op = TensorCore[dtype, out_type, Index(M, N, K)]()
result = mma_op.mma_op(a_reg, b_reg, c_reg)
```

---

## How to use `LayoutTensor`?

`LayoutTensor` can be both on CPU and GPU depending on where the underlying data pointer lives.

```mojo
from layout import Layout, LayoutTensor
# Define layout
alias layout = Layout.row_major(2, 3)
# Allocate and initialized memory either on Host (CPU) or Device (GPU)
data_ptr = ...
# Create tensor using the `date_ptr`
tensor = LayoutTensor[mut=True, dtype, layout](data_ptr)
tensor[0, 0] += 1
```

---

### Back to Puzzle 4: 2D Map

**Raw `UnsafePointer` approach:**

```mojo
row = thread_idx.y
col = thread_idx.x
if row < size and col < size:
    output[row * size + col] = a[row * size + col] + 10.0
```

**`LayoutTensor` approach:**

```mojo
row = thread_idx.y
col = thread_idx.x
if row < size and col < size:
    output[row, col] = a[row, col] + 10.0
```

---

## [🧩Puzzle 8: Shared Memory](https://builds.modular.com/puzzles/puzzle_08/puzzle_08.html)

Add `10` to each position of a vector `a` and stores it in vector `output`. You have **fewer threads per block** than the size of `a`.

</div>
<div style="flex: 0 0 800px;">
<img src="./gifs/puzzle_08_viz.gif" alt="Puzzle 08" width="800" height="400" style="border-radius: 5px;">
</div>
</div>

---

## [🧩Puzzle 8: Shared Memory](https://builds.modular.com/puzzles/puzzle_08/puzzle_08.html)

Add `10` to each position of a vector `a` and stores it in vector `output`. You have **fewer threads per block** than the size of `a`.

```mojo
# Allocate shared memory using tensor builder
shared = tb[dtype]().row_major[TPB]().shared().alloc()

global_i = block_dim.x * block_idx.x + thread_idx.x
local_i = thread_idx.x
# Copy local data into shared memory
if global_i < size:
    shared[local_i] = a[global_i]

# wait for all threads to complete works within a thread block
barrier()

# FILL ME IN (roughly 2 lines)
```

---

## Solution

```mojo
# Allocate shared memory using tensor builder
shared = tb[dtype]().row_major[TPB]().shared().alloc()

global_i = block_dim.x * block_idx.x + thread_idx.x
local_i = thread_idx.x

if global_i < size:
    shared[local_i] = a[global_i]
# Copy local data into shared memory
barrier()
# Bound-check and store to global memory
if global_i < size:
    output[global_i] = shared[local_i] + 10
```

---

<!-- .slide: class="center-slide" -->
## Model Development with MAX🧑‍🚀 (TODO: Brad)

---

<!-- .slide: class="center-slide" -->
## PyTorch Custom Ops Integration (Beta Feature)

### 🚀 Mojo Kernels 🤝 PyTorch Models

---

### Key Benefits

- ⚡ **High-performance kernels** in Mojo🔥
- 🐍 **Familiar PyTorch workflow**
- 🔄 **Seamless integration** - no framework switching
- 🎯 **Target-agnostic** - same code runs on CPU/GPU

---

### Simple Integration

```python
import torch
from max.torch import CustomOpLibrary

# Load compiled Mojo kernels
ops = CustomOpLibrary("./operations")

# Use in PyTorch as normal
@torch.compile
def my_model(x):
    return ops.my_custom_kernel(x)
```

📖 **Tutorial**: [docs.modular.com/max/tutorials/custom-kernels-pytorch](https://docs.modular.com/max/tutorials/custom-kernels-pytorch)

💻 **Code**: [https://github.com/modular/modular/tree/main/examples/pytorch_custom_ops](https://github.com/modular/modular/tree/main/examples/pytorch_custom_ops)

---

### Up for a challenge? ✋

- [🧩 Puzzle 19: Embedding Op](https://builds.modular.com/puzzles/puzzle_19/puzzle_19.html)
- [🧩 Puzzle 20: Kernel Fusion and Custom Backward Pass](https://builds.modular.com/puzzles/puzzle_20/puzzle_20.html)

---

<!-- .slide: class="center-slide" -->
# Thank You! 🔥

## Questions & Discussion

_Let's build the future of AI programming together!_

Note: Thank you for joining this workshop! Continue learning with the resources provided, and don't hesitate to reach out to the community for support.
