## Mojo🔥 GPU Programming Workshop

### June 27, 2025

<div style="text-align: center;">
<img src="./image/workshop.png" alt="workshop" width="600" height="400">
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
ctx.enqueue_function[vector_add[DType.int32, size]](
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

### GPU Programming Fundamentals

---

## Clone the Mojo🔥 GPU Puzzles

```bash
git clone https://github.com/modular/mojo-gpu-puzzles
cd mojo-gpu-puzzles
```

🌐 **Online Version**: [builds.modular.com/puzzles](https://builds.modular.com/puzzles)

📝 **Note**: Extensive use in Part 2!

---

# Thank You! 🔥

## Questions & Discussion

_Let's build the future of AI programming together!_

**GPU Puzzles**
[github.com/modular/mojo-gpu-puzzles](https://github.com/modular/mojo-gpu-puzzles)

**Community**
[community.modular.com](https://community.modular.com/)

Note: Thank you for joining this workshop! Continue learning with the resources provided, and don't hesitate to reach out to the community for support.
