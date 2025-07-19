import math
import compiler
from algorithm import parallelize, vectorize
from sys import simdwidthof
from memory import UnsafePointer
from layout import LayoutTensor, Layout, UNKNOWN_VALUE
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from math import ceildiv

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias Dyn1DLayout = Layout.row_major(UNKNOWN_VALUE)
alias Dyn3DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)

# Core NewGELU computation - can be used in both CPU and GPU contexts
@always_inline
fn new_gelu_computation[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """
    Core NewGELU computation for a single scalar value.

    NewGELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    alias SQRT_2_OVER_PI = Scalar[dtype](0.7978845608028654)  # sqrt(2/pi) for float32
    alias GELU_COEFF = Scalar[dtype](0.044715)

    return 0.5 * x * (1.0 + math.tanh(SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)))

# CPU implementation using simple loop (avoiding parallelize for now)
fn new_gelu_cpu[dtype: DType, layout: Layout](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],

):
    var num_elements = input.shape[0]()

    # Simple sequential loop to avoid segfault
    for i in range(num_elements):
        var x_val = input[i][0]
        var result = new_gelu_computation[dtype](x_val)
        output[i] = result

# GPU kernel implementation for flattened tensor
fn new_gelu_gpu_kernel_flat[dtype: DType](
    output: LayoutTensor[dtype, Dyn1DLayout, MutableAnyOrigin],
    input: LayoutTensor[dtype, Dyn1DLayout, MutableAnyOrigin],
    num_elements: Int,
):
    i = block_idx.x * block_dim.x + thread_idx.x

    if i < num_elements:
        x = input[i][0]
        result = new_gelu_computation[dtype](x)
        output[i] = result

# GPU kernel implementation for static layout tensors (for tests)
fn new_gelu_gpu_kernel[dtype: DType, layout: Layout](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    i = block_idx.x * block_dim.x + thread_idx.x
    num_elements = input.shape[0]()

    if i < num_elements:
        x = input[i][0]
        result = new_gelu_computation[dtype](x)
        output[i] = result

# MAX custom operation using @compiler.register
@compiler.register("new_gelu")
struct NewGELU:
    @staticmethod
    fn execute[target: StaticString](
        # Outputs
        result: OutputTensor[dtype=DType.float32, rank=3], # (batch_size, seq_len, hidden_size)
        # Inputs
        x: InputTensor[dtype=DType.float32, rank=3],
        # Context
        ctx: DeviceContextPtr,
    ) raises:
        # rebind is required to convince the compiler the tensor types are the same
        @parameter
        if target == "cpu":
            # CPU implementation - element-wise operation on 3D tensor
            var batch_size = x.dim_size(0)
            var seq_len = x.dim_size(1)
            var hidden_size = x.dim_size(2)

            # Apply NewGELU element-wise across all dimensions
            for b in range(batch_size):
                for s in range(seq_len):
                    for h in range(hidden_size):
                        var x_val = x[b, s, h]
                        var result_val = new_gelu_computation[dtype](x_val)
                        result[b, s, h] = result_val

        elif target == "gpu":
            # GPU implementation for 3D tensor
            gpu_ctx = ctx.get_device_context()

            # Calculate total number of elements
            var batch_size = x.dim_size(0)
            var seq_len = x.dim_size(1)
            var hidden_size = x.dim_size(2)
            var num_elements = batch_size * seq_len * hidden_size

            # Calculate grid and block dimensions
            var grid_size = ceildiv(num_elements, BLOCK_SIZE)
            var block_size = BLOCK_SIZE

            # Get tensor data for GPU kernel
            var x_tensor = x.to_layout_tensor()
            var output_tensor = result.to_layout_tensor()

            # Launch GPU kernel with flattened indexing
            # Convert 3D tensors to 1D for flattened processing
            var flat_x_tensor = rebind[LayoutTensor[dtype, Dyn1DLayout, MutableAnyOrigin]](x_tensor)
            var flat_output_tensor = rebind[LayoutTensor[dtype, Dyn1DLayout, MutableAnyOrigin]](output_tensor)

            gpu_ctx.enqueue_function[new_gelu_gpu_kernel_flat[dtype]](
                flat_output_tensor,
                flat_x_tensor,
                num_elements,
                grid_dim=grid_size,
                block_dim=block_size,
            )
        else:
            raise Error("Unsupported target: " + target)
