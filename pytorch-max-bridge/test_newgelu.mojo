import math
from testing import assert_almost_equal, assert_true, assert_false
from memory import UnsafePointer, memset_zero
from random import random_float64
from time import perf_counter
from tensor import InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from math import ceildiv

from ops import new_gelu_computation, new_gelu_cpu, new_gelu_gpu_kernel_flat, new_gelu_gpu_kernel, NewGELU

alias dtype = DType.float32
alias Dyn1DLayout = Layout.row_major(UNKNOWN_VALUE)
alias Dyn3DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
alias TEST_SIZE = 1024
alias TOLERANCE = 1e-5
alias StaticLayout = Layout.row_major(TEST_SIZE)
# Test dimensions for 3D tensors (batch, sequence, hidden_size)
alias TEST_BATCH_SIZE = 2
alias TEST_SEQ_LEN = 16
alias TEST_HIDDEN_SIZE = 32
alias Static3DLayout = Layout.row_major(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_HIDDEN_SIZE)

def test_new_gelu_single_values():
    """Test NewGELU computation for individual scalar values."""
    print("Testing NewGELU single scalar computation...")

    # Test specific values with known approximate results
    var test_values = List[Float64]()
    var expected_values = List[Float64]()

    test_values.append(0.0)
    expected_values.append(0.0)
    test_values.append(1.0)
    expected_values.append(0.8411919)
    test_values.append(-1.0)
    expected_values.append(-0.1588081)
    test_values.append(2.0)
    expected_values.append(1.9545977)
    test_values.append(-2.0)
    expected_values.append(-0.0454023)
    test_values.append(0.5)
    expected_values.append(0.3457068)
    test_values.append(-0.5)
    expected_values.append(-0.1542932)

    for i in range(len(test_values)):
        var input_val = Scalar[dtype](test_values[i])
        var expected = Scalar[dtype](expected_values[i])
        var result = new_gelu_computation[dtype](input_val)

        # print("  NewGELU test passed for value", input_val)
        assert_almost_equal[dtype](result, expected, rtol=TOLERANCE, atol=TOLERANCE)

    print("✓ Single scalar tests passed")

def test_new_gelu_mathematical_properties():
    """Test mathematical properties of NewGELU."""
    print("Testing NewGELU mathematical properties...")

    # Test that NewGELU(0) = 0
    var zero_result = new_gelu_computation[dtype](Scalar[dtype](0.0))
    assert_almost_equal[dtype](zero_result, Scalar[dtype](0.0), rtol=1e-10, atol=1e-10)

    # Test monotonicity (NewGELU should be monotonically increasing)
    var x1 = Scalar[dtype](-1.0)
    var x2 = Scalar[dtype](1.0)
    var result1 = new_gelu_computation[dtype](x1)
    var result2 = new_gelu_computation[dtype](x2)
    assert_true(result1 < result2, "NewGELU should be monotonically increasing")

    # Test asymptotic behavior for large positive values
    var large_pos = Scalar[dtype](5.0)
    var large_pos_result: Scalar[dtype] = new_gelu_computation[dtype](large_pos)
    assert_true(abs(large_pos_result - large_pos) < 0.01, "NewGELU(x) ≈ x for large positive x")

    # Test asymptotic behavior for large negative values
    var large_neg = Scalar[dtype](-5.0)
    var large_neg_result: Scalar[dtype] = new_gelu_computation[dtype](large_neg)
    assert_true(abs(large_neg_result) < 0.01, "NewGELU(x) ≈ 0 for large negative x")

    print("✓ Mathematical properties tests passed")

def test_new_gelu_cpu_implementation():
    """Test CPU implementation of NewGELU."""
    print("Testing NewGELU CPU implementation...")

    # Allocate host buffers
    var host_input = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var host_output = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var host_reference = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)

    # Initialize input data with random values
    for i in range(TEST_SIZE):
        var val = random_float64(-3.0, 3.0).cast[dtype]()
        host_input[i] = val
        # Compute reference using single element function
        host_reference[i] = new_gelu_computation[dtype](val)

    # Create host layout tensors directly (no device context needed for CPU)
    var input_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](host_input))
    var output_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](host_output))

    # Test CPU implementation
    var start_time = perf_counter()
    new_gelu_cpu[dtype, StaticLayout](output_tensor, input_tensor)
    var cpu_time = perf_counter() - start_time

    # Verify results
    var max_error = Scalar[dtype](0)
    for i in range(TEST_SIZE):
        var error = abs(host_output[i] - host_reference[i])
        if error > max_error:
            max_error = error

    print("  CPU implementation time: " + String(cpu_time) + " seconds")
    print("  Maximum error vs reference: " + String(max_error))
    assert_true(max_error < TOLERANCE, "CPU implementation accuracy check failed")

    # Cleanup
    host_input.free()
    host_output.free()
    host_reference.free()

    print("✓ CPU implementation tests passed")

def test_new_gelu_gpu_implementation():
    """Test GPU implementation of NewGELU."""
    print("Testing NewGELU GPU implementation...")

    # Allocate host buffers
    var host_input = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var host_output = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var host_reference = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)

    # Initialize input data with random values
    for i in range(TEST_SIZE):
        var val = random_float64(-3.0, 3.0).cast[dtype]()
        host_input[i] = val
        # Compute reference using single element function
        host_reference[i] = new_gelu_computation[dtype](val)

    # Create device context and buffers
    with DeviceContext() as ctx:
        # Create device buffers
        var device_input = ctx.enqueue_create_buffer[dtype](TEST_SIZE)
        var device_output = ctx.enqueue_create_buffer[dtype](TEST_SIZE)

        # Create host buffers for device communication
        var host_in_buf = ctx.enqueue_create_host_buffer[dtype](TEST_SIZE)
        var host_out_buf = ctx.enqueue_create_host_buffer[dtype](TEST_SIZE)

        # Copy data to host buffer
        for i in range(TEST_SIZE):
            host_in_buf.unsafe_ptr()[i] = host_input[i]

        # Copy to device
        host_in_buf.enqueue_copy_to(device_input)

        # Create layout tensors using the correct type for flattened access
        var input_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](device_input))
        var output_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](device_output))

        # Test GPU implementation
        var start_time = perf_counter()

        # Launch GPU kernel
        var grid_size = ceildiv(TEST_SIZE, 256)
        ctx.enqueue_function[new_gelu_gpu_kernel[dtype, StaticLayout]](
            output_tensor,
            input_tensor,
            grid_dim=grid_size,
            block_dim=256,
        )
        ctx.synchronize()

        var gpu_time = perf_counter() - start_time

        # Copy result back
        device_output.enqueue_copy_to(host_out_buf)
        ctx.synchronize()

        # Copy to host output
        for i in range(TEST_SIZE):
            host_output[i] = host_out_buf.unsafe_ptr()[i]

        # Verify results
        var max_error = Scalar[dtype](0)
        for i in range(TEST_SIZE):
            var error = abs(host_output[i] - host_reference[i])
            if error > max_error:
                max_error = error

        print("  GPU implementation time: " + String(gpu_time) + " seconds")
        print("  Maximum error vs reference: " + String(max_error))
        assert_true(max_error < TOLERANCE, "GPU implementation accuracy check failed")

    # Cleanup
    host_input.free()
    host_output.free()
    host_reference.free()

    print("✓ GPU implementation tests passed")

def test_new_gelu_cpu_vs_gpu():
    """Test CPU vs GPU implementation consistency."""
    print("Testing CPU vs GPU implementation consistency...")

    # Allocate host buffers
    var host_input = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var host_cpu_output = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var host_gpu_output = UnsafePointer[Scalar[dtype]].alloc(TEST_SIZE)
    var cpu_time: Float64 = 0.0
    var gpu_time: Float64 = 0.0

    # Initialize input data with random values
    for i in range(TEST_SIZE):
        host_input[i] = random_float64(-3.0, 3.0).cast[dtype]()

    # Test CPU implementation on HOST memory (outside device context)
    var cpu_input_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](host_input))
    var cpu_output_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](host_cpu_output))

    var cpu_start = perf_counter()
    new_gelu_cpu[dtype, StaticLayout](cpu_output_tensor, cpu_input_tensor)
    cpu_time = perf_counter() - cpu_start

    # Test GPU implementation on DEVICE memory (inside device context)
    with DeviceContext() as ctx:
        # Create device buffers
        var device_input = ctx.enqueue_create_buffer[dtype](TEST_SIZE)
        var device_output = ctx.enqueue_create_buffer[dtype](TEST_SIZE)

        # Create host buffers for device communication
        var host_in_buf = ctx.enqueue_create_host_buffer[dtype](TEST_SIZE)
        var host_out_buf = ctx.enqueue_create_host_buffer[dtype](TEST_SIZE)

        # Copy data to host buffer
        for i in range(TEST_SIZE):
            host_in_buf.unsafe_ptr()[i] = host_input[i]

        # Copy to device
        host_in_buf.enqueue_copy_to(device_input)

        # Create GPU layout tensors
        var gpu_input_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](device_input))
        var gpu_output_tensor = rebind[LayoutTensor[dtype, StaticLayout, MutableAnyOrigin]](LayoutTensor[dtype, StaticLayout](device_output))

        # Test GPU implementation
        var gpu_start = perf_counter()
        var grid_size = ceildiv(TEST_SIZE, 256)
        ctx.enqueue_function[new_gelu_gpu_kernel[dtype, StaticLayout]](
            gpu_output_tensor,
            gpu_input_tensor,
            grid_dim=grid_size,
            block_dim=256,
        )
        ctx.synchronize()
        gpu_time = perf_counter() - gpu_start

        # Copy GPU results back to host
        device_output.enqueue_copy_to(host_out_buf)
        ctx.synchronize()

        # Copy to host output
        for i in range(TEST_SIZE):
            host_gpu_output[i] = host_out_buf.unsafe_ptr()[i]

    # Compare CPU vs GPU results
    var max_error = Scalar[dtype](0)
    for i in range(TEST_SIZE):
        var error = abs(host_cpu_output[i] - host_gpu_output[i])
        if error > max_error:
            max_error = error

    var speedup = cpu_time / gpu_time
    print("  CPU time: " + String(cpu_time) + " seconds")
    print("  GPU time: " + String(gpu_time) + " seconds")
    print("  GPU speedup: " + String(speedup) + "x")
    print("  Maximum CPU vs GPU error: " + String(max_error))
    assert_true(max_error < TOLERANCE, "CPU vs GPU consistency check failed")

    # Cleanup
    host_input.free()
    host_cpu_output.free()
    host_gpu_output.free()

    print("✓ CPU vs GPU consistency tests passed")

def test_new_gelu_edge_cases():
    """Test NewGELU with edge cases and boundary conditions."""
    print("Testing NewGELU edge cases...")

    # Test edge cases with individual values
    var test_vals = List[Float64]()
    test_vals.append(0.0)
    test_vals.append(1e-8)
    test_vals.append(-1e-8)
    test_vals.append(10.0)
    test_vals.append(-10.0)
    test_vals.append(1e-5)
    test_vals.append(-1e-5)

    for i in range(len(test_vals)):
        var val = Scalar[dtype](test_vals[i])
        var result = new_gelu_computation[dtype](val)

        print("  NewGELU(" + String(val) + ") = " + String(result))

        # Basic sanity checks
        if val > 0:
            assert_true(result > 0, "Positive input should give positive output")
        elif val < 0:
            # For NewGELU, small negative values give negative outputs,
            # but large negative values approach 0
            if abs(val) < 1.0:
                assert_true(result < 0, "Small negative input should give negative output")
            else:
                # For large negative inputs, NewGELU approaches 0
                assert_true(abs(result) < 1e-5, "Large negative input should approach 0")
        else:
            assert_almost_equal[dtype](result, Scalar[dtype](0.0), rtol=1e-10, atol=1e-10)

        # Check for NaN or infinity
        assert_false(math.isnan(result), "Result should not be NaN")
        assert_false(math.isinf(result), "Result should not be infinite")

    print("✓ Edge cases tests passed")

def main():
    """Run all NewGELU tests."""
    var separator = "=" * 60
    print(separator)
    print("NewGELU Implementation Test Suite")
    print(separator)

    test_new_gelu_single_values()
    test_new_gelu_mathematical_properties()
    test_new_gelu_cpu_implementation()
    test_new_gelu_gpu_implementation()
    test_new_gelu_cpu_vs_gpu()
    test_new_gelu_edge_cases()

    print("\n" + separator)
    print("🎉 All NewGELU tests passed successfully! 🎉")
    print(separator)
