
import compiler_internal as compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
    foreach,
)
from tensor import ManagedTensorSlice
from tensor_internal.managed_tensor_slice import (
  _FusedOutputTensor as FusedOutputTensor,
  _FusedInputTensor as FusedInputTensor,
)
from sys import size_of
from utils.index import IndexList, Index

alias benching = True

@always_inline
fn gbps[
    func: fn () raises capturing -> None
](ctx: DeviceContextPtr, input: ManagedTensorSlice[rank=3]) raises -> Float64:
    var device = ctx.get_device_context()

    var size = size_of[input.dtype]()
    var numel = input.size()

    alias num_iters = 100

    var ns = Float64(device.execution_time[func](num_iters))
    var ms = (ns / 1.0e6) / num_iters

    return (Float64((4.0 / 3.0) * numel * size) * 1.0e-9) / (ms * 1.0e-3)

@compiler.register("grayscale")
struct Grayscale:
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        img_out: FusedOutputTensor[dtype = dtype, rank=2],
        img_in: InputTensor[dtype = dtype, rank=3],
        ctx: DeviceContextPtr,
    ) raises capturing:

        @parameter
        @always_inline
        fn color_to_grayscale[
            width: Int
        ](idx: IndexList[img_out.rank]) capturing -> SIMD[dtype, width]:
            var row, col = idx[0], idx[1]

            var r = img_in.load[width](Index(0, row, col))
            var g = img_in.load[width](Index(1, row, col))
            var b = img_in.load[width](Index(2, row, col))
            return 0.21 * r + 0.71 * g + 0.07 * b

        @always_inline
        fn doit() raises capturing:
            foreach[color_to_grayscale, target=target](img_out, ctx)

        @parameter
        if benching:
            var dt = gbps[doit](ctx, img_in)
            print("shape =", img_out.shape(), "gbps =", dt, "GB/s")
        else:
            doit()
