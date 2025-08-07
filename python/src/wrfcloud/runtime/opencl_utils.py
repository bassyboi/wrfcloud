"""Utility functions using OpenCL for GPU acceleration."""
from typing import Sequence

import numpy

try:
    import pyopencl as cl  # type: ignore
    _CONTEXT = cl.create_some_context()
    _QUEUE = cl.CommandQueue(_CONTEXT)
    _OPENCL_AVAILABLE = True
except Exception:  # pragma: no cover - OpenCL not always available
    _OPENCL_AVAILABLE = False
    _CONTEXT = None
    _QUEUE = None


def reshape_1d_to_2d(data: Sequence[float], x: int, y: int) -> numpy.ndarray:
    """Reshape a 1D array into a 2D array using GPU if available."""
    array = numpy.asarray(data, dtype=numpy.float32)
    if array.size != x * y:
        raise ValueError('Input data size does not match provided dimensions')

    if not _OPENCL_AVAILABLE:
        return array.reshape((y, x))

    kernel_code = """
    __kernel void reshape(__global const float *in,
                          __global float *out,
                          const unsigned int width) {
        int gid = get_global_id(0);
        int row = gid / width;
        int col = gid % width;
        out[row*width + col] = in[gid];
    }
    """
    program = cl.Program(_CONTEXT, kernel_code).build()
    mf = cl.mem_flags
    in_buffer = cl.Buffer(_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
    out_buffer = cl.Buffer(_CONTEXT, mf.WRITE_ONLY, array.nbytes)
    program.reshape(_QUEUE, array.shape, None, in_buffer, out_buffer, numpy.uint32(x))
    result = numpy.empty_like(array)
    cl.enqueue_copy(_QUEUE, result, out_buffer)
    return result.reshape((y, x))
