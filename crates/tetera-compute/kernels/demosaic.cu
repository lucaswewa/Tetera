// Demosaicking (P1.05a): bilinear, and Malvar, He and Cutler's gradient-corrected linear
// interpolation (ICASSP 2004), from a 16-bit Bayer mosaic to interleaved RGB in 16 bits or
// floats. The CPU twins are in tetera-imaging's `demosaic` module, and compute the same sums;
// the sums themselves are in `demosaic.cuh`, which the fused kernel shares.
//
// Beyond the edges the mosaic is mirrored without repeating the edge (-1 reads 1), which keeps
// every site's colour.

#include "demosaic.cuh"

// A mosaic in device memory, read mirrored.
struct Mosaic {
    const unsigned short* in;
    int width, height;
    __device__ __forceinline__ int at(int x, int y) const
    {
        return in[mirror(y, height) * width + mirror(x, width)];
    }
};

template <typename T, bool MHC>
__device__ __forceinline__ void demosaic(T* out, const unsigned short* in, int width, int height,
                                         int ox, int oy, int maxv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    Mosaic m = {in, width, height};
    demosaic_pixel<T>(m, MHC, x, y, ox, oy, maxv, out + (y * width + x) * 3);
}

extern "C" __global__ void demosaic_bilinear_u16(unsigned short* out, const unsigned short* in,
                                                 int width, int height, int ox, int oy, int maxv)
{
    demosaic<unsigned short, false>(out, in, width, height, ox, oy, maxv);
}

extern "C" __global__ void demosaic_bilinear_f32(float* out, const unsigned short* in, int width,
                                                 int height, int ox, int oy, int maxv)
{
    demosaic<float, false>(out, in, width, height, ox, oy, maxv);
}

extern "C" __global__ void demosaic_mhc_u16(unsigned short* out, const unsigned short* in,
                                            int width, int height, int ox, int oy, int maxv)
{
    demosaic<unsigned short, true>(out, in, width, height, ox, oy, maxv);
}

extern "C" __global__ void demosaic_mhc_f32(float* out, const unsigned short* in, int width,
                                            int height, int ox, int oy, int maxv)
{
    demosaic<float, true>(out, in, width, height, ox, oy, maxv);
}
