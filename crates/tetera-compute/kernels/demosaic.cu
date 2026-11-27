// Demosaicking (P1.05a): bilinear, and Malvar, He and Cutler's gradient-corrected linear
// interpolation (ICASSP 2004), from a 16-bit Bayer mosaic to interleaved RGB in 16 bits or
// floats. The CPU twins are in tetera-imaging's `demosaic` module, and compute the same sums.
//
// Every value is an integer sum of weighted sites over a power of two: 4 for bilinear, 16 for
// Malvar-He-Cutler. In 16 bits it is rounded and clamped; as a float, divided exactly. So the
// twins agree bit for bit. Beyond the edges the mosaic is mirrored without repeating the edge
// (-1 reads 1), which keeps every site's colour.

__device__ __forceinline__ int demosaic_channel(int x, int y, int ox, int oy)
{
    return ((y + oy) & 1) * 2 + ((x + ox) & 1);
}

__device__ __forceinline__ int mirror(int i, int n)
{
    if (i < 0) {
        i = -i;
    }
    if (i >= n) {
        i = 2 * (n - 1) - i;
    }
    return i;
}

struct Mosaic {
    const unsigned short* in;
    int width, height;
    __device__ __forceinline__ int at(int x, int y) const
    {
        return in[mirror(y, height) * width + mirror(x, width)];
    }
};

template <typename T> __device__ __forceinline__ T demosaic_out(int sum, int shift, int maxv);

template <>
__device__ __forceinline__ unsigned short demosaic_out<unsigned short>(int sum, int shift, int maxv)
{
    if (sum < 0) {
        return 0;
    }
    return (unsigned short)min((sum + (1 << (shift - 1))) >> shift, maxv);
}

template <> __device__ __forceinline__ float demosaic_out<float>(int sum, int shift, int maxv)
{
    return (float)sum / (float)(1 << shift);
}

// Bilinear, in quarters: the site itself, or the mean of its two or four nearest of a colour.
__device__ __forceinline__ void bilinear_sums(const Mosaic& m, int x, int y, int c, int* rgb)
{
    int s = 4 * m.at(x, y);
    int cross = m.at(x - 1, y) + m.at(x + 1, y) + m.at(x, y - 1) + m.at(x, y + 1);
    int diagonal = m.at(x - 1, y - 1) + m.at(x + 1, y - 1) + m.at(x - 1, y + 1) + m.at(x + 1, y + 1);
    int across = 2 * (m.at(x - 1, y) + m.at(x + 1, y));
    int down = 2 * (m.at(x, y - 1) + m.at(x, y + 1));
    switch (c) {
    case 0: rgb[0] = s; rgb[1] = cross; rgb[2] = diagonal; break;
    case 3: rgb[0] = diagonal; rgb[1] = cross; rgb[2] = s; break;
    case 1: rgb[0] = across; rgb[1] = s; rgb[2] = down; break;
    default: rgb[0] = down; rgb[1] = s; rgb[2] = across; break;
    }
}

// Malvar-He-Cutler, in sixteenths: the paper's filters, doubled to whole numbers.
__device__ __forceinline__ void mhc_sums(const Mosaic& m, int x, int y, int c, int* rgb)
{
    int s = m.at(x, y);
    int h1 = m.at(x - 1, y) + m.at(x + 1, y), v1 = m.at(x, y - 1) + m.at(x, y + 1);
    int h2 = m.at(x - 2, y) + m.at(x + 2, y), v2 = m.at(x, y - 2) + m.at(x, y + 2);
    int d1 = m.at(x - 1, y - 1) + m.at(x + 1, y - 1) + m.at(x - 1, y + 1) + m.at(x + 1, y + 1);
    // Green at red or blue.
    int green = 8 * s + 4 * (h1 + v1) - 2 * (h2 + v2);
    // Red or blue at the other of the two.
    int opposite = 12 * s + 4 * d1 - 3 * (h2 + v2);
    // Red or blue at green, from the row's neighbours, and from the column's.
    int along_row = 10 * s + 8 * h1 - 2 * (h2 + d1) + v2;
    int along_column = 10 * s + 8 * v1 - 2 * (v2 + d1) + h2;
    switch (c) {
    case 0: rgb[0] = 16 * s; rgb[1] = green; rgb[2] = opposite; break;
    case 3: rgb[0] = opposite; rgb[1] = green; rgb[2] = 16 * s; break;
    case 1: rgb[0] = along_row; rgb[1] = 16 * s; rgb[2] = along_column; break;
    default: rgb[0] = along_column; rgb[1] = 16 * s; rgb[2] = along_row; break;
    }
}

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
    int c = demosaic_channel(x, y, ox, oy);
    int rgb[3];
    if (MHC) {
        mhc_sums(m, x, y, c, rgb);
    } else {
        bilinear_sums(m, x, y, c, rgb);
    }
    int shift = MHC ? 4 : 2;
    int i = (y * width + x) * 3;
    for (int k = 0; k < 3; ++k) {
        out[i + k] = demosaic_out<T>(rgb[k], shift, maxv);
    }
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
