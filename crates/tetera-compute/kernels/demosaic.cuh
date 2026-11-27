// Demosaicking's device code (P1.05a), shared by `demosaic.cu` and the fused kernel
// (`fused.cu`, P1.06b): the sums of bilinear and Malvar-He-Cutler over any source of sites.
// `build.rs` inlines this file where it is included, for NVRTC.
//
// A source is a type with `int at(int x, int y) const`, mirroring beyond its edges. Every value
// is an integer sum of weighted sites over a power of two: 4 for bilinear, 16 for
// Malvar-He-Cutler. In 16 bits it is rounded and clamped; as a float, divided exactly.

#pragma once

// A site's channel, 0 red, 1 green on a red row, 2 green on a blue row, 3 blue, for a pattern
// whose first site is (ox, oy) in an RGGB tile.
__device__ __forceinline__ int demosaic_channel(int x, int y, int ox, int oy)
{
    return ((y + oy) & 1) * 2 + ((x + ox) & 1);
}

// Index i mirrored into 0..n without repeating the edge: −1 reads 1, which keeps the colour.
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
template <typename S>
__device__ __forceinline__ void bilinear_sums(const S& m, int x, int y, int c, int* rgb)
{
    int s = 4 * m.at(x, y);
    int left = m.at(x - 1, y), right = m.at(x + 1, y), up = m.at(x, y - 1), down = m.at(x, y + 1);
    int cross = left + right + up + down;
    int diagonal = m.at(x - 1, y - 1) + m.at(x + 1, y - 1) + m.at(x - 1, y + 1) + m.at(x + 1, y + 1);
    int across = 2 * (left + right);
    int along = 2 * (up + down);
    switch (c) {
    case 0: rgb[0] = s; rgb[1] = cross; rgb[2] = diagonal; break;
    case 3: rgb[0] = diagonal; rgb[1] = cross; rgb[2] = s; break;
    case 1: rgb[0] = across; rgb[1] = s; rgb[2] = along; break;
    default: rgb[0] = along; rgb[1] = s; rgb[2] = across; break;
    }
}

// Malvar-He-Cutler, in sixteenths: the paper's filters, doubled to whole numbers.
template <typename S>
__device__ __forceinline__ void mhc_sums(const S& m, int x, int y, int c, int* rgb)
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

// The pixel at (x, y) of source m, demosaicked by bilinear or Malvar-He-Cutler, into out[0..3].
template <typename T, typename S>
__device__ __forceinline__ void demosaic_pixel(const S& m, bool mhc, int x, int y, int ox, int oy,
                                               int maxv, T* out)
{
    int c = demosaic_channel(x, y, ox, oy);
    int rgb[3];
    if (mhc) {
        mhc_sums(m, x, y, c, rgb);
    } else {
        bilinear_sums(m, x, y, c, rgb);
    }
    int shift = mhc ? 4 : 2;
    for (int k = 0; k < 3; ++k) {
        out[k] = demosaic_out<T>(rgb[k], shift, maxv);
    }
}
