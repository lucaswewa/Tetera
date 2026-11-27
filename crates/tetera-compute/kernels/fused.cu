// The fused ISP kernel (P1.06b, plan D7, ADR-0011): a raw frame to an 8-bit preview in one
// pass. For each output pixel it walks its resize window; at each tap it demosaics the pixel
// from sites it reduces, dark-subtracts and flat-fields on the fly; then it applies colour once.
//
// Every step is the same integer arithmetic as the unfused chain (tetera-imaging's `isp`), in
// the same order, so the two agree exactly: binning's rounded mean, the dark's clamp at zero,
// the flat field's 1/4096ths, the demosaic's rounded and clamped sums (`demosaic.cuh`), the
// resize's 64-bit sums of 1/16384ths rounded once, and colour's matrix and tables.

#include "demosaic.cuh"

// Must match `FusedParams` in tetera-compute's `fused.rs`, field for field.
struct FusedParams {
    int raw_width;              // the raw frame's sites in a row
    int width, height;          // the reduced mosaic's size
    int crop_x, crop_y;         // where a crop starts in the raw frame; 0 otherwise
    int factor;                 // the binning factor, 1 for none
    int ox, oy;                 // the reduced mosaic's pattern, as its offset in an RGGB tile
    int maxv;                   // the reduced mosaic's largest value
    int mhc;                    // 1 for Malvar-He-Cutler, 0 for bilinear
    int has_dark, has_flat;     // whether to subtract the dark, and apply the flat field
    int out_width, out_height;  // the preview's size
    int x_taps, y_taps;         // the resize's window across and down
};

// The reduced, corrected mosaic, computed site by site from the raw frame.
struct Corrected {
    const unsigned short* raw;
    const unsigned short* dark;
    const unsigned short* gains;
    FusedParams p;

    __device__ __forceinline__ int at(int x, int y) const
    {
        x = mirror(x, p.width);
        y = mirror(y, p.height);
        unsigned int v;
        if (p.factor > 1) {
            int dx = x & 1, dy = y & 1, tx = (x >> 1) * p.factor, ty = (y >> 1) * p.factor;
            unsigned int sum = 0;
            for (int j = 0; j < p.factor; ++j) {
                const unsigned short* row = raw + (long long)(2 * (ty + j) + dy) * p.raw_width + 2 * tx + dx;
                for (int i = 0; i < p.factor; ++i) {
                    sum += row[2 * i];
                }
            }
            unsigned int n = p.factor * p.factor;
            v = (sum + n / 2) / n;
        } else {
            v = raw[(long long)(y + p.crop_y) * p.raw_width + x + p.crop_x];
        }
        long long i = (long long)y * p.width + x;
        if (p.has_dark) {
            int d = (int)v - (int)dark[i];
            v = d > 0 ? d : 0;
        }
        if (p.has_flat) {
            unsigned int f = (v * gains[i] + 2048u) >> 12;
            v = f < (unsigned int)p.maxv ? f : (unsigned int)p.maxv;
        }
        return (int)v;
    }
};

extern "C" __global__ void fused_isp(unsigned char* out, const unsigned short* raw,
                                     const unsigned short* dark, const unsigned short* gains,
                                     const int* x_start, const unsigned short* x_weights,
                                     const int* y_start, const unsigned short* y_weights,
                                     const int* matrix, const unsigned char* tables,
                                     FusedParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.out_width || y >= p.out_height) {
        return;
    }
    Corrected m = {raw, dark, gains, p};
    const unsigned short* wx = x_weights + (long long)x * p.x_taps;
    const unsigned short* wy = y_weights + (long long)y * p.y_taps;
    int xs = x_start[x], ys = y_start[y];
    unsigned long long sums[3] = {0, 0, 0};
    for (int a = 0; a < p.y_taps; ++a) {
        if (wy[a] == 0) {
            continue;
        }
        unsigned long long across[3] = {0, 0, 0};
        for (int b = 0; b < p.x_taps; ++b) {
            if (wx[b] == 0) {
                continue;
            }
            unsigned short rgb[3];
            demosaic_pixel<unsigned short>(m, p.mhc != 0, xs + b, ys + a, p.ox, p.oy, p.maxv, rgb);
            for (int k = 0; k < 3; ++k) {
                across[k] += (unsigned long long)wx[b] * rgb[k];
            }
        }
        for (int k = 0; k < 3; ++k) {
            sums[k] += across[k] * wy[a];
        }
    }
    long long resized[3];
    for (int k = 0; k < 3; ++k) {
        resized[k] = (long long)((sums[k] + (1ull << 27)) >> 28);
    }
    for (int c = 0; c < 3; ++c) {
        long long sum = matrix[c * 3] * resized[0] + matrix[c * 3 + 1] * resized[1] + matrix[c * 3 + 2] * resized[2];
        long long v = (sum + 8192) >> 14;
        v = v < 0 ? 0 : (v > 65535 ? 65535 : v);
        out[((long long)y * p.out_width + x) * 3 + c] = tables[c * 65536 + v];
    }
}
