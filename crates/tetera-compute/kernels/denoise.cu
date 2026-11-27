// Denoising (P1.07a): temporal averaging over a window of frames, and the bilateral filter.
// The CPU twins are in tetera-imaging's `denoise` module, and compute the same integers.

#include "demosaic.cuh"

// One frame into a window: the window's sum gains the frame and, when the window is full, loses
// the oldest frame, which `slot` holds and the new frame replaces. `mean` is the window's mean
// over `n` frames, rounded half up.
extern "C" __global__ void temporal_push_u16(unsigned short* mean, unsigned int* sum,
                                             unsigned short* slot, const unsigned short* frame,
                                             long long sites, int full, int n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) {
        return;
    }
    unsigned int s = sum[i] + frame[i] - (full ? slot[i] : 0u);
    sum[i] = s;
    slot[i] = frame[i];
    mean[i] = (unsigned short)((s + (unsigned int)n / 2) / (unsigned int)n);
}

// The bilateral filter on an image of `channels` interleaved values (1 or 3): each neighbour
// within `radius` weighs its spatial weight times the range weight of its difference from the
// centre, both in 1/32768ths from tables; each channel is the weighted mean, rounded half up.
// For RGB the difference is (|ΔR| + 2|ΔG| + |ΔB|) / 4; for grey, |Δ|.
extern "C" __global__ void bilateral_u16(unsigned short* out, const unsigned short* in, int width,
                                         int height, int channels, int radius,
                                         const unsigned short* spatial,
                                         const unsigned short* range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const unsigned short* centre = in + ((long long)y * width + x) * channels;
    unsigned long long acc[3] = {0, 0, 0};
    unsigned long long total = 0;
    int side = 2 * radius + 1;
    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = mirror(y + dy, height);
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = mirror(x + dx, width);
            const unsigned short* p = in + ((long long)ny * width + nx) * channels;
            unsigned int d;
            if (channels == 3) {
                d = (abs((int)p[0] - (int)centre[0]) + 2 * abs((int)p[1] - (int)centre[1])
                     + abs((int)p[2] - (int)centre[2])) / 4;
            } else {
                d = abs((int)p[0] - (int)centre[0]);
            }
            unsigned long long w = (unsigned long long)spatial[(dy + radius) * side + dx + radius] * range[d];
            total += w;
            for (int c = 0; c < channels; ++c) {
                acc[c] += w * p[c];
            }
        }
    }
    for (int c = 0; c < channels; ++c) {
        out[((long long)y * width + x) * channels + c] = (unsigned short)((acc[c] + total / 2) / total);
    }
}
