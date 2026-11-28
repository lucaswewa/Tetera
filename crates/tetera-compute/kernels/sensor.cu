// The sensor on the GPU (P1.12b, ADR-0016): the twin of tetera-sim's `sensor::pixel`, in its
// order of operations. Each thread makes one pixel's count from the field's irradiance in its
// colour of the RGGB mosaic: Poisson photoelectrons and dark current, Gaussian read noise,
// the well, the gain and the black level, 12 significant bits or 8. The noise comes from the
// hash of the seed, the pixel and the frame, so a frame repeats bit for bit; against the CPU
// it differs only where single-precision functions round a draw across a count's edge.

#include "noise.cuh"

#define SHOT 40u
#define READ 41u
#define POISSON_EXACT 16.0f
#define TAU 6.28318530717958647692f

// The colour of pixel (col, row): 0 red, 1 green, 2 blue.
__device__ int bayer(int col, int row)
{
    int r = row & 1, c = col & 1;
    if (r == 0 && c == 0) return 0;
    if (r == 1 && c == 1) return 2;
    return 1;
}

// A uniform draw in (0, 1].
__device__ float draw(u32 seed, u32 salt, int col, int row, u32 frame_lo, u32 frame_hi, u32 k)
{
    u32 h = hash3(seed, (salt + (k << 8)) ^ frame_hi, col, row, (int)frame_lo);
    return 1.0f - unit_f(h);
}

// A standard normal draw, by Box-Muller.
__device__ float normal(u32 seed, u32 salt, int col, int row, u32 frame_lo, u32 frame_hi)
{
    float u1 = draw(seed, salt, col, row, frame_lo, frame_hi, 0);
    float u2 = draw(seed, salt, col, row, frame_lo, frame_hi, 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(__fmul_rn(TAU, u2));
}

// A Poisson draw: by inversion for a small mean, by a Gaussian above.
__device__ float poisson(u32 seed, int col, int row, u32 frame_lo, u32 frame_hi, float mean)
{
    if (mean <= 0.0f) return 0.0f;
    if (mean > POISSON_EXACT) {
        float g = __fadd_rn(mean, __fmul_rn(sqrtf(mean), normal(seed, SHOT, col, row, frame_lo, frame_hi)));
        return fmaxf(roundf(g), 0.0f);
    }
    float u = draw(seed, SHOT + 1u, col, row, frame_lo, frame_hi, 0);
    float p = expf(-mean);
    float cumulative = p;
    float k = 0.0f;
    while (u > cumulative && k < 64.0f) {
        k += 1.0f;
        p = __fmul_rn(p, __fdiv_rn(mean, k));
        cumulative = __fadd_rn(cumulative, p);
    }
    return k;
}

// The frame: `rgb` the field's irradiance, three to a pixel, photoelectrons a ms at quantum
// efficiency 1 for one 2.4 um pixel; `raw` a count to a pixel, its rows `pitch` counts apart.
extern "C" __global__ void sensor_expose(
    unsigned short* raw, int pitch, const float* rgb, int width, int height,
    unsigned int seed, unsigned int frame_lo, unsigned int frame_hi,
    float qe_r, float qe_g, float qe_b, float full_well, float read_noise, float dark_current,
    float black_level, float electrons_per_count, float ms, float gain, int bits, int binning)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    long long at = (long long)row * width + col;
    int colour = bayer(col, row);
    float b = (float)binning;
    float area = __fmul_rn(b, b);
    float irradiance = __fmul_rn(rgb[3 * at + colour], area);
    float qe = colour == 0 ? qe_r : (colour == 1 ? qe_g : qe_b);
    float light = __fmul_rn(__fmul_rn(irradiance, qe), ms);
    float dark = __fmul_rn(__fmul_rn(__fmul_rn(dark_current, area), ms), 1e-3f);
    float mean = __fadd_rn(light, dark);
    float electrons = __fadd_rn(
        poisson(seed, col, row, frame_lo, frame_hi, mean),
        __fmul_rn(__fmul_rn(read_noise, b), normal(seed, READ, col, row, frame_lo, frame_hi)));
    float well = fminf(electrons, __fmul_rn(full_well, area));
    float counts = roundf(__fadd_rn(black_level, __fdiv_rn(__fmul_rn(well, gain), electrons_per_count)));
    counts = fminf(fmaxf(counts, 0.0f), 4095.0f);
    unsigned short c = (unsigned short)counts;
    raw[(long long)row * pitch + col] = bits == 8 ? (unsigned short)(c >> 4) : c;
}
