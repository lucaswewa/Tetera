// The ISP's corrections (P1.06a): dark subtraction and the flat field on the mosaic, and
// colour on RGB. The CPU twins are in tetera-imaging's `correct` and `colour` modules, and
// compute the same integers.

// out = in − dark, clamped at zero.
extern "C" __global__ void subtract_dark_u16(unsigned short* out, const unsigned short* in,
                                             const unsigned short* dark, long long sites)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) {
        return;
    }
    int v = (int)in[i] - (int)dark[i];
    out[i] = (unsigned short)(v > 0 ? v : 0);
}

// out = in × gain, the gain in 1/4096ths, rounded and clamped to maxv.
extern "C" __global__ void apply_flat_u16(unsigned short* out, const unsigned short* in,
                                          const unsigned short* gains, long long sites, int maxv)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sites) {
        return;
    }
    unsigned int v = ((unsigned int)in[i] * gains[i] + 2048u) >> 12;
    out[i] = (unsigned short)(v < (unsigned int)maxv ? v : (unsigned int)maxv);
}

// Colour: each RGB pixel times a 3 × 3 matrix in 1/16384ths (white balance and the CCM in
// one), rounded and clamped to 0..65535, then each channel looked up in its table (levels and
// the transfer curve) to 8 bits.
extern "C" __global__ void colour_u16_u8(unsigned char* out, const unsigned short* in,
                                         long long values, const int* matrix,
                                         const unsigned char* tables)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= values / 3) {
        return;
    }
    const unsigned short* rgb = in + p * 3;
    for (int c = 0; c < 3; ++c) {
        long long sum = (long long)matrix[c * 3] * rgb[0] + (long long)matrix[c * 3 + 1] * rgb[1]
                        + (long long)matrix[c * 3 + 2] * rgb[2];
        long long v = (sum + 8192) >> 14;
        v = v < 0 ? 0 : (v > 65535 ? 65535 : v);
        out[p * 3 + c] = tables[c * 65536 + v];
    }
}
