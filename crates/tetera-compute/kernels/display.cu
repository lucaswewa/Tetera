// The display transform (P1.05b): black and white levels, then a tone curve, to 8 bits, as one
// lookup table of 65,536 entries per channel, made on the host. The CPU twin is in
// tetera-imaging's `display` module and uses the same tables.

extern "C" __global__ void display_u16_u8(unsigned char* out, const unsigned short* in,
                                          long long values, int channels,
                                          const unsigned char* tables)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= values) {
        return;
    }
    out[i] = tables[(i % channels) * 65536 + in[i]];
}
