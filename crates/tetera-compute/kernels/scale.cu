// Scale and offset 16-bit samples to 32-bit floats: out[i] = in[i] * scale + offset, fused.
// The first kernel, which proves the path every later one takes (P1.04b, ADR-0010). Its CPU
// twin, `ops::scale_offset_cpu`, fuses the same way, so the two agree bit for bit.

extern "C" __global__ void scale_offset_u16_f32(
    float* out,
    const unsigned short* in,
    float scale,
    float offset,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaf((float)in[i], scale, offset);
    }
}
