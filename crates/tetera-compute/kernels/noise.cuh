// The integer hash the simulator's kernels share (P1.11b, P1.12b): the twins of
// `tetera-sim`'s `noise::hash` and `noise::unit`. Included by he.cu and sensor.cu; build.rs
// inlines it for NVRTC.

#ifndef TETERA_NOISE_CUH
#define TETERA_NOISE_CUH

typedef unsigned int u32;


__device__ u32 lowbias32(u32 x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ u32 hash3(u32 seed, u32 salt, int x, int y, int z)
{
    u32 h = lowbias32(seed ^ (salt * 0x9e3779b9U));
    h = lowbias32(h ^ (u32)x);
    h = lowbias32(h ^ (u32)y);
    h = lowbias32(h ^ (u32)z);
    return h;
}

__device__ float unit_f(u32 h)
{
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}

#endif
