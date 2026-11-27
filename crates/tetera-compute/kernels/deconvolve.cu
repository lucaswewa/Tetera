// Richardson–Lucy's element-wise steps (P1.07c), between cuFFT's transforms. The CPU twins are
// in tetera-imaging's `deconvolve` module. Each product, sum and quotient is rounded on its own,
// through the `_rn` intrinsics, as Rust rounds them: nvcc would otherwise fuse a multiply and
// an add into one, rounded once, and the twins would differ in the last bit.

// Index i reflected into 0..n about its ends without repeating them, as often as it takes.
__device__ long long reflect(long long i, long long n)
{
    if (n == 1) {
        return 0;
    }
    long long period = 2 * (n - 1);
    i %= period;
    if (i < 0) {
        i += period;
    }
    return i < n ? i : period - i;
}

// The image, w × h × d, mirrored into the padded volume, pw × ph × pd, its first pixel at
// (ox, oy, oz).
extern "C" __global__ void rl_pad_f32(float* out, const float* image, int w, int h, int d,
                                      int pw, int ph, int pd, int ox, int oy, int oz)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)pw * ph * pd) {
        return;
    }
    long long px = i % pw;
    long long py = i / pw % ph;
    long long pz = i / ((long long)pw * ph);
    long long x = reflect(px - ox, w);
    long long y = reflect(py - oy, h);
    long long z = reflect(pz - oz, d);
    out[i] = image[(z * h + y) * w + x];
}

// The image's part of the padded volume.
extern "C" __global__ void rl_crop_f32(float* out, const float* padded, int w, int h, int d,
                                       int pw, int ph, int ox, int oy, int oz)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)w * h * d) {
        return;
    }
    long long x = i % w;
    long long y = i / w % h;
    long long z = i / ((long long)w * h);
    out[i] = padded[((z + oz) * ph + y + oy) * pw + x + ox];
}

// Real values as complex ones, their imaginary parts 0.
extern "C" __global__ void rl_to_complex(float2* out, const float* values, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = make_float2(values[i], 0.0f);
    }
}

// Each value times the PSF's transform scaled by `scale`, or by its conjugate, in place.
extern "C" __global__ void rl_multiply(float2* buffer, const float2* otf, float scale,
                                       int conjugate, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float hr = __fmul_rn(otf[i].x, scale);
    float hi = __fmul_rn(otf[i].y, scale);
    if (conjugate) {
        hi = -hi;
    }
    float2 b = buffer[i];
    buffer[i] = make_float2(__fsub_rn(__fmul_rn(b.x, hr), __fmul_rn(b.y, hi)),
                            __fadd_rn(__fmul_rn(b.y, hr), __fmul_rn(b.x, hi)));
}

// The ratio d / (u ⊛ h), in place: 1 where the blurred value isn't above `floor`.
extern "C" __global__ void rl_ratio(float2* blurred, const float* data, float floor, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float b = blurred[i].x;
    blurred[i] = make_float2(b > floor ? __fdiv_rn(data[i], b) : 1.0f, 0.0f);
}

// The estimate times the correction's real part, held at 0 or more.
extern "C" __global__ void rl_update(float* estimate, const float2* correction, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        estimate[i] = fmaxf(__fmul_rn(estimate[i], correction[i].x), 0.0f);
    }
}

// The same, over total variation's denominators.
extern "C" __global__ void rl_update_tv(float* estimate, const float2* correction,
                                        const float* denominators, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        estimate[i] = fmaxf(__fdiv_rn(__fmul_rn(estimate[i], correction[i].x), denominators[i]),
                            0.0f);
    }
}

// The unit gradient of u at (x, y, z), by forward differences, 0 across the far edges; its
// length held at `epsilon` or more.
__device__ float3 unit_gradient(const float* u, int nx, int ny, int nz, int x, int y, int z,
                                float epsilon)
{
    long long i = ((long long)z * ny + y) * nx + x;
    float c = u[i];
    float gx = x + 1 < nx ? __fsub_rn(u[i + 1], c) : 0.0f;
    float gy = y + 1 < ny ? __fsub_rn(u[i + nx], c) : 0.0f;
    float gz = z + 1 < nz ? __fsub_rn(u[i + (long long)nx * ny], c) : 0.0f;
    float squares = __fadd_rn(__fadd_rn(__fmul_rn(gx, gx), __fmul_rn(gy, gy)), __fmul_rn(gz, gz));
    float norm = fmaxf(__fsqrt_rn(squares), epsilon);
    return make_float3(__fdiv_rn(gx, norm), __fdiv_rn(gy, norm), __fdiv_rn(gz, norm));
}

// Total variation's denominators, 1 − λ div(∇u / |∇u|), held at 0.1 or more; the divergence
// by backward differences, 0 across the near edges. One thread a point; z is the grid's z.
extern "C" __global__ void rl_total_variation(float* out, const float* u, int nx, int ny, int nz,
                                              float lambda, float epsilon)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }
    float3 n = unit_gradient(u, nx, ny, nz, x, y, z, epsilon);
    float divergence = __fadd_rn(__fadd_rn(n.x, n.y), n.z);
    if (x > 0) {
        divergence = __fsub_rn(divergence, unit_gradient(u, nx, ny, nz, x - 1, y, z, epsilon).x);
    }
    if (y > 0) {
        divergence = __fsub_rn(divergence, unit_gradient(u, nx, ny, nz, x, y - 1, z, epsilon).y);
    }
    if (z > 0) {
        divergence = __fsub_rn(divergence, unit_gradient(u, nx, ny, nz, x, y, z - 1, epsilon).z);
    }
    out[((long long)z * ny + y) * nx + x] = fmaxf(__fsub_rn(1.0f, __fmul_rn(lambda, divergence)), 0.1f);
}
