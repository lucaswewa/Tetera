// The field the sensor sees (P1.12a, ADR-0016): the GPU twin of `tetera-sim`'s `optics`.
// A depth group's tiles, resampled into the sensor's pixels; blurred along rows, then along
// columns into the sum of the groups; then lit. The same order of operations as the CPU.

// A half's bits as a float, by PTX, so NVRTC needs no header.
__device__ float from_half(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// One stain of the grid's level pixel (gx, gy), held within the grid, from its tile.
__device__ float texel(
    const unsigned long long* tiles, int c0, int r0, int columns,
    long long gi0, long long gj0, int gw, int gh, long long cx, long long cy, int stain)
{
    long long c = cx < 0 ? 0 : (cx > gw - 1 ? gw - 1 : cx);
    long long r = cy < 0 ? 0 : (cy > gh - 1 ? gh - 1 : cy);
    long long gx = gi0 + c, gy = gj0 + r;
    long long tc = (gx >= 0 ? gx / 256 : -((-gx + 255) / 256)) - c0;
    long long tr = (gy >= 0 ? gy / 256 : -((-gy + 255) / 256)) - r0;
    long long ix = gx - (tc + c0) * 256, iy = gy - (tr + r0) * 256;
    const unsigned short* tile = (const unsigned short*)tiles[tr * columns + tc];
    return from_half(tile[2 * (iy * 256 + ix) + stain]);
}

// Each sensor pixel's stains, bilinear from the grid of tiles.
extern "C" __global__ void field_resample(
    float* out_h, float* out_e,
    const unsigned long long* tiles, int c0, int r0, int columns,
    long long gi0, long long gj0, int gw, int gh, double level_pixel,
    double centre_x, double centre_y, double pixel, double cos_a, double sin_a,
    int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }
    double u = ((double)col + 0.5 - (double)width / 2.0) * pixel;
    double v = ((double)row + 0.5 - (double)height / 2.0) * pixel;
    double x = centre_x + u * cos_a - v * sin_a;
    double y = centre_y + u * sin_a + v * cos_a;
    double fx = (x / level_pixel - (double)gi0) - 0.5;
    double fy = (y / level_pixel - (double)gj0) - 0.5;
    double ix = floor(fx), iy = floor(fy);
    float tx = (float)(fx - ix), ty = (float)(fy - iy);
    long long i = (long long)ix, j = (long long)iy;
    float out[2];
    for (int s = 0; s < 2; s++) {
        float a = texel(tiles, c0, r0, columns, gi0, gj0, gw, gh, i, j, s);
        float b = texel(tiles, c0, r0, columns, gi0, gj0, gw, gh, i + 1, j, s);
        float c = texel(tiles, c0, r0, columns, gi0, gj0, gw, gh, i, j + 1, s);
        float d = texel(tiles, c0, r0, columns, gi0, gj0, gw, gh, i + 1, j + 1, s);
        float top = a + tx * (b - a);
        float bottom = c + tx * (d - c);
        out[s] = top + ty * (bottom - top);
    }
    out_h[row * width + col] = out[0];
    out_e[row * width + col] = out[1];
}

// A blur along rows: out = in ⊛ weights, held at the edges.
extern "C" __global__ void field_blur_rows(
    float* out_h, float* out_e, const float* in_h, const float* in_e,
    int width, int height, const float* weights, int radius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }
    float h = 0.0f, e = 0.0f;
    for (int i = 0; i <= 2 * radius; i++) {
        int c = col + i - radius;
        c = c < 0 ? 0 : (c > width - 1 ? width - 1 : c);
        h += weights[i] * in_h[row * width + c];
        e += weights[i] * in_e[row * width + c];
    }
    out_h[row * width + col] = h;
    out_e[row * width + col] = e;
}

// A blur along columns, put into the sum of the groups: written for the first group, added
// for each later one.
extern "C" __global__ void field_blur_cols(
    float* sum_h, float* sum_e, const float* in_h, const float* in_e,
    int width, int height, const float* weights, int radius, int accumulate)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }
    float h = 0.0f, e = 0.0f;
    for (int i = 0; i <= 2 * radius; i++) {
        int r = row + i - radius;
        r = r < 0 ? 0 : (r > height - 1 ? height - 1 : r);
        h += weights[i] * in_h[r * width + col];
        e += weights[i] * in_e[r * width + col];
    }
    if (accumulate) {
        sum_h[row * width + col] += h;
        sum_e[row * width + col] += e;
    } else {
        sum_h[row * width + col] = h;
        sum_e[row * width + col] = e;
    }
}

// The same blur along rows, a block's row of 256 and its halo first read into shared memory.
// Launched with 256 threads a block, a block a row segment, and 2 · (256 + 2 · radius) floats
// of shared memory. It sums in the same order as the global one.
extern "C" __global__ void field_blur_rows_shared(
    float* out_h, float* out_e, const float* in_h, const float* in_e,
    int width, int height, const float* weights, int radius)
{
    extern __shared__ float shared[];
    int span = 256 + 2 * radius;
    float* sh = shared;
    float* se = shared + span;
    int row = blockIdx.y, x0 = blockIdx.x * 256;
    for (int i = threadIdx.x; i < span; i += 256) {
        int c = x0 + i - radius;
        c = c < 0 ? 0 : (c > width - 1 ? width - 1 : c);
        sh[i] = in_h[row * width + c];
        se[i] = in_e[row * width + c];
    }
    __syncthreads();
    int col = x0 + threadIdx.x;
    if (col >= width || row >= height) {
        return;
    }
    float h = 0.0f, e = 0.0f;
    for (int i = 0; i <= 2 * radius; i++) {
        h += weights[i] * sh[threadIdx.x + i];
        e += weights[i] * se[threadIdx.x + i];
    }
    out_h[row * width + col] = h;
    out_e[row * width + col] = e;
}

// The same blur along columns, a block's 32 × 8 and its halo first read into shared memory.
// Launched with 32 × 8 threads a block and 2 · 32 · (8 + 2 · radius) floats of shared memory.
extern "C" __global__ void field_blur_cols_shared(
    float* sum_h, float* sum_e, const float* in_h, const float* in_e,
    int width, int height, const float* weights, int radius, int accumulate)
{
    extern __shared__ float shared[];
    int span = 8 + 2 * radius;
    float* sh = shared;
    float* se = shared + span * 32;
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * 32 + tx, row0 = blockIdx.y * 8;
    int c = col > width - 1 ? width - 1 : col;
    for (int i = ty; i < span; i += 8) {
        int r = row0 + i - radius;
        r = r < 0 ? 0 : (r > height - 1 ? height - 1 : r);
        sh[i * 32 + tx] = in_h[r * width + c];
        se[i * 32 + tx] = in_e[r * width + c];
    }
    __syncthreads();
    int row = row0 + ty;
    if (col >= width || row >= height) {
        return;
    }
    float h = 0.0f, e = 0.0f;
    for (int i = 0; i <= 2 * radius; i++) {
        h += weights[i] * sh[(ty + i) * 32 + tx];
        e += weights[i] * se[(ty + i) * 32 + tx];
    }
    if (accumulate) {
        sum_h[row * width + col] += h;
        sum_e[row * width + col] += e;
    } else {
        sum_h[row * width + col] = h;
        sum_e[row * width + col] = e;
    }
}

// Each pixel lit: irradiance in red, green and blue, three to a pixel.
extern "C" __global__ void field_light(
    float* rgb, const float* sum_h, const float* sum_e, int width, int height,
    float intensity, float unevenness, float vignetting, float wr, float wg, float wb)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }
    float u = ((float)col + 0.5f) / (float)width * 2.0f - 1.0f;
    float v = ((float)row + 0.5f) / (float)height * 2.0f - 1.0f;
    float uneven = 1.0f + unevenness * (0.6f * u - 0.3f * v + 0.1f * u * v);
    float vignette = 1.0f - vignetting * (u * u + v * v) / 2.0f;
    float k = intensity * uneven * vignette;
    float h = sum_h[row * width + col], e = sum_e[row * width + col];
    float warmth[3] = {wr, wg, wb};
    const float hv[3] = {0.650f, 0.704f, 0.286f};
    const float ev[3] = {0.072f, 0.990f, 0.105f};
    for (int c = 0; c < 3; c++) {
        float od = h * hv[c] + e * ev[c];
        rgb[3 * (row * width + col) + c] = k * warmth[c] * powf(10.0f, -od);
    }
}
