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

// Both stains of the grid's level pixel (c, r), held within the grid, from its tile: one
// 32-bit load of the two halves. (off_x, off_y) is the grid's first pixel within its first
// tile, so the tile and the pixel within it come by shifts and masks.
__device__ __forceinline__ void texel(
    const unsigned long long* tiles, int columns, int off_x, int off_y, int gw, int gh,
    int c, int r, float* h, float* e)
{
    c = c < 0 ? 0 : (c > gw - 1 ? gw - 1 : c);
    r = r < 0 ? 0 : (r > gh - 1 ? gh - 1 : r);
    int lx = off_x + c, ly = off_y + r;
    const unsigned int* tile = (const unsigned int*)tiles[(ly >> 8) * columns + (lx >> 8)];
    unsigned int pair = tile[((ly & 255) << 8) + (lx & 255)];
    *h = from_half((unsigned short)(pair & 0xffffu));
    *e = from_half((unsigned short)(pair >> 16));
}

// Each sensor pixel's stains, bilinear from the grid of tiles, averaged along its row's path
// during its exposure. Pixel (col, row) starts at (ax + col·bx + row·cx, ay + col·by +
// row·cy) on the grid, in its pixels, the map worked out on the host in double precision,
// the rolling shutter's delay in its row terms; the path's samples are single-precision steps
// from there.
extern "C" __global__ void field_resample(
    float* out_h, float* out_e,
    const unsigned long long* tiles, int columns, int off_x, int off_y, int gw, int gh,
    double ax, double bx, double cx, double ay, double by, double cy,
    float step_x, float step_y, int width, int height, int samples)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }
    double gx = ax + (double)col * bx + (double)row * cx;
    double gy = ay + (double)col * by + (double)row * cy;
    double fbx = floor(gx), fby = floor(gy);
    int ibx = (int)fbx, iby = (int)fby;
    float rx = (float)(gx - fbx), ry = (float)(gy - fby);
    float sum_h = 0.0f, sum_e = 0.0f;
    for (int k = 0; k < samples; k++) {
        float fx = rx + ((float)k + 0.5f) * step_x;
        float fy = ry + ((float)k + 0.5f) * step_y;
        float flx = floorf(fx), fly = floorf(fy);
        int i = ibx + (int)flx, j = iby + (int)fly;
        float tx = fx - flx, ty = fy - fly;
        float ah, ae, bh, be, ch, ce, dh, de;
        texel(tiles, columns, off_x, off_y, gw, gh, i, j, &ah, &ae);
        texel(tiles, columns, off_x, off_y, gw, gh, i + 1, j, &bh, &be);
        texel(tiles, columns, off_x, off_y, gw, gh, i, j + 1, &ch, &ce);
        texel(tiles, columns, off_x, off_y, gw, gh, i + 1, j + 1, &dh, &de);
        float top = ah + tx * (bh - ah);
        float bottom = ch + tx * (dh - ch);
        sum_h += top + ty * (bottom - top);
        top = ae + tx * (be - ae);
        bottom = ce + tx * (de - ce);
        sum_e += top + ty * (bottom - top);
    }
    out_h[row * width + col] = sum_h / (float)samples;
    out_e[row * width + col] = sum_e / (float)samples;
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
