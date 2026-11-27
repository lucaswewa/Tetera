// The H&E sample on the GPU (P1.11b, ADR-0015): the CPU twin's functions, tetera-sim's
// `noise`, `he` and `render` modules, in the same order of operations. Positions are split
// into lattice cell and offset in double precision; everything else is single precision.
// Two passes: `he_nuclei` and `he_red_cells` decide each grid cell's nucleus or red cell at
// its own centre; `he_render` then makes each pixel, its stains as half-precision pairs.

#include "noise.cuh"

// Salts, as `he.rs` numbers them.
#define MASK 1u
#define GLANDS 10u
#define FAT 11u
#define ANGLE 12u
#define WAVER 13u
#define CLEFT 14u
#define GLAND_CELLS 15u
#define GLAND_EDGE 16u
#define FAT_CELLS 17u
#define VESSELS 18u
#define NUCLEI 20u
#define NUCLEUS_EDGE 21u
#define CHROMATIN 22u
#define RED_CELLS 23u
#define FOLDS 24u

#define NUCLEUS_CELL 10.0
#define RED_CELL_CELL 7.0
#define RED_CELL_RADIUS 3.6f
#define FIBRE_CELL 40.0
#define LAYER 0.5f
#define PI_F 3.14159265358979f
#define TAU_F 6.28318530717959f

// --- noise.rs ---

__device__ void lattice(double x, double wavelength, int* cell, float* t)
{
    double f = x / wavelength;
    double c = floor(f);
    *cell = (int)(long long)c;
    *t = (float)(f - c);
}

__device__ int plus(int a, int b)
{
    return (int)((u32)a + (u32)b);
}

__device__ float fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp_f(float a, float b, float t)
{
    return a + t * (b - a);
}

// The 12 gradients toward the cube's edge midpoints, without their z, dotted with (dx, dy).
__device__ float gradient(u32 h, float dx, float dy)
{
    switch (h % 12u) {
    case 0: return dx + dy;
    case 1: return -dx + dy;
    case 2: return dx - dy;
    case 3: return -dx - dy;
    case 4: case 6: return dx;
    case 5: case 7: return -dx;
    case 8: case 10: return dy;
    default: return -dy;
    }
}

__device__ float improved(u32 seed, u32 salt, double x, double y, double wavelength)
{
    int ix, iy;
    float tx, ty;
    lattice(x, wavelength, &ix, &tx);
    lattice(y, wavelength, &iy, &ty);
    float u = fade(tx), v = fade(ty);
    float c00 = gradient(hash3(seed, salt, ix, iy, 0), tx, ty);
    float c10 = gradient(hash3(seed, salt, plus(ix, 1), iy, 0), tx - 1.0f, ty);
    float c01 = gradient(hash3(seed, salt, ix, plus(iy, 1), 0), tx, ty - 1.0f);
    float c11 = gradient(hash3(seed, salt, plus(ix, 1), plus(iy, 1), 0), tx - 1.0f, ty - 1.0f);
    return lerp_f(lerp_f(c00, c10, u), lerp_f(c01, c11, u), v);
}

__device__ float fbm(u32 seed, u32 salt, double x, double y, double wavelength, u32 octaves)
{
    float sum = 0.0f, amplitude = 1.0f;
    double w = wavelength;
    for (u32 octave = 0; octave < octaves; octave++) {
        sum += amplitude * improved(seed, salt + octave, x, y, w);
        amplitude *= 0.5f;
        w *= 0.5;
    }
    return sum;
}

struct Cellular {
    float f1, f2;
    u32 hash;
    float ox, oy;
};

__device__ Cellular cellular(u32 seed, u32 salt, double x, double y, double cell)
{
    int ix, iy;
    float tx, ty;
    lattice(x, cell, &ix, &tx);
    lattice(y, cell, &iy, &ty);
    float size = (float)cell;
    Cellular best;
    best.f1 = INFINITY;
    best.f2 = INFINITY;
    best.hash = 0;
    best.ox = 0.0f;
    best.oy = 0.0f;
    for (int cy = -1; cy <= 1; cy++) {
        for (int cx = -1; cx <= 1; cx++) {
            u32 h = hash3(seed, salt, plus(ix, cx), plus(iy, cy), 0);
            float fx = (float)cx + unit_f(h);
            float fy = (float)cy + unit_f(lowbias32(h));
            float dx = (tx - fx) * size, dy = (ty - fy) * size;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < best.f1) {
                best.f2 = best.f1;
                best.f1 = d;
                best.hash = h;
                best.ox = dx;
                best.oy = dy;
            } else if (d < best.f2) {
                best.f2 = d;
            }
        }
    }
    return best;
}

__device__ float smoothstep_f(float a, float b, float x)
{
    float t = fminf(fmaxf((x - a) / (b - a), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// --- he.rs ---

struct Column {
    float h, e, nuclei, vessel, angle;
};

__device__ bool folded(u32 seed, u32 folds, double x, double y)
{
    return folds != 0u && fabsf(improved(seed, FOLDS, x, y, 2000.0)) < 0.03f;
}

__device__ float thickness_at(u32 seed, float thickness, u32 folds, double x, double y)
{
    return folded(seed, folds, x, y) ? 2.0f * thickness : thickness;
}

__device__ float fibre_angle(u32 seed, double x, double y)
{
    return PI_F * improved(seed, ANGLE, x, y, 250.0);
}

__device__ float stripe(u32 seed, double x, double y, double cx, double cy, float waver)
{
    double ox = cx * FIBRE_CELL, oy = cy * FIBRE_CELL;
    float theta = fibre_angle(seed, ox, oy);
    float dx = (float)(x - ox), dy = (float)(y - oy);
    float across = -sinf(theta) * dx + cosf(theta) * dy;
    float bundle = 0.5f + 0.5f * sinf(TAU_F * across / 6.0f + waver);
    float fibril = 0.5f + 0.5f * sinf(TAU_F * across / 1.7f + 2.0f * waver);
    float s = 0.65f * bundle + 0.35f * fibril;
    return s * s * s;
}

__device__ float collagen(u32 seed, double x, double y)
{
    double fx = x / FIBRE_CELL, fy = y / FIBRE_CELL;
    double cx = floor(fx), cy = floor(fy);
    float tx = (float)(fx - cx), ty = (float)(fy - cy);
    float waver = 2.5f * improved(seed, WAVER, x, y, 30.0);
    float top = stripe(seed, x, y, cx, cy, waver) * (1.0f - tx) + stripe(seed, x, y, cx + 1.0, cy, waver) * tx;
    float bottom = stripe(seed, x, y, cx, cy + 1.0, waver) * (1.0f - tx) + stripe(seed, x, y, cx + 1.0, cy + 1.0, waver) * tx;
    float f = top * (1.0f - ty) + bottom * ty;
    float cleft = smoothstep_f(-0.44f, -0.46f, improved(seed, CLEFT, x, y, 40.0));
    float grain = 0.85f + 0.15f * improved(seed, WAVER + 100u, x, y, 0.8);
    return 0.16f * (0.45f + 0.55f * f) * grain * (1.0f - cleft);
}

__device__ Column column(u32 seed, double x, double y)
{
    Column out;
    out.h = 0.0f;
    out.e = 0.0f;
    out.nuclei = 0.0f;
    out.vessel = 0.0f;
    out.angle = 0.0f;
    float tissue = smoothstep_f(-0.152f, -0.148f, fbm(seed, MASK, x, y, 800.0, 4u));
    if (tissue == 0.0f) {
        return out;
    }
    float glands = smoothstep_f(0.175f, 0.225f, improved(seed, GLANDS, x, y, 450.0));
    float fat = smoothstep_f(0.275f, 0.325f, improved(seed, FAT, x, y, 350.0)) * (1.0f - glands);
    float stroma = 1.0f - glands - fat;
    float h = 0.0f, e = 0.0f, epithelium = 0.0f;
    float angle = fibre_angle(seed, x, y);
    if (glands > 0.0f) {
        Cellular g = cellular(seed, GLAND_CELLS, x, y, 180.0);
        float radius = 180.0f * (0.30f + 0.12f * unit_f(g.hash));
        float d = g.f1 * (1.0f + 0.06f * improved(seed, GLAND_EDGE, x, y, 25.0));
        float lumen = 1.0f - smoothstep_f(radius - 17.0f, radius - 15.0f, d);
        float epi = smoothstep_f(radius - 17.0f, radius - 15.0f, d) * (1.0f - smoothstep_f(radius - 1.0f, radius + 1.0f, d));
        epithelium = glands * epi;
        stroma += glands * (1.0f - lumen - epi);
        h = h + 0.02f * epithelium;
        e = e + 0.10f * epithelium;
        if (epi > 0.0f) {
            angle = atan2f(g.oy, g.ox);
        }
    }
    float v = fabsf(improved(seed, VESSELS, x, y, 350.0));
    float vessel = (1.0f - smoothstep_f(0.022f, 0.028f, v)) * (1.0f - glands);
    float wall = smoothstep_f(0.022f, 0.028f, v) * (1.0f - smoothstep_f(0.030f, 0.034f, v)) * (1.0f - glands);
    float open = fmaxf(1.0f - vessel - wall, 0.0f);
    if (stroma > 0.0f) {
        float k = stroma * open;
        h = h + 0.0f * k;
        e = e + collagen(seed, x, y) * k;
    }
    if (fat > 0.0f) {
        Cellular c = cellular(seed, FAT_CELLS, x, y, 70.0);
        float membrane = 1.0f - smoothstep_f(0.9f, 1.5f, c.f2 - c.f1);
        e += 0.12f * membrane * fat * open;
    }
    e += 0.16f * wall;
    float nuclei = (0.30f * stroma + 0.85f * epithelium + 0.06f * fat) * open;
    out.h = h * tissue;
    out.e = e * tissue;
    out.nuclei = nuclei * tissue;
    out.vessel = vessel * tissue;
    out.angle = angle;
    return out;
}

// A nucleus: centre (double2), and depth, a, b, c, angle, darkness, present, in 8 floats.
extern "C" __global__ void he_nuclei(
    double* centres, float* params, int i0, int j0, int columns, int n,
    u32 seed, float thickness, u32 folds)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) {
        return;
    }
    int i = i0 + k % columns, j = j0 + k / columns;
    u32 h = hash3(seed, NUCLEI, i, j, 0);
    float u[7];
    for (u32 q = 0; q < 7u; q++) {
        u[q] = unit_f(hash3(seed, NUCLEI + 100u + q, i, j, 0));
    }
    double cx = ((double)i + 0.5 + 0.35 * (double)(2.0f * u[0] - 1.0f)) * NUCLEUS_CELL;
    double cy = ((double)j + 0.5 + 0.35 * (double)(2.0f * u[1] - 1.0f)) * NUCLEUS_CELL;
    Column col = column(seed, cx, cy);
    float* p = params + 8 * k;
    centres[2 * k] = cx;
    centres[2 * k + 1] = cy;
    if (unit_f(h) >= col.nuclei) {
        p[6] = 0.0f;
        return;
    }
    float a = 2.2f + 1.6f * u[2];
    p[0] = thickness_at(seed, thickness, folds, cx, cy) * u[3];
    p[1] = a;
    p[2] = a * (0.55f + 0.45f * u[4]);
    p[3] = 2.0f + u[5];
    p[4] = col.angle;
    p[5] = 0.50f * (0.8f + 0.4f * u[6]);
    p[6] = 1.0f;
    p[7] = 0.0f;
}

// A red cell: centre (double2), and depth, present, in 2 floats.
extern "C" __global__ void he_red_cells(
    double* centres, float* params, int i0, int j0, int columns, int n,
    u32 seed, float thickness, u32 folds)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) {
        return;
    }
    int i = i0 + k % columns, j = j0 + k / columns;
    float u[4];
    for (u32 q = 0; q < 4u; q++) {
        u[q] = unit_f(hash3(seed, RED_CELLS + q, i, j, 0));
    }
    double cx = ((double)i + 0.5 + 0.25 * (double)(2.0f * u[1] - 1.0f)) * RED_CELL_CELL;
    double cy = ((double)j + 0.5 + 0.25 * (double)(2.0f * u[2] - 1.0f)) * RED_CELL_CELL;
    centres[2 * k] = cx;
    centres[2 * k + 1] = cy;
    float* p = params + 2 * k;
    if (u[0] >= 0.7f * column(seed, cx, cy).vessel) {
        p[1] = 0.0f;
        return;
    }
    float t = thickness_at(seed, thickness, folds, cx, cy);
    p[0] = 1.0f + (t - 2.0f) * u[3];
    p[1] = 1.0f;
}

__device__ float nucleus_density(u32 seed, const double* centre, const float* p, double x, double y, float depth)
{
    float dx = (float)(x - centre[0]), dy = (float)(y - centre[1]);
    float s, c;
    sincosf(p[4], &s, &c);
    float along = c * dx + s * dy, across = -s * dx + c * dy;
    float dz = depth - p[0];
    float la = along / p[1], lb = across / p[2], lc = dz / p[3];
    float lateral = la * la + lb * lb;
    float q = lateral + lc * lc;
    if (q > 1.2f) {
        return 0.0f;
    }
    float edge = q * (1.0f + 0.08f * improved(seed, NUCLEUS_EDGE, x, y, 2.0));
    float body = smoothstep_f(1.0f, 0.82f, edge) * (1.0f - 0.25f * q);
    float cross_section = fmaxf(1.0f - lc * lc, 1e-3f);
    float rim = 1.0f + 0.3f * smoothstep_f(0.6f, 0.85f, lateral / cross_section);
    float clumps = 1.0f - smoothstep_f(0.0f, 0.35f, cellular(seed, CHROMATIN, x, y, 0.7).f1);
    float grain = improved(seed, CHROMATIN + 100u, x, y, 0.5);
    float chromatin = 0.62f + 0.25f * clumps + 0.13f * grain;
    return p[5] * body * rim * chromatin;
}

__device__ float red_cell_density(const double* centre, const float* p, double x, double y, float depth)
{
    float dx = (float)(x - centre[0]), dy = (float)(y - centre[1]);
    float r = sqrtf(dx * dx + dy * dy) / RED_CELL_RADIUS;
    if (r > 1.0f) {
        return 0.0f;
    }
    float disc = smoothstep_f(1.0f, 0.9f, r) * smoothstep_f(1.0f, 0.8f, fabsf(depth - p[0]));
    return 0.40f * (1.0f - 0.35f * (1.0f - r * r)) * disc;
}

// --- render.rs ---

struct Grid {
    const double* centres;
    const float* params;
    int i0, j0, columns, stride;
    double cell;
};

__device__ unsigned short to_half(float v)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

// Each pixel's stains, summed over its depths and averaged over its points, as two halves.
extern "C" __global__ void he_render(
    unsigned short* out,
    const double* n_centres, const float* n_params, int n_i0, int n_j0, int n_columns,
    const double* r_centres, const float* r_params, int r_i0, int r_j0, int r_columns,
    double x0, double y0, int width, int height, double pixel, int points,
    int whole, float from_depth, float to_depth,
    u32 seed, float thickness, u32 folds)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }
    float weight = 1.0f / (float)(points * points);
    float sum_h = 0.0f, sum_e = 0.0f;
    for (int k = 0; k < points * points; k++) {
        double x = x0 + ((double)col + ((double)(k % points) + 0.5) / (double)points) * pixel;
        double y = y0 + ((double)row + ((double)(k / points) + 0.5) / (double)points) * pixel;
        Column c = column(seed, x, y);
        float t = thickness_at(seed, thickness, folds, x, y);
        float from = whole ? 0.0f : fmaxf(from_depth, 0.0f);
        float to = whole ? t : fminf(to_depth, t);
        float ph = 0.0f, pe = 0.0f;
        if (!(c.h == 0.0f && c.e == 0.0f && c.nuclei == 0.0f && c.vessel == 0.0f)) {
            int ni = (int)floor(x / NUCLEUS_CELL) - n_i0, nj = (int)floor(y / NUCLEUS_CELL) - n_j0;
            int ri = (int)floor(x / RED_CELL_CELL) - r_i0, rj = (int)floor(y / RED_CELL_CELL) - r_j0;
            for (float z = from; z < to; z += LAYER) {
                float layer = fminf(LAYER, to - z);
                float mid = z + layer / 2.0f;
                float nucleus = 0.0f, red = 0.0f;
                for (int dj = -1; dj <= 1; dj++) {
                    for (int di = -1; di <= 1; di++) {
                        int nk = (nj + dj) * n_columns + ni + di;
                        if (n_params[8 * nk + 6] != 0.0f) {
                            nucleus = fmaxf(nucleus, nucleus_density(seed, n_centres + 2 * nk, n_params + 8 * nk, x, y, mid));
                        }
                        int rk = (rj + dj) * r_columns + ri + di;
                        if (r_params[2 * rk + 1] != 0.0f) {
                            red = fmaxf(red, red_cell_density(r_centres + 2 * rk, r_params + 2 * rk, x, y, mid));
                        }
                    }
                }
                ph += (c.h + nucleus) * layer;
                pe += (c.e + red) * layer;
            }
        }
        sum_h += ph * weight;
        sum_e += pe * weight;
    }
    int at = 2 * (row * width + col);
    out[at] = to_half(sum_h);
    out[at + 1] = to_half(sum_e);
}
