// Bayer mosaics (P1.05a): helpers the imaging kernels share, the split into four colour planes,
// and the half-resolution superpixel. The CPU twins are in tetera-imaging's `bayer` module.
//
// A pattern is given as its first site's offset (ox, oy) in an RGGB tile. A site's channel is
// 0 red, 1 green on a red row, 2 green on a blue row, 3 blue.

__device__ __forceinline__ int bayer_channel(int x, int y, int ox, int oy)
{
    return ((y + oy) & 1) * 2 + ((x + ox) & 1);
}

// The four planes, back to back in channel order, each (width / 2) × (height / 2): an odd last
// column or row has no tile of its own and is left out.
extern "C" __global__ void bayer_split_u16(unsigned short* out, const unsigned short* in, int width,
                                           int height, int ox, int oy)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pw = width / 2, ph = height / 2;
    if (px >= pw || py >= ph) {
        return;
    }
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int x = 2 * px + dx, y = 2 * py + dy;
            int plane = bayer_channel(x, y, ox, oy);
            out[plane * pw * ph + py * pw + px] = in[y * width + x];
        }
    }
}

// One RGB pixel per tile: its red, the rounded mean of its greens, and its blue.
extern "C" __global__ void bayer_superpixel_u16(unsigned short* out, const unsigned short* in,
                                                int width, int height, int ox, int oy)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pw = width / 2, ph = height / 2;
    if (px >= pw || py >= ph) {
        return;
    }
    int sites[4];
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int x = 2 * px + dx, y = 2 * py + dy;
            sites[bayer_channel(x, y, ox, oy)] = in[y * width + x];
        }
    }
    int i = (py * pw + px) * 3;
    out[i] = sites[0];
    out[i + 1] = (sites[1] + sites[2] + 1) >> 1;
    out[i + 2] = sites[3];
}

// Binning by `factor` (P1.05b), keeping the Bayer layout: each output site sums, or averages
// rounding halves up, the factor × factor sites of its own colour in the block of tiles it
// stands for. The output is out_width × out_height, both even, and keeps the pattern.
extern "C" __global__ void bayer_bin_u16(unsigned short* out, const unsigned short* in, int width,
                                         int factor, int mean, int out_width, int out_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) {
        return;
    }
    int dx = x & 1, dy = y & 1;
    int tx = (x >> 1) * factor, ty = (y >> 1) * factor;
    unsigned int sum = 0;
    for (int j = 0; j < factor; ++j) {
        for (int i = 0; i < factor; ++i) {
            sum += in[(2 * (ty + j) + dy) * width + 2 * (tx + i) + dx];
        }
    }
    unsigned int n = factor * factor;
    out[y * out_width + x] = (unsigned short)(mean ? (sum + n / 2) / n : sum);
}
