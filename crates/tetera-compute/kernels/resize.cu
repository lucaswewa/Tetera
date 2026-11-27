// Resizing (P1.05b): each output pixel is a weighted sum of a window of input pixels, its
// weights from two tables made on the host, one per axis: area averaging to shrink, bilinear to
// enlarge. The CPU twin is in tetera-imaging's `resize` module and uses the same tables.
//
// A table gives each output column (or row) the first input it reads and `taps` weights in
// 1/16384ths, summing to 16384, so a pixel's weights sum to 2^28 and the sum is exact in 64
// bits. The window always lies inside the input: the host checked it.

extern "C" __global__ void resize_u16(unsigned short* out, const unsigned short* in, int in_width,
                                      int channels, int out_width, int out_height,
                                      const int* x_start, const unsigned short* x_weights,
                                      int x_taps, const int* y_start,
                                      const unsigned short* y_weights, int y_taps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) {
        return;
    }
    int xs = x_start[x], ys = y_start[y];
    const unsigned short* wx = x_weights + x * x_taps;
    const unsigned short* wy = y_weights + y * y_taps;
    for (int c = 0; c < channels; ++c) {
        unsigned long long sum = 0;
        for (int a = 0; a < y_taps; ++a) {
            const unsigned short* row = in + ((long long)(ys + a) * in_width + xs) * channels + c;
            unsigned long long across = 0;
            for (int b = 0; b < x_taps; ++b) {
                across += (unsigned long long)wx[b] * row[b * channels];
            }
            sum += across * wy[a];
        }
        out[((long long)y * out_width + x) * channels + c] = (unsigned short)((sum + (1ull << 27)) >> 28);
    }
}
