#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 2

__global__ void forward_conv_gpu_kernel(float *x, float *weight, float *y, 
                                int b, int c_in, int c_out, int height, int width, int ks, int num_tiles_w) {
    
                                    int oh = height - ks + 1; int ow = width - ks + 1;
    #define x_4d(batch, channel, height_, width_) x[batch*(c_in*height*width) + channel*(height*width) + height_*width_ + width_]
    #define y_4d(batch, channel, height_, width_) y[batch*(c_out*oh*ow) + channel*(oh*ow) + height_*ow + width_]
    #define weight_4d(out_channel, in_channel, height_, width_) weight[out_channel*(c_in*ks*ks) + in_channel*(ks*ks) + height_*ks + width_]

    int b_idx = blockIdx.z;
    int c_out_idx = blockIdx.x; 
    int tile_row = blockIdx.y / num_tiles_w;
    int tile_col = blockIdx.y % num_tiles_w;
    int h_idx = tile_row * TILE_SIZE + threadIdx.y;
    int w_idx = tile_col * TILE_SIZE + threadIdx.x;

    if (h_idx < oh && w_idx < ow) {
        y_4d(b_idx, c_out_idx, h_idx, w_idx) = 0;
        for (int p = 0; p < c_in; p++) {
            for (int q = 0; q < ks; q++) {
                for (int r = 0; r < ks; r++) {
                    y_4d(b_idx, c_out_idx, h_idx, w_idx) += x_4d(b_idx, p, h_idx+q, w_idx+r) * weight_4d(c_out_idx, p, q, r);
                }
            }
        }
    }
}


torch::Tensor forward_conv_gpu(torch::Tensor x, torch::Tensor weight) {
    // forward pass for convolutional layer
    // x: torch:: Tensor: [batch_size, in_channels, in_height, in_width]
    // w: torch:: Tensor: [out_channels, in_channels, kernel_height, kernel_width]
    // returns: torch:: Tensor: [batch_size, out_channels, out_height, out_width]
    int b = x.size(0); int height = x.size(2); int width = x.size(3);
    int c_in = x.size(1); int c_out = weight.size(0);
    int kh = weight.size(2); int kw = weight.size(3);
    int oh = height - kh + 1; int ow = width - kw + 1;
    auto y = torch::empty({b, c_out, oh, ow}, x.options());

    float *x_ptr = x.data_ptr<float>();
    float *weight_ptr = weight.data_ptr<float>();
    float *y_ptr = y.data_ptr<float>();

    // Number of horizontal tiles per output channel
    int num_tiles_w = (ow + TILE_SIZE - 1) / TILE_SIZE;
    // Number of vertical tiles per output channel
    int num_tiles_h = (oh + TILE_SIZE - 1) / TILE_SIZE;


    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(c_out, num_tiles_h*num_tiles_w, b);

    forward_conv_gpu_kernel<<<gridDim, blockDim>>>(x_ptr, weight_ptr, y_ptr, b, c_in, c_out, height, width, kh, num_tiles_w);
    return y;
}