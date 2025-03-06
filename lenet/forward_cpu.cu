#include <torch/extension.h>


torch::Tensor forward_conv_cpu(torch::Tensor x, torch::Tensor weight) {
    // forward pass for convolutional layer
    // x: torch:: Tensor: [batch_size, in_channels, in_height, in_width]
    // w: torch:: Tensor: [out_channels, in_channels, kernel_height, kernel_width]
    // returns: torch:: Tensor: [batch_size, out_channels, out_height, out_width]
    int b = x.size(0); int height = x.size(2); int width = x.size(3);
    int c_in = x.size(1); int c_out = weight.size(0);
    int kh = weight.size(2); int kw = weight.size(3);
    int oh = height - kh + 1; int ow = width - kw + 1;
    auto y = torch::empty({b, c_out, oh, ow}, x.options());

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < c_out; j++) {
            for (int h = 0; h < oh; h++) {
                for (int w = 0; w < ow; w++) {
                    y[i][j][h][w] = 0;
                    for (int p = 0; p < c_in; p++) {
                        for (int q = 0; q < kh; q++) {
                            for (int r = 0; r < kw; r++) {
                                y[i][j][h][w] += x[i][p][h+q][w+r] * weight[j][p][q][r];
                            }
                        }
                    }

                }
            }
        }
    }
    return y;
}

torch::Tensor forward_max_pool_cpu(torch::Tensor x, int kernel_size) {
    // forward pass for max pooling layer
    // x: torch:: Tensor: [batch_size, in_channels, in_height, in_width]
    // kernel_size: int: size of the kernel
    // returns: torch:: Tensor: [batch_size, in_channels, out_height, out_width]
    int b = x.size(0); int height = x.size(2); int width = x.size(3);
    int c = x.size(1);
    int oh = height / kernel_size; int ow = width / kernel_size;
    auto y = torch::empty({b, c, oh, ow}, x.options());

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < c; j++) {
            for (int h = 0; h < oh; h++) {
                for (int w = 0; w < ow; w++) {
                    float max_val = -1e9;
                    for (int p = 0; p < kernel_size; p++) {
                        for (int q = 0; q < kernel_size; q++) {
                            max_val = std::max(max_val, x[i][j][h*kernel_size+p][w*kernel_size+q].item<float>());
                        }
                    }
                    y[i][j][h][w] = max_val;
                }
            }
        }
    }
    return y;
}

torch::Tensor forward_relu_cpu(torch::Tensor x) {
    // forward pass for ReLU activation
    // x: torch:: Tensor: [batch_size, in_channels, in_height, in_width]
    // returns: torch:: Tensor: [batch_size, in_channels, in_height, in_width]
    auto y = torch::empty_like(x);
    
    for (int i = 0; i < x.size(0); i++) {
        for (int j = 0; j < x.size(1); j++) {
            for (int h = 0; h < x.size(2); h++) {
                for (int w = 0; w < x.size(3); w++) {
                    y[i][j][h][w] = std::max(0.0f, x[i][j][h][w].item<float>());
                }
            }
        }
    }
}