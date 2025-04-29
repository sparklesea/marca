#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 快速指数内核
__device__ inline float fast_exp_device(float y) {
    union {
        uint32_t i;
        float f;
    } v;
    v.i = static_cast<uint32_t>((1.4426950409f * y + 126.94201519f) * (1 << 23));
    return v.f + 0.0285784f;
}

// 并行处理张量的每个元素
__global__ void fast_exp_kernel(
    const float* input,
    float* output,
    int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fast_exp_device(input[idx]);
    }
}

// PyTorch 封装函数
torch::Tensor fast_exp_launcher(torch::Tensor x) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");

    // 创建输出张量
    auto output = torch::empty_like(x);

    // 获取指针和元素数量
    const int num_elements = x.numel();
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // 启动 CUDA 内核
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    fast_exp_kernel<<<blocks, threads_per_block>>>(
        input_ptr,
        output_ptr,
        num_elements
    );

    return output;
}

// 绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_exp", &fast_exp_launcher, "Fast approximate exp (CUDA)");
}