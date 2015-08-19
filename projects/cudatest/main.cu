#include "cuda_buffer.h"
#include "cuda_error.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <iterator>
#include <vector>

using namespace CudaUtil;


__global__ void Kernel(float* in1, float* in2, float* out, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in1[i] + in2[i];
    }
}


void AddGPU(float* in1, float* in2, float* out, int n) {
    CudaBuffer<float> deviceInput1(in1, n);
    CudaBuffer<float> deviceInput2(in2, n);
    CudaBuffer<float> deviceOutput(n);

    unsigned int blockSize = 1024;
    unsigned int gridSize = (n - 1) / blockSize + 1;

    Kernel<<<gridSize, blockSize>>>(deviceInput1.getData(),
                                    deviceInput2.getData(),
                                    deviceOutput.getData(),
                                    n);

    deviceOutput.copyToHost(out);
}


int main(int argc, const char** argv) {
    int n;
    std::cin >> n;
    std::vector<float> v1(n);
    std::vector<float> v2(n);
    for (int i = 0; i < n; ++i) std::cin >> v1[i];
    for (int i = 0; i < n; ++i) std::cin >> v2[i];

    std::vector<float> res(n);
    AddGPU(v1.data(), v2.data(), res.data(), n);

    std::copy(res.begin(), res.end(),
            std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}

