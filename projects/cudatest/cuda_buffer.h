#pragma once

#include "cuda_error.h"


namespace CudaUtil {

    template <typename T>
    class CudaBuffer {
    private:
        CudaBuffer(const CudaBuffer& other);
        CudaBuffer& operator = (const CudaBuffer& other);

    public:
        CudaBuffer(size_t count)
            : Count(count)
        {
            SAFE_CUDA(cudaMalloc((void**)&Data, count * sizeof(T)));
        }

        CudaBuffer(const T* sourceData, size_t count)
            : Count(count)
        {
            size_t byteSize = Count * sizeof(T);
            SAFE_CUDA(cudaMalloc((void**)&Data, byteSize));
            SAFE_CUDA(cudaMemcpy(Data, sourceData, byteSize, cudaMemcpyHostToDevice));
        }

        void copyToHost(T* hostData) const {
            size_t byteSize = Count * sizeof(T);
            SAFE_CUDA(cudaMemcpy(hostData, Data, byteSize, cudaMemcpyDeviceToHost));
        }

        T* getData() const {
            return Data;
        }

        size_t getCount() const {
            return Count;
        }

        size_t getSize() const {
            return Count * sizeof(T);
        }

        ~CudaBuffer() {
            cudaFree(Data);
        }

    private:
        T* Data;
        size_t Count;
    };

}

