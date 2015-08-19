#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>


namespace CudaUtil {

    class CudaError : public std::runtime_error {
    public:
        CudaError(cudaError_t cudaErr);

        virtual const char* what() const throw();
        cudaError_t getCudaError() const;

    private:
        cudaError_t CudaErr;
    };

}


#define SAFE_CUDA(CALL) \
    { \
        cudaError_t err = CALL; \
        if (err != cudaSuccess) { \
            throw CudaUtil::CudaError(err); \
        } \
    }

