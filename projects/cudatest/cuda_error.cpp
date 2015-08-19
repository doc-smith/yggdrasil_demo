#include "cuda_error.h"


namespace CudaUtil {

    CudaError::CudaError(cudaError_t cudaErr)
        : runtime_error("cuda error")
        , CudaErr(cudaErr)
    {
    }


    const char* CudaError::what() const throw() {
        return cudaGetErrorString(CudaErr);
    }


    cudaError_t CudaError::getCudaError() const {
        return CudaErr;
    }

}

