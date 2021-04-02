// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "cuda_kernels.h"

#include <stdexcept>
//#include <cuda_fp16.h>

#include <iostream>

#define GOOGLE_CUDA 1

// bfloat16.h(25): error: expected a ";" compilation error without the line below
#include "tensorflow/core/framework/bfloat16.h"

#include "alltoall_kernel.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"


namespace tensorflow {
namespace tensorflow {

typedef Eigen::GpuDevice GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void AlltoallCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
      out[i] = in[i];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void AlltoallFunctor<GPUDevice, T>::operator()(
	const GPUDevice& d, int size, const T* in, T* out) {
	// Launch the cuda kernel.
	// See core/util/gpu_kernel_helper.h for example of computing
	// block count and thread_per_block count.
	int block_count = 1024;
	int thread_per_block = 20;
	AlltoallCudaKernel<T>
	     <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct AlltoallFunctor<GPUDevice, float>;

}  // end namespace functor
}  // end namespace tensorflow
