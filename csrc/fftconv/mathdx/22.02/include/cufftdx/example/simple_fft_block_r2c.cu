#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "block_io.hpp"
#include "common.hpp"

template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_kernel_r2c(ScalarType* input_data, ComplexType* output_data) {
    using complex_type = ComplexType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load_r2c(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store_r2c(thread_data, output_data, local_fft_id);
}

// In this example a one-dimensional real-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point R2C float precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
// Notice different sizes of input and output buffer, and R2C load and store operations in the kernel.
template<unsigned int Arch>
void simple_block_fft_r2c() {
    using namespace cufftdx;

    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    using FFT          = decltype(Block() + Size<128>() + Type<fft_type::r2c>() + Direction<fft_direction::forward>() +
                         Precision<float>() + ElementsPerThread<8>() + FFTsPerBlock<2>() + SM<Arch>());
    using complex_type = typename FFT::value_type;
    using real_type    = typename complex_type::value_type;

    // Allocate managed memory for input/output
    real_type* input_data;
    auto       input_size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto       input_size_bytes = input_size * sizeof(real_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input_data, input_size_bytes));
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = float(i);
    }
    complex_type* output_data;
    auto          output_size       = FFT::ffts_per_block * (cufftdx::size_of<FFT>::value / 2 + 1);
    auto          output_size_bytes = output_size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output_data, output_size_bytes));

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << input_data[i] << std::endl;
    }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel_r2c<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel_r2c<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data, output_data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < (cufftdx::size_of<FFT>::value / 2 + 1); i++) {
        std::cout << output_data[i].x << " " << output_data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(input_data));
    CUDA_CHECK_AND_EXIT(cudaFree(output_data));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct simple_block_fft_r2c_functor {
    void operator()() { return simple_block_fft_r2c<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_block_fft_r2c_functor>();
}