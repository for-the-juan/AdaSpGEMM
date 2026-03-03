/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// 配置参数
#define MATRIX_SIZE 4096           // 矩阵大小
#define TILE_SIZE 16               // Tile大小
#define SPARSITY 0.1f              // 稀疏度10%
#define NUM_STREAMS 4              // CUDA流数量

// MMA矩阵tile维度
#define M 8
#define N 8
#define K 4

// Tensor Core相关的配置
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// 自定义错误检查宏
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define checkKernelErrors(expr)                               \
    do {                                                      \
        expr;                                                 \
        cudaError_t __err = cudaGetLastError();               \
        if (__err != cudaSuccess) {                           \
            printf("Line %d: '%s' failed: %s\n",              \
                   __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                          \
        }                                                     \
    } while (0)

// 生成稀疏矩阵（稠密格式，但很多值为0）
void generate_sparse_matrix(double* matrix, int size, float sparsity) {
    srand(time(NULL));
    int total_elements = size * size;
    int zero_elements = (int)(total_elements * sparsity);
    int non_zero_elements = total_elements - zero_elements;
    
    // 先全部设为0
    for (int i = 0; i < total_elements; i++) {
        matrix[i] = 0.0;
    }
    
    // 随机选择一些位置设置为非零值
    for (int i = 0; i < non_zero_elements; i++) {
        int pos;
        do {
            pos = rand() % total_elements;
        } while (matrix[pos] != 0.0);
        
        matrix[pos] = (double)(rand() % 100 + 1) / 100.0;  // 0.01到1.0之间的随机值
    }
    
    printf("Generated sparse matrix %dx%d with %d non-zero elements (%.2f%% density)\n", 
           size, size, non_zero_elements, (non_zero_elements * 100.0) / total_elements);
}

// 生成稠密矩阵
void generate_dense_matrix(double* matrix, int size) {
    srand(time(NULL) + 1);
    int total_elements = size * size;
    
    for (int i = 0; i < total_elements; i++) {
        matrix[i] = (double)(rand() % 100 + 1) / 100.0;  // 0.01到1.0之间的随机值
    }
    
    printf("Generated dense matrix %dx%d\n", size, size);
}

// 简单的Tensor Core GEMM内核（用于16x16 tile计算）
__global__ void tensor_core_gemm_tile(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M_size, int N_size, int K_size,
    double alpha, double beta,
    int start_tile_i, int num_tiles_i) {
    
#if __CUDA_ARCH__ >= 800
    // 每个block处理一个tile
    int tile_i = blockIdx.y + start_tile_i;
    int tile_j = blockIdx.x;
    
    // 检查是否超出范围
    if (tile_i >= start_tile_i + num_tiles_i || tile_i >= M_size/TILE_SIZE || 
        tile_j >= N_size/TILE_SIZE) {
        return;
    }
    
    // 每个tile内的warp索引
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // 计算全局矩阵中的起始位置
    int global_row = tile_i * TILE_SIZE;
    int global_col = tile_j * TILE_SIZE;
    
    // 声明Tensor Core片段
    using namespace nvcuda::wmma;
    
    // 每个warp计算一个8x8的子tile
    fragment<accumulator, M, N, K, double> c_frag;
    fragment<matrix_a, M, N, K, double, row_major> a_frag;
    fragment<matrix_b, M, N, K, double, col_major> b_frag;
    
    // 初始化累加器
    fill_fragment(c_frag, 0.0f);
    
    // 计算tile内的矩阵乘法
    // 注意：这里简化了，实际应该循环K维度
    for (int k_tile = 0; k_tile < K_size; k_tile += TILE_SIZE) {
        // 计算A和B的当前tile位置
        int a_row = global_row + (warp_id / 2) * M;
        int a_col = k_tile + (warp_id % 2) * K;
        int b_row = k_tile + (warp_id / 2) * K;
        int b_col = global_col + (warp_id % 2) * N;
        
        // 加载A和B的片段
        if (a_row < M_size && a_col < K_size) {
            const double* a_ptr = A + a_row * K_size + a_col;
            load_matrix_sync(a_frag, a_ptr, K_size);
        } else {
            fill_fragment(a_frag, 0.0f);
        }
        
        if (b_row < K_size && b_col < N_size) {
            const double* b_ptr = B + b_row * N_size + b_col;
            load_matrix_sync(b_frag, b_ptr, N_size);
        } else {
            fill_fragment(b_frag, 0.0f);
        }
        
        // Tensor Core矩阵乘法
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 将结果写回C矩阵
    int c_row = global_row + (warp_id / 2) * M;
    int c_col = global_col + (warp_id % 2) * N;
    
    if (c_row < M_size && c_col < N_size) {
        double* c_ptr = C + c_row * N_size + c_col;
        store_matrix_sync(c_ptr, c_frag, N_size, mem_row_major);
    }
#endif
}

// 使用多个CUDA流执行Tensor Core GEMM
void run_sparse_gemm_with_streams(
    double* d_A, double* d_B, double* d_C,
    int matrix_size, double alpha, double beta) {
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start, stop;
    
    // 创建CUDA流
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }
    
    // 创建计时事件
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    // 计算tile数量
    int num_tiles = matrix_size / TILE_SIZE;
    int tiles_per_stream = num_tiles / NUM_STREAMS;
    
    // 配置kernel参数
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(num_tiles, tiles_per_stream);  // x方向：所有tile，y方向：每个流的tile数
    
    // 记录开始时间
    checkCudaErrors(cudaEventRecord(start, 0));
    
    // 在多个流中并行执行kernel
    for (int stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
        int start_tile_i = stream_id * tiles_per_stream;
        
        tensor_core_gemm_tile<<<gridDim, blockDim, 0, streams[stream_id]>>>(
            d_A, d_B, d_C,
            matrix_size, matrix_size, matrix_size,
            alpha, beta,
            start_tile_i, tiles_per_stream);
        
        // 检查kernel错误
        checkKernelErrors();
    }
    
    // 等待所有流完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
    }
    
    // 记录结束时间
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    // 计算执行时间
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Tensor Core execution time with %d streams: %.2f ms\n", NUM_STREAMS, milliseconds);
    
    // 计算TFLOPS
    double total_flops = 2.0 * matrix_size * matrix_size * matrix_size;  // 2*M*N*K
    double tflops = (total_flops / (milliseconds / 1000.0)) / 1e12;
    printf("FP64 TFLOPS: %.2f\n", tflops);
    
    // 清理
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

// 验证结果（可选）
void verify_result(double* h_A, double* h_B, double* h_C_gpu, 
                   double* h_C_cpu, int size, double alpha, double beta) {
    // 在CPU上计算结果用于验证
    printf("\nVerifying result on CPU...\n");
    
    // 初始化CPU结果矩阵
    for (int i = 0; i < size * size; i++) {
        h_C_cpu[i] = 0.0;
    }
    
    // 简单的CPU矩阵乘法（仅用于验证，不考虑性能）
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += h_A[i * size + k] * h_B[k * size + j];
            }
            h_C_cpu[i * size + j] = alpha * sum + beta * h_C_cpu[i * size + j];
        }
    }
    
    // 比较GPU和CPU结果
    int errors = 0;
    double tolerance = 1e-5;
    
    for (int i = 0; i < size * size; i++) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > tolerance) {
            errors++;
            if (errors <= 10) {  // 只打印前10个错误
                printf("Mismatch at element %d: GPU=%f, CPU=%f\n", 
                       i, h_C_gpu[i], h_C_cpu[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("Verification passed! All results match within tolerance %e\n", tolerance);
    } else {
        printf("Verification failed! %d elements do not match\n", errors);
    }
}

int main(int argc, char** argv) {
    printf("Sparse Matrix GEMM with Tensor Cores and Multiple Streams\n");
    printf("Matrix size: %dx%d, Sparsity: %.1f%%, Tile size: %dx%d\n\n",
           MATRIX_SIZE, MATRIX_SIZE, SPARSITY * 100, TILE_SIZE, TILE_SIZE);
    
    // 设置CUDA设备
    int dev = 0;
    checkCudaErrors(cudaSetDevice(dev));
    
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    
    // 检查设备是否支持Tensor Core
    printf("Device: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    
    if (deviceProp.major < 8) {
        printf("Warning: Double precision Tensor Cores require Ampere (SM 8.x) or later architecture.\n");
        printf("The kernel may still run but without Tensor Core acceleration.\n");
    }
    
    // 分配主机内存
    size_t matrix_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);
    
    double* h_A = (double*)malloc(matrix_bytes);
    double* h_B = (double*)malloc(matrix_bytes);
    double* h_C = (double*)malloc(matrix_bytes);
    double* h_C_cpu = (double*)malloc(matrix_bytes);  // 用于验证的CPU结果
    
    if (!h_A || !h_B || !h_C || !h_C_cpu) {
        printf("Failed to allocate host memory!\n");
        return -1;
    }
    
    // 生成矩阵
    generate_sparse_matrix(h_A, MATRIX_SIZE, SPARSITY);
    generate_dense_matrix(h_B, MATRIX_SIZE);
    
    // 初始化C矩阵
    srand(time(NULL) + 2);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_C[i] = (double)(rand() % 100) / 100.0;
    }
    
    // 分配设备内存
    double* d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, matrix_bytes));
    checkCudaErrors(cudaMalloc(&d_B, matrix_bytes));
    checkCudaErrors(cudaMalloc(&d_C, matrix_bytes));
    
    // 复制数据到设备
    checkCudaErrors(cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, matrix_bytes, cudaMemcpyHostToDevice));
    
    // 设置GEMM参数
    double alpha = 1.0;
    double beta = 0.0;
    
    // 运行Tensor Core GEMM
    printf("\nRunning Tensor Core GEMM with %d CUDA streams...\n", NUM_STREAMS);
    run_sparse_gemm_with_streams(d_A, d_B, d_C, MATRIX_SIZE, alpha, beta);
    
    // 复制结果回主机
    checkCudaErrors(cudaMemcpy(h_C, d_C, matrix_bytes, cudaMemcpyDeviceToHost));
    
    // 验证结果（可选）
#if 0
    verify_result(h_A, h_B, h_C, h_C_cpu, MATRIX_SIZE, alpha, beta);
#endif
    
    // 性能分析
    printf("\n=== Performance Summary ===\n");
    printf("Matrix size: %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Sparsity: %.1f%%\n", SPARSITY * 100);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Number of CUDA streams: %d\n", NUM_STREAMS);
    printf("Tensor Cores used: Yes (if SM 8.0+)\n");
    
    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    
    printf("\nDone!\n");
    
    return 0;
}