#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/std/type_traits>
#include <mma.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace nvcuda;

// 配置参数
const int MATRIX_SIZE = 4096;           // 矩阵大小
const double SPARSITY = 0.1;           // 稀疏度10%
const int TILE_SIZE = 8;               // DMMA瓦片大小
const int BLOCK_SIZE = 16;             // CUDA Core块大小
const int NUM_WARPS = 8;               // 每个块的warp数
const int WARP_SIZE = 32;              // warp大小

// 用于双精度DMMA的配置
#define M 8
#define N 8
#define K 4

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 生成稀疏矩阵（CSR格式）
void generateSparseMatrixCSR(int rows, int cols, double sparsity, 
                             std::vector<double>& values,
                             std::vector<int>& col_indices,
                             std::vector<int>& row_offsets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> val_dis(0.5, 2.0);
    
    row_offsets.resize(rows + 1, 0);
    
    // 第一遍：统计每行的非零元素数量
    for (int i = 0; i < rows; i++) {
        int nnz_in_row = 0;
        for (int j = 0; j < cols; j++) {
            if (dis(gen) < sparsity) {
                nnz_in_row++;
            }
        }
        row_offsets[i + 1] = row_offsets[i] + nnz_in_row;
    }
    
    int total_nnz = row_offsets[rows];
    values.resize(total_nnz);
    col_indices.resize(total_nnz);
    
    // 第二遍：填充值和列索引
    for (int i = 0; i < rows; i++) {
        int offset = row_offsets[i];
        for (int j = 0; j < cols; j++) {
            if (dis(gen) < sparsity) {
                values[offset] = val_dis(gen);
                col_indices[offset] = j;
                offset++;
            }
        }
    }
}

// 生成密集矩阵
void generateDenseMatrix(int rows, int cols, double* matrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.5, 2.0);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// CUDA Core版本的稀疏矩阵乘法（CSR格式）
__global__ void sparseMatMulCUDA(const double* values, const int* col_indices, 
                                 const int* row_offsets, const double* B, 
                                 double* C, int rows, int cols, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        int start = row_offsets[row];
        int end = row_offsets[row + 1];
        
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int idx = start; idx < end; idx++) {
                int col = col_indices[idx];
                sum += values[idx] * B[col * k + j];
            }
            C[row * k + j] = sum;
        }
    }
}

// CUDA Core版本的稠密矩阵乘法
__global__ void denseMatMulCUDA(const double* A, const double* B, double* C, 
                                int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// DMMA版本的稠密矩阵乘法（使用Tensor Core）
__global__ void denseMatMulDMMA(const double* A, const double* B, double* C, 
                                int m, int n, int k) {
#if __CUDA_ARCH__ >= 800
    // 每个线程块处理一个8x8的瓦片
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // 声明片段
    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> acc_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 循环遍历K维度
    for (int i = 0; i < n; i += K) {
        int aCol = i;
        int aRow = warpM * M;
        
        int bCol = warpN * N;
        int bRow = i;
        
        // 边界检查
        if (aRow < m && aCol < n && bRow < n && bCol < k) {
            // 加载矩阵片段
            wmma::load_matrix_sync(a_frag, A + aCol + aRow * n, n);
            wmma::load_matrix_sync(b_frag, B + bRow + bCol * k, k);
            
            // 执行矩阵乘法
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // 加载C矩阵并存储结果
    int cCol = warpN * N;
    int cRow = warpM * M;
    
    if (cRow < m && cCol < k) {
        wmma::load_matrix_sync(c_frag, C + cCol + cRow * k, k, wmma::mem_row_major);
        
        // 将结果加回C（这里直接替换，实际可以是C = αA*B + βC）
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc_frag.x[i];
        }
        
        // 存储结果
        wmma::store_matrix_sync(C + cCol + cRow * k, c_frag, k, wmma::mem_row_major);
    }
#endif
}

// 优化版本的DMMA矩阵乘法（使用共享内存）
__global__ void denseMatMulDMMAOtimized(const double* A, const double* B, double* C,
                                        int m, int n, int k) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ double shmem[];
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // 每个块处理64x64的瓦片
    const int blockRow = blockIdx.y * 64;
    const int blockCol = blockIdx.x * 64;
    
    // 每个warp处理8x8的瓦片
    const int warpRow = (warpId / 4) * 8;
    const int warpCol = (warpId % 4) * 8;
    
    // 声明累加器片段
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;
    wmma::fill_fragment(c_frag, 0.0);
    
    // 遍历K维度
    for (int tile_k = 0; tile_k < n; tile_k += 16) {
        // 声明矩阵片段
        wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
        
        // 从全局内存加载A和B的瓦片到共享内存
        // 这里简化处理，实际需要更复杂的共享内存管理
        
        // 加载A片段
        int aRow = blockRow + warpRow;
        int aCol = tile_k;
        
        if (aRow < m && aCol < n) {
            wmma::load_matrix_sync(a_frag, A + aCol + aRow * n, n);
        }
        
        // 加载B片段
        int bRow = tile_k;
        int bCol = blockCol + warpCol;
        
        if (bRow < n && bCol < k) {
            wmma::load_matrix_sync(b_frag, B + bRow + bCol * k, k);
        }
        
        // 执行MMA操作
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    int cRow = blockRow + warpRow;
    int cCol = blockCol + warpCol;
    
    if (cRow < m && cCol < k) {
        wmma::store_matrix_sync(C + cCol + cRow * k, c_frag, k, wmma::mem_row_major);
    }
#endif
}

// 将稀疏矩阵转换为稠密矩阵
void sparseToDense(const std::vector<double>& values,
                   const std::vector<int>& col_indices,
                   const std::vector<int>& row_offsets,
                   int rows, int cols,
                   std::vector<double>& dense_matrix) {
    dense_matrix.resize(rows * cols, 0.0);
    
    for (int i = 0; i < rows; i++) {
        int start = row_offsets[i];
        int end = row_offsets[i + 1];
        
        for (int idx = start; idx < end; idx++) {
            int col = col_indices[idx];
            dense_matrix[i * cols + col] = values[idx];
        }
    }
}

// 验证矩阵乘法结果
bool verifyResult(const double* A, const double* B, const double* C,
                  int m, int n, int k, double epsilon = 1e-6) {
    std::vector<double> reference(m * k, 0.0);
    
    // CPU参考计算
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int t = 0; t < n; t++) {
                sum += A[i * n + t] * B[t * k + j];
            }
            reference[i * k + j] = sum;
        }
    }
    
    // 比较结果
    for (int i = 0; i < m * k; i++) {
        if (fabs(C[i] - reference[i]) > epsilon) {
            std::cout << "Mismatch at position " << i 
                      << ": GPU=" << C[i] << ", CPU=" << reference[i] 
                      << ", diff=" << fabs(C[i] - reference[i]) << std::endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "=== Sparse Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << "Matrix Size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    std::cout << "Sparsity: " << SPARSITY * 100 << "%" << std::endl;
    
    // 设备检查
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Tensor Cores Support: " 
              << (prop.major >= 8 ? "Yes (Ampere+)" : "Limited/No") << std::endl;
    
    if (prop.major < 8) {
        std::cout << "Warning: DMMA requires compute capability 8.0 or higher for optimal performance" << std::endl;
    }
    
    // 生成稀疏矩阵（CSR格式）
    std::vector<double> sparse_values;
    std::vector<int> col_indices;
    std::vector<int> row_offsets;
    
    std::cout << "\nGenerating sparse matrix with " << SPARSITY * 100 << "% sparsity..." << std::endl;
    generateSparseMatrixCSR(MATRIX_SIZE, MATRIX_SIZE, SPARSITY, 
                            sparse_values, col_indices, row_offsets);
    
    int nnz = sparse_values.size();
    std::cout << "Non-zero elements: " << nnz << " (" 
              << (double)nnz / (MATRIX_SIZE * MATRIX_SIZE) * 100 << "%)" << std::endl;
    
    // 生成稠密矩阵
    std::vector<double> dense_A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> dense_B(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> dense_C(MATRIX_SIZE * MATRIX_SIZE, 0.0);
    std::vector<double> dense_result(MATRIX_SIZE * MATRIX_SIZE, 0.0);
    
    // 将稀疏矩阵转换为稠密矩阵用于DMMA版本
    sparseToDense(sparse_values, col_indices, row_offsets,
                  MATRIX_SIZE, MATRIX_SIZE, dense_A);
    generateDenseMatrix(MATRIX_SIZE, MATRIX_SIZE, dense_B.data());
    
    // 设备内存分配
    double *d_sparse_values, *d_col_indices, *d_row_offsets;
    double *d_A, *d_B, *d_C, *d_result;
    int *d_col_indices_int, *d_row_offsets_int;
    
    CUDA_CHECK(cudaMalloc(&d_sparse_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col_indices_int, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_offsets_int, (MATRIX_SIZE + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_sparse_values, sparse_values.data(), 
                         nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices_int, col_indices.data(), 
                         nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_offsets_int, row_offsets.data(), 
                         (MATRIX_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, dense_A.data(), 
                         MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, dense_B.data(), 
                         MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // ========== 1. CUDA Core稀疏矩阵乘法 ==========
    std::cout << "\n1. CUDA Core Sparse Matrix Multiplication" << std::endl;
    
    dim3 blockDim(256, 1);
    dim3 gridDim((MATRIX_SIZE + blockDim.x - 1) / blockDim.x, 1);
    
    CUDA_CHECK(cudaEventRecord(start));
    
    sparseMatMulCUDA<<<gridDim, blockDim>>>(
        d_sparse_values, d_col_indices_int, d_row_offsets_int,
        d_B, d_C, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaGetLastError());
    
    float cuda_core_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_core_time, start, stop));
    
    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(dense_C.data(), d_C, 
                         MATRIX_SIZE * MATRIX_SIZE * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    // ========== 2. CUDA Core稠密矩阵乘法 ==========
    std::cout << "\n2. CUDA Core Dense Matrix Multiplication" << std::endl;
    
    dim3 blockDimDense(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimDense((MATRIX_SIZE + blockDimDense.x - 1) / blockDimDense.x,
                     (MATRIX_SIZE + blockDimDense.y - 1) / blockDimDense.y);
    
    CUDA_CHECK(cudaEventRecord(start));
    
    denseMatMulCUDA<<<gridDimDense, blockDimDense>>>(
        d_A, d_B, d_result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaGetLastError());
    
    float cuda_core_dense_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_core_dense_time, start, stop));
    
    // 验证稠密矩阵乘法结果
    // bool dense_correct = verifyResult(dense_A.data(), dense_B.data(), dense_result.data(),
    //                                  MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    // std::cout << "Dense multiplication verification: " 
    //           << (dense_correct ? "PASSED" : "FAILED") << std::endl;
    
    // ========== 3. Tensor Core DMMA稠密矩阵乘法 ==========
    std::cout << "\n3. Tensor Core DMMA Dense Matrix Multiplication" << std::endl;
    
    // 检查设备是否支持Tensor Core
    if (prop.major >= 8) {
        // 配置DMMA内核
        dim3 blockDimDMMA(32 * NUM_WARPS, 1);  // 256 threads
        dim3 gridDimDMMA((MATRIX_SIZE + M - 1) / M,
                        (MATRIX_SIZE + N - 1) / N);
        
        // 清空结果
        CUDA_CHECK(cudaMemset(d_result, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double)));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        denseMatMulDMMA<<<gridDimDMMA, blockDimDMMA>>>(
            d_A, d_B, d_result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        CUDA_CHECK(cudaGetLastError());
        
        float dmma_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&dmma_time, start, stop));
        
        // 拷贝结果回主机
        std::vector<double> dmma_result(MATRIX_SIZE * MATRIX_SIZE);
        CUDA_CHECK(cudaMemcpy(dmma_result.data(), d_result, 
                            MATRIX_SIZE * MATRIX_SIZE * sizeof(double), 
                            cudaMemcpyDeviceToHost));
        
        // // 验证DMMA结果
        // bool dmma_correct = verifyResult(dense_A.data(), dense_B.data(), dmma_result.data(),
        //                                 MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1e-5);
        // std::cout << "DMMA verification: " 
        //           << (dmma_correct ? "PASSED" : "FAILED") << std::endl;
        
        // ========== 性能统计 ==========
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        double total_ops = 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
        double cuda_core_gflops = total_ops / (cuda_core_dense_time * 1e-3) / 1e9;
        double dmma_gflops = total_ops / (dmma_time * 1e-3) / 1e9;
        
        std::cout << "\nCUDA Core (Sparse CSR): " << cuda_core_time << " ms" << std::endl;
        std::cout << "CUDA Core (Dense): " << cuda_core_dense_time << " ms" 
                  << " | " << cuda_core_gflops << " GFLOPS" << std::endl;
        std::cout << "Tensor Core DMMA: " << dmma_time << " ms" 
                  << " | " << dmma_gflops << " GFLOPS" << std::endl;
        
        std::cout << "\nSpeedup (DMMA vs CUDA Core Dense): " 
                  << cuda_core_dense_time / dmma_time << "x" << std::endl;
        std::cout << "Speedup (CUDA Core Sparse vs Dense): " 
                  << cuda_core_dense_time / cuda_core_time << "x" << std::endl;
        
        // 理论性能分析
        std::cout << "\n=== Theoretical Analysis ===" << std::endl;
        std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
        std::cout << "Total operations: " << total_ops / 1e9 << " GFLOP" << std::endl;
        std::cout << "Sparsity utilization: " << SPARSITY * 100 << "%" << std::endl;
        
        // 计算理论内存带宽
        double bytes_sparse = (nnz * (sizeof(double) + sizeof(int)) + 
                              (MATRIX_SIZE + 1) * sizeof(int) +
                              MATRIX_SIZE * MATRIX_SIZE * sizeof(double)) / 1e6;
        double bytes_dense = 3.0 * MATRIX_SIZE * MATRIX_SIZE * sizeof(double) / 1e6;
        
        std::cout << "Memory traffic (sparse): " << bytes_sparse << " MB" << std::endl;
        std::cout << "Memory traffic (dense): " << bytes_dense << " MB" << std::endl;
        
    } else {
        std::cout << "Device does not support DMMA (requires compute capability 8.0+)" << std::endl;
        
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CUDA Core (Sparse CSR): " << cuda_core_time << " ms" << std::endl;
        std::cout << "CUDA Core (Dense): " << cuda_core_dense_time << " ms" << std::endl;
        
        double total_ops = 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
        double cuda_core_gflops = total_ops / (cuda_core_dense_time * 1e-3) / 1e9;
        
        std::cout << "CUDA Core (Dense) GFLOPS: " << cuda_core_gflops << std::endl;
    }
    
    // 清理资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    CUDA_CHECK(cudaFree(d_sparse_values));
    CUDA_CHECK(cudaFree(d_col_indices_int));
    CUDA_CHECK(cudaFree(d_row_offsets_int));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_result));
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
    
    return 0;
}