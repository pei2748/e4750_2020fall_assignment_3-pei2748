import time
import numpy as np
import math
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import random



class MatrixMultiply:
    def __init__(self, height, width):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # host variables
        
        # tile parameters
        self.tile_width = 16
        self.block = (self.tile_width, self.tile_width, 1)
        m = max(height, width)
        self.grid = (math.ceil(m/self.tile_width), math.ceil(m/self.tile_width), 1)

        
        matrix_mul_naive_kernel_code = """
        #define TILE_WIDTH 16
        __global__ void Matrix_multiply_naive(float *A, float *B, float *C,
        const unsigned int M, const unsigned int N)
        {
            // Your kernel code here
            // one thread computes one row of input
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int Col = blockDim.x * bx + tx;
            int Row = blockDim.y * by + ty;
            float tmp;
            int i;
//            if(Row < M && Col < M) {
            tmp = 0;
            for(i = 0; i < N; i++)
                     tmp += A[Row * N + i] * B[i * M + Col];
            C[Row * M + Col] = tmp;
//            }
        }
        """

        matrix_mul_optimized1_kernel_code = """
        #define TILE_WIDTH 16
        __global__ void Matrix_multiply_optimized1(float *A, float *B, float *C,
        const unsigned int M, const unsigned int N)
        {
            //'B' tiles are shared. 'A' tiles are in global memory
            __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
            // float ds_A[TILE_WIDTH][TILE_WIDTH];
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int Col = blockDim.x * bx + tx;
            int Row = blockDim.y * by + ty;
            float tmp = 0;
            for(int t = 0; t < (N/TILE_WIDTH + 1); ++t) {
                if(Row < M && Col < N) {
                     //ds_A[ty][tx] = A[Row * N + t * TILE_WIDTH + tx];
                     ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * M + Col];
                     __syncthreads();
                     for (int i = 0; i < TILE_WIDTH; ++i)
                          tmp += A[Row * N + t * TILE_WIDTH + tx + i] * ds_B[i][tx];
                     __syncthreads();
                }
            }
            C[Row * M + Col] = tmp;
            
        }
        """

        matrix_mul_optimized2_kernel_code = """
        #define TILE_WIDTH 16

        __global__ void Matrix_multiply_optimized2(float *A, float *B, float *C,
        const unsigned int M, const unsigned int N)
        {
            // Your kernel code here
            __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
            __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int Col = blockDim.x * bx + tx;
            int Row = blockDim.y * by + ty;
            float tmp = 0;
            for(int t = 0; t < N/TILE_WIDTH; ++t) {
                ds_A[ty][tx] = A[Row * N + t * TILE_WIDTH + tx];
                ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * M + Col];
                __syncthreads();
                for (int i = 0; i < TILE_WIDTH; ++i)
                     tmp += ds_A[ty][i] * ds_B[i][tx];
                __syncthreads();
            }
            C[Row * M + Col] = tmp;
        }
        """
        # Build kernel codes (x3)
        self.prg_naive = SourceModule(matrix_mul_naive_kernel_code)
        self.prg_opt1 = SourceModule(matrix_mul_optimized1_kernel_code)
        self.prg_opt2 = SourceModule(matrix_mul_optimized2_kernel_code)

    def matrix_mul_naive(self, a_cpu, b_cpu):
        """
        Function to perform matrix multipication. Should return the result
        and execution time.
        """
        da = gpuarray.to_gpu(a_cpu);
        db = gpuarray.to_gpu(b_cpu);
        height_c = a_cpu.shape[0]
        width_c = b_cpu.shape[1]
        dc = gpuarray.empty((np.int32(height_c), np.int32(width_c)), np.dtype(np.float32))
        _start = cuda.Event()
        _end = cuda.Event()        
        func = self.prg_naive.get_function("Matrix_multiply_naive")
        _start.record()
        func(da, db, dc, np.int32(height_c), np.int32(a_cpu.shape[1]), block=self.block, grid=self.grid)
        _end.record()
        _end.synchronize()
        c_naive = dc.get()
        time_naive = _start.time_till(_end) # in ns
        return c_naive, time_naive

    def matrix_mul_optimized1(self, a_cpu, b_cpu):
        """
        Function to perform partially optimized matrix multipication. 
        Should return the result and execution time.
        (Only B tiled in shared memory)
        """
        da = gpuarray.to_gpu(a_cpu);
        db = gpuarray.to_gpu(b_cpu);
        height_c = a_cpu.shape[0]
        width_c = b_cpu.shape[1]
        dc = gpuarray.empty((np.int32(height_c), np.int32(width_c)), np.dtype(np.float32))
        _start = cuda.Event()
        _end = cuda.Event()        
        func = self.prg_opt1.get_function("Matrix_multiply_optimized1")
        _start.record()
        func(da, db, dc, np.int32(height_c), np.int32(a_cpu.shape[1]), block=self.block, grid=self.grid)
        _end.record()
        _end.synchronize()
        c_optimized1 = dc.get()
        time_optimized1 = _start.time_till(_end)  # in ns
        return c_optimized1, time_optimized1

    def matrix_mul_optimized2(self, a_cpu, b_cpu):
        """
        Function to perform optimized matrix multiplication using shared
        memory. Should return the result and execution time.
        (A and B both tiled in shared memory)
        """
        da = gpuarray.to_gpu(a_cpu);
        db = gpuarray.to_gpu(b_cpu);
        height_c = a_cpu.shape[0]
        width_c = b_cpu.shape[1]
        dc = gpuarray.empty((np.int32(height_c), np.int32(width_c)), np.dtype(np.float32))
        _start = cuda.Event()
        _end = cuda.Event()        
        func = self.prg_opt2.get_function("Matrix_multiply_optimized2")
        _start.record()
        func(da, db, dc, np.int32(height_c), np.int32(a_cpu.shape[1]), block=self.block, grid=self.grid)
        _end.record()
        _end.synchronize()
        c_optimized2 = dc.get()
        time_optimized2 = _start.time_till(_end) # in ns
        return c_optimized2, time_optimized2

if __name__ == '__main__':
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
    height = random.randint(1, 10)
    width = random.randint(1, 10)
    matrix_factor = 1
    time_naive = np.ones((5, matrix_factor))
    time_optimized1 = np.ones((5, matrix_factor))
    time_optimized2 = np.ones((5, matrix_factor))
    time_cpu = np.ones((5, matrix_factor))
    # 5, 10, 50, 100, 500, 1000, 5000, 10000,
    # 2^i (2^10: 1024)
    print("The base matrix size is (", height,", ", width, ")\n")
    for p in range(matrix_factor):
        print("the matrix size factor is ",2**p)
        for i in range(1):
            a_cpu = np.random.rand(height*(2**p), width*(2**p)).astype(np.float32)
            b_cpu = np.random.rand(width*(2**p), height*(2**p)).astype(np.float32)
            print(a_cpu)
            print("=====================")
            print(b_cpu)
            Mmul_obj = MatrixMultiply(height*(2**p),width*(2**p))
            out_naive, time_naive[i][p] = Mmul_obj.matrix_mul_naive(a_cpu, b_cpu)
#            out_optimized1, time_optimized1[i][p] = Mmul_obj.matrix_mul_optimized1(a_cpu, b_cpu)
#            out_optimized2, time_optimized2[i][p] = Mmul_obj.matrix_mul_optimized2(a_cpu, b_cpu)
            start = time.time()
            out_cpu = np.matmul(a_cpu, b_cpu)
            end = time.time()
            time_cpu[i][p] = (end - start)*1e+9 # in ns
            try:
                assert (np.all(out_naive == out_cpu))
#                assert (np.all(out_optimized1 == out_cpu))
#                assert (np.all(out_optimized2 == out_cpu))
            except AssertionError:
                print("\nCheckpoint failed: GPU results doesn't match CPU result!\n")
                print(out_cpu)
                print(out_naive)
#            print(out_optimized1)
#            print(out_optimized2)
"""
    avg_t_naive = np.average(time_naive, axis=0)
    avg_t_optimized1 = np.average(time_optimized1, axis=0)
    avg_t_optimized2 = np.average(time_optimized2, axis=0)
    avg_t_cpu = np.average(time_cpu, axis=0)

    print(avg_t_cpu.shape)
    print(avg_t_cpu)
    # Execution time
    factor = []
    for i in range(matrix_factor):
        factor.append(2**i)
    print(factor)
        
    print("\n============Exec time of each matrix size for each run =============")
#    print("{0:<12}{1:<12}  {2:<12}   {3:<16} {4:<16} {5:<16}".format("run NO.","matrix_size", "time_cpu(ns)", "time_naive(ns)", "time_opt1(ns)", "time_opt2(ns)"))
    
 #   for r in range(5):
 #       for i in range(matrix_factor):
 #           print("{0:<12}{1:<12}  {2:<12}     {3:.6f}         {4:.6f}   {5:.6f}".format(r, factor[i], time_cpu[r][i], time_naive[r][i], time_optimized1[r][i]), time_optimized2[r][i]))
    print("-------------------------------------------------------------------------")
    print("\n=========Exec time of each matrix size average over 5 run ==========")
    print("{0:<12}  {1:<12}   {2:<16} {3:<16}  {4:<16}".format("matrix_size", "time_cpu(ns)", "time_naive(ns)", "time_opt1(ns)", "time_opt2(ns)"))
#    for i in range(matrix_factor):
#       print('{0:<12}  {1:<12}     {2:.6f}         {3:.6f}  {4:.6f}'.format(factor[i], ave_t_cpu[i], avg_t_naive[i], avg_t_optimized1[i]), avg_t_optimized2[i]))


    
    x = np.linspace(1,matrix_factor,matrix_factor)
    fig, ax = plt.subplots()
    line1, = ax.plot(x, avg_t_cpu, label='CPU')
    line2, = ax.plot(x, avg_t_naive, label = 'GPU-naive')
    line3, = ax.plot(x, avg_t_optimized1, label = 'GPU-optimized1')
    line4, = ax.plot(x, avg_t_optimized2, label = 'GPU-optimized2')

    ax.set_title('Execution Time comparison')
    ax.set_xlabel('factor of matrix size')
    ax.set_ylabel('Time (us) -- Log scale')
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlim(0, matrix_factor + 1)
    ax.xaxis.set_ticks(np.arange(0,matrix_factor + 1, 2))
    ax.legend()                
    plt.savefig("plots/HW3_mmul_time_comparison.png")
                                                                                        
"""
