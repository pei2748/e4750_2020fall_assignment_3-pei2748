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
    def __init__(self, height):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # host variables
        
        # tile parameters
        self.tile_width = 16
        self.block = (self.tile_width, self.tile_width, 1)
        m = height
        self.grid = (math.floor((m-1)/self.tile_width) + 1, math.floor((m-1)/self.tile_width) + 1, 1)

        
        matrix_mul_naive_kernel_code = """
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
            if(Row < M && Col < N) {
                tmp = 0;
                for(i = 0; i < N; i++)  
                     tmp += A[Row * N + i] * B[Col + i * M];
                C[Row * M + Col] = tmp;
            } // end of if
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
            for(int t = 0; t < N/TILE_WIDTH; ++t) {
                //ds_A[ty][tx] = A[Row * N + t * TILE_WIDTH + tx];
                ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * M + Col];
                __syncthreads();
                for (int i = 0; i < TILE_WIDTH; ++i)
                     tmp += A[Row * N + t * TILE_WIDTH + tx + i] * ds_B[i][tx];
                __syncthreads();
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
    height = random.randint(1, 10) * 1000
    width = random.randint(1, 10) * 1000
    a_cpu = np.random.rand(height, width).astype(np.float32)
    b_cpu = np.random.rand(width, height).astype(np.float32)

    Mmul_obj = MatrixMultiply(height)
    out_naive, time_naive = Mmul_obj.matrix_mul_naive(a_cpu, b_cpu)
    out_optimized1, time_optimized1 = Mmul_obj.matrix_mul_optimized1(a_cpu, b_cpu)
    out_optimized2, time_optimized2 = Mmul_obj.matrix_mul_optimized2(a_cpu, b_cpu)

    print(time_naive, time_optimized1, time_optimized2)
    
