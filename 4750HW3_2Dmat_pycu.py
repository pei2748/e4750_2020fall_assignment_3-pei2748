import time
import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

class MatrixMultiply:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # host variables
        
        # tile parameters
        self.tile_x = # tile size x
        self.tile_y = # when changing this, remember to change it in kernel as well!
        
        matrix_mul_naive_kernel_code = """
        __global__ void Matrix_multiply_naive(float *a, float *b, float *c,
        const unsigned int X, const unsigned int Y)
        {
            // Your kernel code here
        }
        """

        matrix_mul_optimized1_kernel_code = """
        __global__ void Matrix_multiply_optimized1(float *a, float *b, float *c,
        const unsigned int X, const unsigned int Y)
        {

            //'B' tiles are shared. 'A' tiles are in global memory
            
        }
        """

        matrix_mul_optimized2_kernel_code = """
        __global__ void Matrix_multiply_optimized2(float *a, float *b, float *c,
        const unsigned int X, const unsigned int Y)
        {
            // Your kernel code here
        }
        """
        # Build kernel codes (x3)
        

    def matrix_mul_naive(self, a_cpu, b_cpu):
        """
        Function to perform matrix multipication. Should return the result
        and execution time.
        """
        
        
        return c_naive, time_naive

    def matrix_mul_optimized1(self, a_cpu, b_cpu):
        """
        Function to perform partially optimized matrix multipication. 
        Should return the result and execution time.
        (Only B tiled in shared memory)
        """
        return c_optimized1, time_optimized1

    def matrix_mul_optimized2(self, a_cpu, b_cpu):
        """
        Function to perform optimized matrix multiplication using shared
        memory. Should return the result and execution time.
        (A and B both tiled in shared memory)
        """
        return c_optimized2, time_optimized2

if __name__ == '__main__':
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
