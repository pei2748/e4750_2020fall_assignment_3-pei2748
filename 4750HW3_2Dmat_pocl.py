import numpy as np
import random
import pyopencl as cl
import pyopencl.array as pocl_array
import matplotlib.pyplot as plt
import time


class Transpose:
    def __init__(self, x_cpu):
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(
            self.ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables
        self.x = x_cpu
        self.height = x_cpu.shape[0] # number of rows
#        print(self.height)
        self.width = x_cpu.shape[1] # number of columns
#        print(self.width)
        self.y_gpu = np.empty([self.width, self.height])
# kernel code for transpose
        self.transpose_kernel_code = """
        __kernel void transpose(__global float *out, 
        __global float *in, const int height, const int width)
        {
             int tx = get_global_id(0);
             int ty = get_global_id(1);
             int idx_in = ty * width + tx;
             int idx_out = tx * height + ty;
             out[idx_out] = in[idx_in];
        }
        """

        # build kernel
        self.prg = cl.Program(self.ctx, self.transpose_kernel_code).build()


    def transpose_parallel(self):
        """
        Function to perform transpose on array using PyOpenCL.
        Should return array transpose and execution time.
        """
        # device memory allocation
        d_x = pocl_array.to_device(self.queue, self.x)
        y_gpu = pocl_array.empty(self.queue, (np.int32(self.width), np.int32(self.height)), np.dtype(np.float32))

        # call function and time it
        event = self.prg.transpose(self.queue,
                                   (np.int32(self.width), np.int32(self.height)),
                                   None,
                                   y_gpu.data,
                                   d_x.data,
                                   np.int32(self.height),
                                   np.int32(self.width))
        event.wait()

        time_ = event.profile.end - event.profile.start
        
        return y_gpu.get(), time_
    #np.average(timing)
    
 def transpose_serial(self):
     # Serial code to transpose the matrix
     start_ = time.time()
     for i in range(self.width):
         for j in range(self.height):
             self.y_gpu[i][j] = self.x[j][i]
     end_ = time.time()

     return self.y_gpu,(end_-start_)*1e+3

class alpha:
    def __init__(self, a_cpu, b_cpu):
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(
            self.ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables

        self.alpha = random.randint(1,10)/10# value for alpha
     
        # kernel code for blending
        self.blend_kernel_code =
        """
        __kernel void blend(__global *out, 
        __global float *a, __global float *b, 
        const int length, const float alpha)
        {
             int tx = get_global_id(0);
             int ty = get_global_id(1);
             int idx = tx * length + ty;
             out[idx] = alpha * a[idx] + (1 - alpha) * b[idx];

        }

        """

        # build kernel
        

    
    def alpha_blend(self):
        """
            Function to perform alpha blending. Should return blended array 
            and execution time.
        """
        # device memory allocation
            
        # call kernel function, time execution
        
        return self.c_gpu.get(), exec_time

        
    

if __name__ == "__main__":

    # Main code

    height = random.randint(1, 10)
    width = random.randint(1, 10)
    data_in = np.random.rand(height, width).astype(np.float32)
    print(data_in)
    Trans_Obj = Transpose(data_in)
    opencl_result = Trans_Obj.transpose_parallel()
    print(opencl_result[0])
    
    ##### image blend #####

    im_1 = Image.open('image_1.jpg')
    im_2 = Image.open('image_2.jpg')
    im1_gray = ImageOps.grayscale(im_1)
    im2_gray = ImageOps.grayscale(im_2)
    im1_gray_array = np.array(im1_gray) # (500, 500)
    im2_gray_array = np.array(im2_gray) # (500, 500)

    Blend_Obj = alpha(im1_gray_array, im2_gray_array)

    
    


    

    
