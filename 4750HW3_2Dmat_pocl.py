import numpy as np
import random
import pyopencl as cl
import pyopencl.array as pocl_array
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageOps

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
        time_ = event.profile.end - event.profile.start # in ns
        
        return y_gpu.get(), time_*1e-3 #return in us
    #np.average(timing)
    
    def transpose_serial(self):
        # Serial code to transpose the matrix
        start_ = time.time()
        for i in range(self.width):
            for j in range(self.height):
                self.y_gpu[i][j] = self.x[j][i]

        end_ = time.time()
        return self.y_gpu,(end_-start_)*1e+6 # in us.

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
        self.a = a_cpu
        self.b = b_cpu
        self.length = a_cpu.shape[0]
        self.alpha = random.randint(1,10)/10# value for alpha
     
        # kernel code for blending
        self.blend_kernel_code = """
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
        self.prg = cl.Program(self.ctx, self.blend_kernel_code).build()

    
    def alpha_blend(self):
        """
            Function to perform alpha blending. Should return blended array 
            and execution time.
        """
        # device memory allocation
        d_a = pocl_array.to_device(self.queue, self.a)
        d_b = pocl_array.to_device(self.queue, self.b)
        d_out = pocl_array.empty_like(d_a)
        
        # call kernel function, time execution
        event = self.prg.blend(self.queue, self.a.shape, None, d_out.data, d_a.data, d_b.data, np.int32(self.length), np.float32(self.alpha))
        event.wait()
        exec_time = event.profile.end - event.profile.start
        
        return d_out.get(), exec_time

        
    

if __name__ == "__main__":

    #### matrix transpose #####

    height = random.randint(1, 10)
    width = random.randint(1, 10)
    num_matrix = 10
    print("matrix transpose operation in parallel and in serial:\n\n")
    print("base matrix size is (", height, ", ", width, ")\n")
    
    time_cl_trans = np.ones((5, num_matrix))
    time_py_trans = np.ones((5, num_matrix))

    for n in range(5):
        for i in range(num_matrix):
            data_in = np.random.rand(height*(i+1), width*(i+1)).astype(np.float32)
            Trans_Obj = Transpose(data_in)
            opencl_out, time_cl_trans[n][i] = Trans_Obj.transpose_parallel()
            py_out, time_py_trans[n][i] = Trans_Obj.transpose_serial()
            
    # Error check
    try:
        print("\nCheckpoint: Do python and kernel matrix transpose match? Checking...")
        assert (np.all(opencl_out == py_out))
    except AssertionError:
        print("\nCheckpoint failed: Python and OpenCL kernel matrix transpose do not match. Try Again#!")
    print("match!!\n")
                           
            
    avg_time_cl_trans = time_cl_trans.sum(axis=0)/5 # us
    avg_time_py_trans = time_py_trans.sum(axis=0)/5 # us

    print("{0:<8},{1:>20},{2:>20}".format("size_factor", "Exec_t_parallel", "Exec_t_serial"))
    for i in range(num_matrix):
        print('{0:<14},{1:>20},{2:>20}'.format(i, avg_time_cl_trans[i], avg_time_py_trans[i]))
        x = np.linspace(1,num_matrix,num_matrix)
        fig, ax = plt.subplots()
        line1, = ax.plot(x, avg_time_cl_trans, label='pyopencl_parallel')
        line2, = ax.plot(x, avg_time_py_trans, label = 'python_serial')
        ax.set_title('Execution Time comparison of parallel and serial')
        ax.set_xlabel('factor of matrix size')
        ax.set_ylabel('Time (us) -- Log scale')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlim(0,num_matrix + 1)
        ax.xaxis.set_ticks(np.arange(0,num_matrix + 1, 2))
        ax.legend()                
        plt.savefig("plots/HW3_transpose_time_comparison.png")
                
    
    ##### image blend #####

    im_1 = Image.open('image_1.jpg')
    im_2 = Image.open('image_2.jpg')
    im1_gray = ImageOps.grayscale(im_1)
    im2_gray = ImageOps.grayscale(im_2)
    im1_gray_array = np.array(im1_gray) # (500, 500)
    im2_gray_array = np.array(im2_gray) # (500, 500)

    Trans_Obj = Transpose(im1_gray_array)
    trans_im1_gray_array, time_trans = Trans_Obj.transpose_parallel()

    
    
    Blend_Obj = alpha(trans_im1_gray_array, im2_gray_array)
    blend_result, blend_time  = Blend_Obj.alpha_blend()


    pil_img = Image.fromarray(blend_result)
    print(pil_img.mode)
    # RGB
    new_img = pil_img.convert('RGB')
    print(new_img.mode)

    new_img.save('plots/alpha_blending.jpg')
    
    


    

    
