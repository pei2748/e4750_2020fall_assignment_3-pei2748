import <packages>


class Transpose:
    def __init__(self, x_cpu):
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables
        self.x = x_cpu

        # kernel code for transpose
        self.transpose_kernel_code = """
        """

        # build kernel
        


    def transpose_parallel(self):
        """
        Function to perform transpose on array using PyOpenCL.
        Should return array transpose and execution time.
        """
        # device memory allocation

        # call function and time it

        return self.y_gpu.get(), np.average(timing)



    def transpose_serial(self):
        # Serial code to transpose the matrix
        
        return result, np.average(timing)

class alpha:
    def __init__(self, a_cpu, b_cpu):
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables

        self.alpha = # value for alpha
        
        # kernel code for blending
        self.blend_kernel_code = """

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
