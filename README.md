# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2020)

## Assignment-3: Matrix Transpose and Tiled Multiplication

Due date: Tentatively 1st November, 2020 at *11:59 pm.*

Total points: 100

### Primer

The goal of this assignment is to introduce complexity to the 2D array manipulation kernel in OpenCL and CUDA through two common matrix operations -- transpose and multiplication. Your tasks will be to: 

i. Compare 3 kernels for matrix multiplication using PyCUDA through plots and profiling. 

ii. Compare the execution speeds for matrix transpose for Python and PyOpenCL.

iii. Perform alpha blending of two images using PyOpenCL by using a 2D matrix kernel.


### Relevant Documentation

The matrix multiplication kernels have been introduced and explained in the lectures, and the relevant references can be found in the lecture slides. Having completed assignment-2, you should be comfortable using NVVP to profile your PyCUDA code. 

### Alpha Blending:

In Image Processing, it is sometimes desirable to combine two images with precise control over how much one image's features will dominate over the other. Mathematically, this can be explained as follows:

If you have two images *A* and *B*, then to blend them with a weight factor `alpha`, you perform the operation

```
I_blend = alpha*A + (1 - alpha)*B
```

Where the values `alpha` and `1-alpha` are multiplied to every element of *A* and  *B* respectively. 

In essence, in this task you are expected to perform a kind of weighted matrix addition.


## Report (5 points)

Prepare a markdown report to go with your code submission. A demo report markdown has been provided with this assignment, with detailed instructions on how to populate it, along with some useful markdown syntax if you are unfamiliar with its use. 

The GitHub wiki has also been updated with instructions for the homework report. You may check the sidebar to look for them.

## Programming Problem (80 points)

The tasks for this assignment split three ways. For PyCUDA, your task is to implement three kernels for matrix multiplication, two of which shall use tiling in shared memory. 

For PyOpenCL, there are two tasks:
1. Task-1: Matrix Transpose
2. Task-2: Image Alpha Blending
 

### Task-1: Matrix Transpose with PyOpenCL (20 points)
1. *(5 points)* Implement a serial transpose algorithm. The input is a 2D matrix of any size, the output is the transpose of the input.
2. *(5 points)* Implement a parallel transpose algorithm PyOpenCL. The input is a 2D matrix of any size, the output is the transpose of the input.

3. *(5 points)* Choose any two integers M and N from 1 to 10. Then, randomly generate matrices with sizes M x N, 2M x 2N, 3M x 3N,.... Calculate their transpose using the two transpose algorithms (parallel v. serial) respectively. Record the *average* running time of each call for each of the algorithm.

4. *(5 points)* Plot running time vs. matrix size in one figure, compare and analyze the results in your report.

### Task-2: Image Alpha Blending with PyOpenCL (15 points)

**Note:** You will have to read the image files. You may use the PIL library to read the image, and then convert it to a numpy array as follows:

```python
import numpy as np

im = np.array(Image.open('data/src/lena.jpg'))

print(type(im))
# <class 'numpy.ndarray'>

print(im.dtype)
# uint8

print(im.shape)
# (225, 400, 3)
```

Note that most standard images are not 2D arrays, but three 2D arrays stacked together. *You need to use either PIL, numpy/scipy or skimage to convert the image to a grayscale image.*

You can turn a numpy array back to a PIL Image object and then write it to file like so:

```python
pil_img = Image.fromarray(im)
print(pil_img.mode)
# RGB

pil_img.save('data/temp/lena_save_pillow.jpg')

```


1. *(10 points)* Write a kernel to blend two input images of the same size. You may choose any alpha value between 0.0 and 1.0. 
2. *(5 points)* Use your matrix transpose kernel to first tranpose `image_1.png`, and then blend the transposed image with `image_2.png`.

### Task-3: Comparing Tiled Multiplication in Shared Memory with PyCUDA (45 points)

The input should be two matrices, one of size `M x N`, and the other of size `N X M`. The output is the multiplication of the input matrices. The result should be a `M x M` matrix.

1. *(5 points)* Write a kernel to perform naive 2D matrix multiplication in global memory (i.e. the regular method, without tiliing. Refer to the lecture slides.)
2. *(10 points)* Write a kernel to perform partially tiled matrix multiplication where matrix *A* is in global memory, but *B*'s tiles are in shared memory.
3. *(10 points)* Write a kernel to perform tiled matrix multiplication in shared memory.
4. *(10 points)* Iteratively increase the size of the matrices and record the CPU, and kernel execution time for all three kernel functions. Plot the average running times for all 3 in a single figure, and include it in your report. Use the log-scale for this plot.
5. *(10 points)* Use the Nvidia Visual Profiler (nvvp) to compare your 3 algorithms written in CUDA. You must include your analysis with screenshots in the report.


## Theory Problems (15 points)

1. *(2 points)* For our tiled matrix-matrix multiplication kernel, if we use a 32X32 tile, what is the reduction of memory bandwidth usage for input matrices A and B?

* a. 1/8 of the original usage
* b. 1/16 of the original usage 
* c. 1/32 of the original usage 
* d. 1/64 of the original usage

2. *(2 points)* For the tiled single-precision matrix multiplication kernel as shown in one of the lectures, assume that the tile size is 32X32 and the system has a DRAM burst size of 128 bytes. How many DRAM bursts will be delivered to the processor as a result of loading one A-matrix tile by a thread block?
* a. 16
* b. 32 
* c. 64 
* d. 128

3. *(3 points)* For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel launch to have a minimal number of thread blocks to cover all output elements. How many threads will be in the grid? 

4. *(5 points)* What is a warp in CUDA?

5. *(3 points)* How does the term *local memory* differ for CUDA and OpenCL? What is the CUDA analog to OpenCL's local memory? How is it different from CUDA's local memory?

