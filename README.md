webgpu
======

Code Passed all Unit Test provided by [WebGPU](http://webgpu.com/)
The Courser is provided by Coursera, [Heterogeneous Parallel Programming](https://www.coursera.org/course/hetero) by [Wen-mei W. Hwu](https://www.coursera.org/instructor/wen-meihwu) [University of Illinois at Urbana-Champaign](https://www.coursera.org/illinois)

* [Basic_Vector_Addition](Basic_Vector_Addition.cu)
    * The "hello world" of CUDA
* [Basic_Matrix_Multiplication](Basic_Matrix_Multiplication.cu)
    * Basic A * B Matrix Multiplication (none shared memory)
* [Tiled_Matrix_Multiplication](Tiled_Matrix_Multiplication.cu)
    * A * B Matrix Multiplication with Tiled memory (shared memory)
* [Image_Convolution](Image_Convolution.cu)
    * Image Template Block pixels Convolution
* [List_Reduction](List_Reduction.cu)
    * Given a list (lst) of length n
    * Output its sum = lst[0] + lst[1] + ... + lst[n-1];
* [List_Scan](List_Scan.cu)
    * Given a list (lst) of length n
    * Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
* [Histogram](Histogram.cu)
    * Image Histogram Equalization
* [Vector_Addition_with_Streams](Vector_Addition_with_Streams.cu)
    * CUDA streaming API vector addition
* [OpenCL_Vecton_Addition](OpenCL_Vecton_Addition.cpp)
    * "hello world" of OpenCL
* [OpenACC_Vector_Addition](OpenACC_Vector_Addition.cpp)
    * "hello world" of OpenACC
* [C++AMP_Vector_Addition](C++AMP_Vector_Addition.cpp)
    * "hello world" of C++ AMP

