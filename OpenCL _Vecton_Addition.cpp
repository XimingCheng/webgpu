#include <wb.h> //@@ wb include opencl.h for you

//@@ OpenCL Kernel

const char *kernelSource =                                       "\n" \
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
                                                                "\n" ;
	
int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
 
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
  

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
	for (int i = 0; i < 3; i++) {
		wbLog(TRACE, "hostInput1 ", (int)100*hostInput1[i]);
	}
	for (int i = 0; i < 3; i++) {
		wbLog(TRACE, "hostInput2 ", (int)100*hostInput2[i]);
	}

  wbLog(TRACE, "The input length is ", inputLength);
  size_t bytes = inputLength*sizeof(float);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be devisor
    globalSize = (inputLength - 1)/localSize + 1;
	globalSize = inputLength;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clGetPlatformIDs failed");
	}
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clGetDeviceIDs");
	}
 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clCreateContext");
	}
 
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clCreateCommandQueue");
	}
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
 
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clCreateProgramWithSource");
	}
    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clBuildProgram");
	}
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
 
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, hostInput1, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, hostInput2, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clEnqueueWriteBuffer");
	}
 
    // Set the arguments to our compute kernel
	  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_a);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_b);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_c);
  err = clSetKernelArg(kernel, 3, sizeof(int), &inputLength);
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clSetKernelArg");
	}
 
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL,
                                                              0, NULL, NULL);
	if (err != CL_SUCCESS) {
		wbLog(TRACE, "clEnqueueNDRangeKernel");
	}
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, hostOutput, 0, NULL, NULL );
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, inputLength * sizeof(float), hostOutput,
					 0, NULL, NULL);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
