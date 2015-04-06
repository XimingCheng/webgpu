// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan(float * input, float * output, float * post, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	__shared__ float XY[BLOCK_SIZE << 1];
	unsigned int t = threadIdx.x;
	unsigned int start = (BLOCK_SIZE << 1) * blockIdx.x;
	
	if (start + t < len)
		XY[t] = input[start + t];
	else
		XY[t] = 0.0f;
	if (start + BLOCK_SIZE + t < len)
		XY[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
	else
		XY[BLOCK_SIZE + t] = 0.0f;
	__syncthreads();
	
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		int index = (t + 1)*stride*2 - 1;
		if (index < (BLOCK_SIZE << 1))
			XY[index] += XY[index - stride];
		__syncthreads();
	}
	
	for (int stride = BLOCK_SIZE/2; stride >= 1; stride /= 2)
	{
		__syncthreads();
		int index = (t + 1)*stride*2 - 1;
		if (index + stride < (BLOCK_SIZE << 1))
			XY[index + stride] += XY[index];
		
	}
	__syncthreads();
	if (start + t < len)
		output[start + t] = XY[t];
	if (start + BLOCK_SIZE + t < len)
		output[start + BLOCK_SIZE + t] = XY[t + BLOCK_SIZE];
	
	if (post && t == 0)
		post[blockIdx.x] = output[(BLOCK_SIZE << 1) * (blockIdx.x + 1) - 1];
	
}

__global__ void postScan(float * output, float * post, int len) {
	unsigned int t = threadIdx.x;
	unsigned int start = (BLOCK_SIZE << 1) * (blockIdx.x + 1);
	if (start + t < len)
		output[start + t] += post[blockIdx.x];
	if (start + t + BLOCK_SIZE < len)
		output[start + t + BLOCK_SIZE] += post[blockIdx.x];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
	float * post = NULL;
	float * post_scan;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

	int blocks = (numElements - 1)/(BLOCK_SIZE << 1) + 1;
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
	if (blocks > 1) {
		wbCheck(cudaMalloc((void**)&post, blocks*sizeof(float)));
		wbCheck(cudaMalloc((void**)&post_scan, blocks*sizeof(float)));
	}
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	scan <<< dimGrid, dimBlock >>> (deviceInput, deviceOutput, post, numElements);
	
	if (blocks > 1) {
		dim3 dimBlock1(BLOCK_SIZE, 1, 1);
		dim3 dimGrid1(1, 1, 1);
		scan <<< dimGrid1, dimBlock1 >>> (post, post_scan, NULL, blocks);
		
		dim3 dimBlock2(BLOCK_SIZE, 1, 1);
		dim3 dimGrid2(blocks - 1, 1, 1);
		postScan <<< dimGrid2, dimBlock2 >>> (deviceOutput, post_scan, numElements);
	}

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
	if (post) {
		cudaFree(post);
		cudaFree(post_scan);
	}
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
