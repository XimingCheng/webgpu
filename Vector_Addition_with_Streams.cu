#include	<wb.h>

#define SegSize 1024

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < len)
		out[idx] = in1[idx] + in2[idx];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;
	
	float * deviceInputA0;
	float * deviceInputA1;
	float * deviceInputB0;
	float * deviceInputB1;
	
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

	cudaMalloc((void**)&deviceOutput, 2 * SegSize * sizeof(float));
	cudaMalloc((void**)&deviceInputA0, SegSize * sizeof(float));
	cudaMalloc((void**)&deviceInputA1, SegSize * sizeof(float));
	cudaMalloc((void**)&deviceInputB0, SegSize * sizeof(float));
	cudaMalloc((void**)&deviceInputB1, SegSize * sizeof(float));
	
	for (int i = 0; i < inputLength; i += SegSize * 2) {
		
		cudaMemcpyAsync(deviceInputA0, hostInput1 + i, SegSize * sizeof(float),
						cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(deviceInputB0, hostInput2 + i, SegSize * sizeof(float),
						cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(deviceInputA1, hostInput1 + i + SegSize, SegSize * sizeof(float),
						cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(deviceInputB1, hostInput2 + i + SegSize, SegSize * sizeof(float),
						cudaMemcpyHostToDevice, stream1);
		vecAdd <<< (SegSize - 1)/256 + 1, 256, 0, stream0 >>> (deviceInputA0, deviceInputB0, deviceOutput, SegSize);
		vecAdd <<< (SegSize - 1)/256 + 1, 256, 0, stream1 >>> (deviceInputA1, deviceInputB1, deviceOutput + SegSize, SegSize);
		cudaMemcpyAsync(hostOutput + i, deviceOutput, SegSize * sizeof(float),
						cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput + i + SegSize, deviceOutput + SegSize, SegSize * sizeof(float),
						cudaMemcpyDeviceToHost, stream1);
	}

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
	cudaFree(deviceInputA0);
	cudaFree(deviceInputA1);
	cudaFree(deviceInputB0);
	cudaFree(deviceInputB1);

    return 0;
}

