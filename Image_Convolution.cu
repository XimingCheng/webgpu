#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius (Mask_width/2)

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + 2*Mask_radius)

//@@ INSERT CODE HERE
__global__ void imageConvolution(float* deviceInputImageData, float* deviceOutputImageData,
						   int imageWidth, int imageHeight, int channels, int maskRows, int maskColumns,
						   const float* __restrict__ deviceMaskData) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;

	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;
	
	__shared__ float NsR[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float NsG[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ float NsB[BLOCK_WIDTH][BLOCK_WIDTH];
	
	if (row_i >= 0 && row_i < imageHeight && col_i >= 0 && col_i < imageWidth)
	{
		int baseIdx = (row_i*imageWidth + col_i)*channels;
		NsR[ty][tx] = deviceInputImageData[baseIdx];
		NsG[ty][tx] = deviceInputImageData[baseIdx + 1];
		NsB[ty][tx] = deviceInputImageData[baseIdx + 2];
	}
	else
	{
		NsR[ty][tx] = 0.0f;
		NsG[ty][tx] = 0.0f;
		NsB[ty][tx] = 0.0f;
	}
	
	__syncthreads();
	
	float outputR = 0.0f;
	float outputG = 0.0f;
	float outputB = 0.0f;
	
	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
	{
		for (int i = 0; i < maskColumns; i++)
		{
			for (int j = 0; j < maskRows; j++)
			{
				float mask = deviceMaskData[i*maskColumns + j];
				outputR += mask*NsR[i + ty][j + tx];
				outputG += mask*NsG[i + ty][j + tx];
				outputB += mask*NsB[i + ty][j + tx];
			}
		}
		
		if (row_o < imageHeight && col_o < imageWidth)
		{
			int baseIdx = (row_o*imageWidth + col_o)*channels;
			deviceOutputImageData[baseIdx] = outputR;
			deviceOutputImageData[baseIdx + 1] = outputG;
			deviceOutputImageData[baseIdx + 2] = outputB;
		}
	}

}


int main(int argc, char* argv[]) {
    wbArg_t args;
	// the mask size data
    int maskRows;
    int maskColumns;
	// input image size and channel data
    int imageChannels;
    int imageWidth;
    int imageHeight;
	// input files
    char * inputImageFile;
    char * inputMaskFile;
	// input and output image object
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGrid((imageWidth - 1)/O_TILE_WIDTH + 1, (imageHeight - 1)/O_TILE_WIDTH + 1, 1);
	imageConvolution <<< dimGrid, dimBlock >>> (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight,
											   imageChannels, maskRows, maskColumns, deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
