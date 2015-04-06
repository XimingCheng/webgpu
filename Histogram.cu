// Histogram Equalization
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


#include    <wb.h>

#define DEBUG_MODE 0
#define BLOCK_SIZE 512
#define CDF_BLOCK_SIZE 128
#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void castImageToUchar(float* deviceInputImageData, unsigned char* ucharImage,
								 int imageWidth, int imageHeight, int channels, int pixelSize) {
	int w = threadIdx.x + blockDim.x * blockIdx.x;
	if (w < pixelSize)
		ucharImage[w] = (unsigned char) (255 * deviceInputImageData[w]);
}

__global__ void castImageTofloat(float* deviceOutputImageData, unsigned char* ucharImage,
								 int imageWidth, int imageHeight, int channels, int pixelSize) {
	int w = threadIdx.x + blockDim.x * blockIdx.x;
	if (w < pixelSize)
		deviceOutputImageData[w] = (float) (ucharImage[w]/255.0);
}

__global__ void castImageToGrayScale(unsigned char* ucharImage, unsigned char* grayImage,
									 int imageWidth, int imageHeight, int channels) {
	int w = threadIdx.x + blockDim.x * blockIdx.x;
	int h = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = imageWidth * h + w;
	if (w < imageWidth && h < imageHeight) {
		unsigned char r = ucharImage[idx * channels];
		unsigned char g = ucharImage[idx * channels + 1];
		unsigned char b = ucharImage[idx * channels + 2];
		grayImage[idx] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b);
	}
}

__global__ void calHistogram(unsigned char* grayImage, int size, unsigned int* histogram) {
	__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
	int t = threadIdx.x;
	if (t < HISTOGRAM_LENGTH)
		histo_private[t] = 0;
	__syncthreads();
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (idx < size) {
		atomicAdd(&histo_private[grayImage[idx]], 1);
		idx += stride;
	}
	__syncthreads();
	
	if (t < HISTOGRAM_LENGTH)
		atomicAdd(&histogram[t], histo_private[t]);
}

__global__ void calCdf(unsigned int* histogram, float* cdf, int size) {
	__shared__ float XY[CDF_BLOCK_SIZE << 1];
	unsigned int t = threadIdx.x;
	unsigned int start = (CDF_BLOCK_SIZE << 1) * blockIdx.x;
	
	if (start + t < HISTOGRAM_LENGTH)
		XY[t] = histogram[start + t]/(float)size;
	else
		XY[t] = 0.0f;
	if (start + CDF_BLOCK_SIZE + t < HISTOGRAM_LENGTH)
		XY[CDF_BLOCK_SIZE + t] = histogram[start + CDF_BLOCK_SIZE + t]/(float)size;
	else
		XY[CDF_BLOCK_SIZE + t] = 0.0f;
	__syncthreads();
	
	for (int stride = 1; stride <= CDF_BLOCK_SIZE; stride *= 2)
	{
		int index = (t + 1)*stride*2 - 1;
		if (index < (CDF_BLOCK_SIZE << 1))
			XY[index] += XY[index - stride];
		__syncthreads();
	}
	
	for (int stride = CDF_BLOCK_SIZE/2; stride >= 1; stride /= 2)
	{
		
		int index = (t + 1)*stride*2 - 1;
		if (index + stride < (CDF_BLOCK_SIZE << 1))
			XY[index + stride] += XY[index];
		__syncthreads();
	}
	
	if (start + t < HISTOGRAM_LENGTH)
		cdf[start + t] = XY[t];
	if (start + CDF_BLOCK_SIZE + t < HISTOGRAM_LENGTH)
		cdf[start + CDF_BLOCK_SIZE + t] = XY[t + CDF_BLOCK_SIZE];
}

__global__ void calMinCdf(float* cdf, float* mincdf) {
	// should change into list reduction
	if (threadIdx.x == 0)
		mincdf[0] = cdf[0];
}

__global__ void equalization(float* cdf, float* mincdf, unsigned char* ucharImage,
							int imageWidth, int imageHeight, int channels, int pixelSize) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < pixelSize) {
		unsigned char val = ucharImage[idx];
		float data = 255*(cdf[val] - mincdf[0])/(1 - mincdf[0]);
		if (data < 0.0f) data = 0.0f;
		else if (data > 255.0f) data = 255.0f;
		ucharImage[idx] = (unsigned char) data;
	}
}
 
int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	float * deviceInputImageData;
    float * deviceOutputImageData;
	unsigned char* ucharImage;
	unsigned char* grayImage;
	unsigned int* histogram;
	float* cdf;
	float* mincdf;
	
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	int pixelSize = imageWidth * imageHeight * imageChannels;

	wbCheck(cudaMalloc((void **) &deviceInputImageData, pixelSize * sizeof(float)));
	wbCheck(cudaMalloc((void **) &deviceOutputImageData, pixelSize * sizeof(float)));
	wbCheck(cudaMalloc((void **) &ucharImage, pixelSize * sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **) &grayImage, imageWidth * imageHeight * sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **) &histogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **) &cdf, HISTOGRAM_LENGTH * sizeof(float)));
	wbCheck(cudaMalloc((void **) &mincdf, sizeof(float)));

#if DEBUG_MODE
	float* debug_f = (float*)malloc(pixelSize * sizeof(float));
	unsigned char* debug_c = (unsigned char*)malloc(pixelSize * sizeof(unsigned char));
	int* debug_i = (int*)malloc(pixelSize * sizeof(int));
	wbLog(TRACE, "The value of imageWidth = ", imageWidth);
	wbLog(TRACE, "The value of imageHeight = ", imageHeight);
#endif
	
	wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, pixelSize * sizeof(float), cudaMemcpyHostToDevice));
#if DEBUG_MODE
	//for (int i = 0; i < 5; i++) {
	//	wbLog(TRACE, "The value of deviceInputImageData = ", (int)(hostInputImageData[i]*255));
	//}
#endif
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((pixelSize - 1)/BLOCK_SIZE + 1, 1, 1);
	castImageToUchar <<< dimGrid, dimBlock >>> (deviceInputImageData, ucharImage, imageWidth, imageHeight,
												imageChannels, pixelSize);
	cudaDeviceSynchronize();
#if DEBUG_MODE
	//wbCheck(cudaMemcpy(debug_c, ucharImage, pixelSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//for (int i = 0;i < 5; i++) {
	//	wbLog(TRACE, "The value of ucharImage = ", (int)debug_c[i]);
	//}
#endif
	
	dim3 dimBlock1(32, 32, 1);
	dim3 dimGrid1((imageWidth - 1)/32 + 1, (imageHeight - 1)/32 + 1, 1);
	castImageToGrayScale <<< dimGrid1, dimBlock1 >>> (ucharImage, grayImage, imageWidth, imageHeight, imageChannels);
	cudaDeviceSynchronize();
#if DEBUG_MODE
	//wbCheck(cudaMemcpy(debug_c, grayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//for (int i = 0;i < 5; i++) {
	//	wbLog(TRACE, "The value of grayImage = ", (int)debug_c[i]);
	//}
#endif
	
	dim3 dimBlock2(BLOCK_SIZE, 1, 1);
	dim3 dimGrid2((imageWidth * imageHeight - 1)/BLOCK_SIZE + 1, 1, 1);
	calHistogram <<< dimGrid2, dimBlock2 >>> (grayImage, imageWidth * imageHeight, histogram);
	cudaDeviceSynchronize();
#if DEBUG_MODE
	//wbCheck(cudaMemcpy(debug_i, histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost));
	//for (int i = 0;i < 256; i++) {
	//	wbLog(TRACE, "The value of histogram = ", (int)debug_i[i]);
	//}
#endif
	
	dim3 dimBlock3(CDF_BLOCK_SIZE, 1, 1);
	dim3 dimGrid3((256 - 1)/(CDF_BLOCK_SIZE << 1) + 1, 1, 1);
	calCdf <<< dimGrid3, dimBlock3 >>> (histogram, cdf, imageWidth * imageHeight);
	cudaDeviceSynchronize();
#if DEBUG_MODE
	//wbCheck(cudaMemcpy(debug_f, cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost));
	//for (int i = 0;i < 256; i++) {
	//	wbLog(TRACE, "The value of cdf = ", (int)(debug_f[i] * imageWidth * imageHeight));
	//}
#endif
	
	dim3 dimBlock4(CDF_BLOCK_SIZE, 1, 1);
	dim3 dimGrid4(1, 1, 1);
	calMinCdf <<< dimGrid4, dimBlock4 >>> (cdf, mincdf);
	cudaDeviceSynchronize();
#if DEBUG_MODE
	wbCheck(cudaMemcpy(debug_f, mincdf, sizeof(float), cudaMemcpyDeviceToHost));
	wbLog(TRACE, "The value of mincdf = ", (int)(debug_f[0] * imageWidth * imageHeight));
#endif
	equalization <<< dimGrid, dimBlock >>> (cdf, mincdf, ucharImage, imageWidth, imageHeight, 
											imageChannels, pixelSize);
	cudaDeviceSynchronize();
#if DEBUG_MODE
	wbCheck(cudaMemcpy(debug_c, ucharImage, pixelSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 6; i++) {
		wbLog(TRACE, "The value of equalization ucharImage = ", (int)debug_c[i]);
	}
#endif	
	castImageTofloat <<< dimGrid, dimBlock >>> (deviceOutputImageData, ucharImage, imageWidth, 
												imageHeight, imageChannels, pixelSize);
	cudaDeviceSynchronize();
	
	wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, pixelSize * sizeof(float), 
					   cudaMemcpyDeviceToHost));

    wbSolution(args, outputImage);

    //@@ insert code here
	wbCheck(cudaFree(deviceInputImageData));
    wbCheck(cudaFree(deviceOutputImageData));
	wbCheck(cudaFree(ucharImage));
	wbCheck(cudaFree(grayImage));
	wbCheck(cudaFree(histogram));
	wbCheck(cudaFree(cdf));
	wbCheck(cudaFree(mincdf));
	
#if DEBUG_MODE
	free(debug_c);
	free(debug_f);
	free(debug_i);
#endif
	
	wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

