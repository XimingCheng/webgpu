#include <wb.h> 

void addVector(float* A, float* B, float* C, int inputLength) {
	#pragma acc parallel loop copyin(A[0:inputLength]) copyin(B[0:inputLength]) copyout(C[0:inputLength])
	#pragma acc loop
	for (int i = 0; i < inputLength; i++) {
		C[i] = A[i] + B[i];
	}
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  addVector(hostInput1, hostInput2, hostOutput, inputLength);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
