#include "header.h"

void PrintStartup()
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	int devID = 0;
	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i) data[i] = randoms.normalRand();
}

__global__ void
matrixMul_compOpt(float* C, float* A, float* B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];

	// Declaration of the shared memory array Bs used to
	// store the sub-matrix of B
	// __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	float cv[BLOCK_SIZE] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	// float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {


		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		for (int i = 0; i < 4; i++) {
			As[(i * 4 + ty) + BLOCK_SIZE * tx] = A[a + wA * (i * 4 + ty) + tx];
		}
		__syncthreads();

		float* ap = &As[0];
		float* bp = &B[b + BLOCK_SIZE * ty + tx];

		for (int i = 0; i < BLOCK_SIZE; i++) {
			float bv = bp[0];
			cv[0] += ap[0] * bv;
			cv[1] += ap[1] * bv;
			cv[2] += ap[2] * bv;
			cv[3] += ap[3] * bv;
			cv[4] += ap[4] * bv;
			cv[5] += ap[5] * bv;
			cv[6] += ap[6] * bv;
			cv[7] += ap[7] * bv;
			cv[8] += ap[8] * bv;
			cv[9] += ap[9] * bv;
			cv[10] += ap[10] * bv;
			cv[11] += ap[11] * bv;
			cv[12] += ap[12] * bv;
			cv[13] += ap[13] * bv;
			cv[14] += ap[14] * bv;
			cv[15] += ap[15] * bv;
			ap += BLOCK_SIZE;
			bp += wB;
		}

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + VECTOR_SIZE * BLOCK_SIZE * bx;
	c += BLOCK_SIZE * ty + tx;
	for (int i = 0; i < BLOCK_SIZE; i++) {
		C[c] = cv[i];
		c += wB;
	}

}



void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	for (unsigned int i = 0; i < hA; ++i)
		for (unsigned int j = 0; j < wB; ++j) {
			double sum = 0;
			for (unsigned int k = 0; k < wA; ++k) {
				double a = A[i * wA + k];
				double b = B[k * wB + j];
				sum += a * b;
			}
			C[i * wB + j] = (float)sum;
		}
}


void PrintMatrix(float* matrix, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << matrix[i * width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

__global__ void
matrixMul(float* C, float* A, float* B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float AS[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float BS[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		AS[ty][tx] = A[a + wA * ty + tx];
		BS[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += AS[ty][k] * BS[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}
__global__ void
matrixMul_noBankConflict(float* C, float* A, float* B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	__shared__ float AS[BLOCK_SIZE][BLOCK_SIZE];

	// Declaration of the shared memory array Bs used to
	// store the sub-matrix of B
	__shared__ float BS[BLOCK_SIZE][BLOCK_SIZE];

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {


		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		AS[ty][tx] = A[a + wA * ty + tx];
		BS[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += AS[ty][k] * BS[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}
__global__ void
matrixMul_unroll(float* C, float* A, float* B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];

	// Declaration of the shared memory array Bs used to
	// store the sub-matrix of B
	// __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	float cv[BLOCK_SIZE] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	int cBegin = wB * BLOCK_SIZE * by + VECTOR_SIZE * BLOCK_SIZE * bx;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	// float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {


		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		float* Ap = &A[a + wA * ty + tx];
		float* ap = &As[ty + BLOCK_SIZE * tx];
#pragma unroll
		for (int i = 0; i < 16; i += 4) {
			ap[i] = Ap[wA * i];
		}
		__syncthreads();

		ap = &As[0];
		float* bp = &B[b + BLOCK_SIZE * ty + tx];

#pragma unroll      
		for (int i = 0; i < BLOCK_SIZE; i++) {
			float bv = bp[0];
			cv[0] += ap[0] * bv;
			cv[1] += ap[1] * bv;
			cv[2] += ap[2] * bv;
			cv[3] += ap[3] * bv;
			cv[4] += ap[4] * bv;
			cv[5] += ap[5] * bv;
			cv[6] += ap[6] * bv;
			cv[7] += ap[7] * bv;
			cv[8] += ap[8] * bv;
			cv[9] += ap[9] * bv;
			cv[10] += ap[10] * bv;
			cv[11] += ap[11] * bv;
			cv[12] += ap[12] * bv;
			cv[13] += ap[13] * bv;
			cv[14] += ap[14] * bv;
			cv[15] += ap[15] * bv;
			ap += BLOCK_SIZE;
			bp += wB;
		}

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	float* Cp = &C[cBegin];
	Cp += BLOCK_SIZE * ty + tx;
	int cStep = wB;
#pragma unroll
	for (int i = 0; i < BLOCK_SIZE; i++) {
		Cp[0] = cv[i]; Cp += cStep;
	}

}


int main() {
	PrintStartup();

	cudaEvent_t start;
	cudaEvent_t stop;
	float msecTotal;

	// 16, 32, 64, 128, 256, 512, 1024
	uint32_t WA = BLOCK_SIZE * 32;
	uint32_t HA = BLOCK_SIZE * 1;
	uint32_t WB = BLOCK_SIZE * 32;
	uint32_t HB = WA;
	uint32_t WC = WB;
	uint32_t HC = HA;


	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);
	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);
	float flop = 2 * (float)WC * (float)HC * (float)WA;

	// initialize host memory
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	// allocate device memory
	float* d_A;
	cudaMalloc((void**)&d_A, mem_size_A);
	float* d_B;
	cudaMalloc((void**)&d_B, mem_size_B);

	// allocate device memory for result
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* d_C;
	cudaMalloc((void**)&d_C, mem_size_C);

	// allocate host memory for the result
	float* h_C = (float*)malloc(mem_size_C);


	
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	// compute reference solution
	float* reference = (float*)malloc(mem_size_C);
	computeGold(reference, h_A, h_B, HA, WA, WB);
	// stop and destroy timer
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "CPU time: " << msecTotal << " ms" << endl;
	//PrintMatrix(reference, WC, HC);


	dim3 threads, grid;

	/****************************************************/
	/*  CUDA SDK example                                */
	/****************************************************/

	memset(h_C, 0, mem_size_C);
	// create and start timer
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C,
		cudaMemcpyHostToDevice);
	// setup execution parameters
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3(WC / threads.x, HC / threads.y);
	// execute the kernel
	matrixMul << < grid, threads >> > (d_C, d_A, d_B, WA, WB);
	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C,
		cudaMemcpyDeviceToHost);
	// stop and destroy timer
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "CUDA time: " << msecTotal << " ms" << endl;
	//PrintMatrix(h_C, WC, HC);

	bool same = true;
	for (int i = 0; i < size_C; i++) {
		if (abs(reference[i] - h_C[i]) > 0.001) {
			cout << "Error: " << i << ", " << reference[i] << ", " << h_C[i] << endl;
			same = false;
			break;
		}
	}
	if (same)  cout << "Success!" << endl;

	
	memset(h_C, 0, mem_size_C);
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	// setup execution parameters
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3(WC / threads.x, HC / threads.y);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C,
		cudaMemcpyHostToDevice);
	// naive implementation
	matrixMul_noBankConflict << < grid, threads >> > (d_C, d_A, d_B, WA, WB);
	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C,
		cudaMemcpyDeviceToHost);
	// stop and destroy timer
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "CUDA time: " << msecTotal << " ms" << endl;
	//PrintMatrix(h_C, WC, HC);
	
	same = true;
	for (int i = 0; i < size_C; i++) {
		if (abs(reference[i] - h_C[i]) > 0.001) {
			cout << "Error: " << i << ", " << reference[i] << ", " << h_C[i] << endl;
			same = false;
			break;
		}
	}
	if (same)  cout << "Success!" << endl;


	memset(h_C, 0, mem_size_C);
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	// setup execution parameters
	threads = dim3(BLOCK_SIZE, 4);
	grid = dim3(WC / (BLOCK_SIZE * 4), HC / BLOCK_SIZE);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C,
		cudaMemcpyHostToDevice);
	// naive implementation
	matrixMul_compOpt << < grid, threads >> > (d_C, d_A, d_B, WA, WB);
	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C,
		cudaMemcpyDeviceToHost);
	// stop and destroy timer
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "CUDA time: " << msecTotal << " ms" << endl;
	//PrintMatrix(h_C, WC, HC);
	
	same = true;
	for (int i = 0; i < size_C; i++) {
		if (abs(reference[i] - h_C[i]) > 0.001) {
			cout << "Error: " << i << ", " << reference[i] << ", " << h_C[i] << endl;
			same = false;
			break;
		}
	}
	if (same)  cout << "Success!" << endl;

	memset(h_C, 0, mem_size_C);
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	// setup execution parameters
	threads = dim3(BLOCK_SIZE, 4);
	grid = dim3(WC / (BLOCK_SIZE * 4), HC / BLOCK_SIZE);
	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B,
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C,
		cudaMemcpyHostToDevice);
	// naive implementation
	matrixMul_unroll << < grid, threads >> > (d_C, d_A, d_B, WA, WB);
	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C,
		cudaMemcpyDeviceToHost);
	// stop and destroy timer
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "CUDA time: " << msecTotal << " ms" << endl;
	//PrintMatrix(h_C, WC, HC);
	
	same = true;
	for (int i = 0; i < size_C; i++) {
		if (abs(reference[i] - h_C[i]) > 0.001) {
			cout << "Error: " << i << ", " << reference[i] << ", " << h_C[i] << endl;
			same = false;
			break;
		}
	}
	if (same)  cout << "Success!" << endl;

	return 0;
}