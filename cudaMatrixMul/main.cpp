#include "header.h"

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

__global__ void matrixMul_unroll(float* C, float* A, float* B, int wA, int wB)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];

	float cv[BLOCK_SIZE] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin + wA - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * VECTOR_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;
	int cBegin = wB * BLOCK_SIZE * by + VECTOR_SIZE * BLOCK_SIZE * bx;
	
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		float* Ap = &A[a + wA * ty + tx];
		float* ap = &As[ty + BLOCK_SIZE * tx];
#pragma unroll
		for (int i = 0; i < BLOCK_SIZE; i += VECTOR_SIZE) {
			ap[i] = Ap[wA * i];
		}
		__syncthreads();
		ap = &As[0];
		float* bp = &B[b + BLOCK_SIZE * ty + tx];
#pragma unroll
		for (int i = 0; i < BLOCK_SIZE; i++) {
			float bv = bp[0];
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE; j++) {
				cv[j] += ap[j] * bv;
			}
			ap += BLOCK_SIZE;
			bp += wB;
		}
		__syncthreads();
	}
	float* Cp = &C[cBegin];
	Cp += BLOCK_SIZE * ty + tx;
	int cStep = wB;
#pragma unroll
	for (int i = 0; i < BLOCK_SIZE; i++) {
		Cp[0] = cv[i]; Cp += cStep;
	}
}

int main() {
	srand(time(NULL));
	
	// 32, 64, 128, 256, 512, 1024
	uint32_t heightA = BLOCK_SIZE * 256;
	uint32_t widthA = BLOCK_SIZE * 256;
	uint32_t widthB = BLOCK_SIZE * 256;

	uint32_t sizeA = heightA * widthA;
	uint32_t sizeB = widthA * widthB;
	uint32_t sizeC = heightA * widthB;

	uint32_t memSizeA = sizeof(float) * sizeA;
	uint32_t memSizeB = sizeof(float) * sizeB;
	uint32_t memSizeC = sizeof(float) * sizeC;

	float flop = 2.0f * heightA * widthA * widthB;

	float* A = (float*)malloc(memSizeA);
	float* B = (float*)malloc(memSizeB);
	float* C = (float*)malloc(memSizeC);

	randomInit(A, sizeA);
	randomInit(B, sizeB);

	float* gpuA;
	float* gpuB;
	float* gpuC;

	cudaMalloc((void**)&gpuA, memSizeA);
	cudaMalloc((void**)&gpuB, memSizeB);
	cudaMalloc((void**)&gpuC, memSizeC);



	dim3 threads, grid;
	cudaEvent_t start, stop;
	float msecTotal;
	
	int iter = 100;
	while (iter--)
	{
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);

		threads = dim3(BLOCK_SIZE, VECTOR_SIZE);
		grid = dim3(widthB / (BLOCK_SIZE * VECTOR_SIZE), heightA / BLOCK_SIZE);
		cudaMemcpy(gpuA, A, memSizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB, B, memSizeB, cudaMemcpyHostToDevice);
		matrixMul_unroll << <grid, threads >> > (gpuA, gpuB, gpuC, widthA, widthB);
		cudaMemcpy(C, gpuC, memSizeC, cudaMemcpyDeviceToHost);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&msecTotal, start, stop);
		cout << "unrollMatrixMul: " << msecTotal << " ms" << endl;
	}

	return 0;
}