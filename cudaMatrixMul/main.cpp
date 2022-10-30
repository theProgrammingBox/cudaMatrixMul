#include "header.h"

void FillRandom(float* matrix, uint32_t columns, uint32_t rows, uint32_t columnsCeil, uint32_t rowsCeil, uint32_t matrixCeilBytes)
{
	memset(matrix, 0, matrixCeilBytes);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < columns; j++)
		{
			matrix[i * columnsCeil + j] = randoms.normalRand();
		}
	}
}

void PrintMatrix(float* matrix, uint32_t columns, uint32_t rows)
{
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < columns; j++)
		{
			cout << matrix[i * columns + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void matrixMulCPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t columnsA, uint32_t rowsA, uint32_t columnsB)
{
	for (uint32_t i = 0; i < rowsA; i++)
	{
		for (uint32_t j = 0; j < columnsB; j++)
		{
			float sum = 0;
			for (uint32_t k = 0; k < columnsA; k++)
			{
				sum += inputMatrix[i * columnsA + k] * weightMatrix[k * columnsB + j];
			}
			outputMatrix[i * columnsB + j] = sum;
		}
	}
}

__global__ void matrixMulGPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t columnsA, uint32_t rowsA, uint32_t columnsB)
{
	uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rowsA && col < columnsB)
	{
		float sum = 0;
		for (uint32_t k = 0; k < columnsA; k++)
		{
			sum += inputMatrix[row * columnsA + k] * weightMatrix[k * columnsB + col];
		}
		outputMatrix[row * columnsB + col] = sum;
	}
}

__global__ void MatrixMulGPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t columnsA, uint32_t rowsA, uint32_t columnsB)
{
	uint32_t threadCol = threadIdx.x;
	uint32_t threadRow = threadIdx.y;
	uint32_t blockCol = blockIdx.x;
	uint32_t blockRow = blockIdx.y;

	__shared__ float inputSubMatrix[BLOCK_SIZE * BLOCK_SIZE];
	
	float output[BLOCK_SIZE] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	
	uint32_t inputSubMatrixStart = columnsA * BLOCK_SIZE * blockRow;
	uint32_t inputSubMatrixEnd = inputSubMatrixStart + columnsA - 1;
	
	uint32_t weightSubMatrixStart = BLOCK_SIZE * VECTOR_SIZE * blockCol;
	uint32_t weightSubMatrixStep = BLOCK_SIZE * columnsB;

	uint32_t outputStart = columnsB * BLOCK_SIZE * blockRow + VECTOR_SIZE * BLOCK_SIZE * blockCol;
	
	for (uint32_t inputSubMatrixIndex = inputSubMatrixStart, weightSubMatrixIndex = weightSubMatrixStart;
		inputSubMatrixIndex <= inputSubMatrixEnd;
		inputSubMatrixIndex += BLOCK_SIZE, weightSubMatrixIndex += weightSubMatrixStep)
	{
		float* inputHostSubMatrixPointer = &inputMatrix[inputSubMatrixIndex + columnsA * threadRow + threadCol];
		float* inputDeviceSubMatrixSharedPointer = &inputSubMatrix[threadRow + BLOCK_SIZE * threadCol];
#pragma unroll
		for (uint32_t i = 0; i < 16; i += 4)
		{
			inputDeviceSubMatrixSharedPointer[i] = inputHostSubMatrixPointer[columnsA * i];
		}
		__syncthreads();

		inputDeviceSubMatrixSharedPointer = &inputSubMatrix[0];
		float* weightDeviceSubMatrixPointer = &weightMatrix[weightSubMatrixIndex + BLOCK_SIZE * threadRow + threadCol];
		
#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)
		{
			float weightValue = weightDeviceSubMatrixPointer[0];
#pragma unroll
			for (uint32_t j = 0; j < BLOCK_SIZE; j++)
			{
				output[j] += inputDeviceSubMatrixSharedPointer[j] * weightValue;
			}
			inputDeviceSubMatrixSharedPointer += BLOCK_SIZE;
			weightDeviceSubMatrixPointer += columnsB;
		}
		__syncthreads();
	}

	float* outputDevicePointer = &outputMatrix[outputStart];
	outputDevicePointer += BLOCK_SIZE * threadRow + threadCol;
#pragma unroll
	for (uint32_t i = 0; i < BLOCK_SIZE; i++)
	{
		outputDevicePointer[0] = output[i];
		outputDevicePointer += columnsB;
	}
}
int main()
{
	uint32_t inputEntries = 18;
	uint32_t inputFeatures = 17;
	uint32_t outputFeatures = 19;

	uint32_t inputMatrixSize = inputFeatures * inputEntries;
	uint32_t weightMatrixSize = inputFeatures * outputFeatures;
	uint32_t outputMatrixSize = outputFeatures * inputEntries;

	uint32_t inputEntriesCeilBlocks = ceil((float)inputEntries / BLOCK_SIZE);
	uint32_t inputFeaturesCeilBlocks = ceil((float)inputFeatures / BLOCK_SIZE);
	uint32_t outputFeaturesCeilBlocks = ceil((float)outputFeatures / BLOCK_SIZE);
	
	uint32_t inputEntriesCeil = inputEntriesCeilBlocks * BLOCK_SIZE;
	uint32_t inputFeaturesCeil = inputFeaturesCeilBlocks * BLOCK_SIZE;
	uint32_t outputFeaturesCeil = outputFeaturesCeilBlocks * BLOCK_SIZE;
	
	uint32_t inputMatrixCeilSize = inputFeaturesCeil * inputEntriesCeil;
	uint32_t weightMatrixCeilSize = inputFeaturesCeil * outputFeaturesCeil;
	uint32_t outputMatrixCeilSize = outputFeaturesCeil * inputEntriesCeil;

	uint32_t inputMatrixCeilBytes = sizeof(float) * inputMatrixCeilSize;
	uint32_t weightMatrixCeilBytes = sizeof(float) * weightMatrixCeilSize;
	uint32_t outputMatrixCeilBytes = sizeof(float) * outputMatrixCeilSize;

	float* inputMatrix = (float*)malloc(inputMatrixCeilBytes);
	float* weightMatrix = (float*)malloc(weightMatrixCeilBytes);
	float* outputMatrix = (float*)malloc(outputMatrixCeilBytes);

	float* gpuInputMatrix;
	float* gpuWeightMatrix;
	float* gpuOutputMatrix;

	cudaMalloc((void**)&gpuInputMatrix, inputMatrixCeilBytes);
	cudaMalloc((void**)&gpuWeightMatrix, weightMatrixCeilBytes);
	cudaMalloc((void**)&gpuOutputMatrix, outputMatrixCeilBytes);

	FillRandom(inputMatrix, inputFeatures, inputEntries, inputFeaturesCeil, inputEntriesCeil, inputMatrixCeilBytes);
	FillRandom(weightMatrix, inputFeatures, outputFeatures, inputFeaturesCeil, outputFeaturesCeil, weightMatrixCeilBytes);
	PrintMatrix(inputMatrix, inputFeaturesCeil, inputEntriesCeil);
	PrintMatrix(weightMatrix, inputFeaturesCeil, outputFeaturesCeil);

	

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	matrixMulCPU(inputMatrix, weightMatrix, outputMatrix, inputFeaturesCeil, inputEntriesCeil, outputFeaturesCeil);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "CPU time: " << elapsedTime << endl;
	PrintMatrix(outputMatrix, outputFeaturesCeil, inputEntriesCeil);
	float* reference = (float*)malloc(outputMatrixCeilBytes);
	memcpy(reference, outputMatrix, outputMatrixCeilBytes);



	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	cudaMemcpy(gpuInputMatrix, inputMatrix, inputMatrixCeilBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuWeightMatrix, weightMatrix, weightMatrixCeilBytes, cudaMemcpyHostToDevice);
	
	dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = dim3(inputEntriesCeilBlocks, outputFeaturesCeilBlocks);
	matrixMulGPU << <blocks, threads >> > (gpuInputMatrix, gpuWeightMatrix, gpuOutputMatrix, inputFeaturesCeil, inputEntriesCeil, outputFeaturesCeil);
	cudaMemcpy(outputMatrix, gpuOutputMatrix, outputMatrixCeilBytes, cudaMemcpyDeviceToHost);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "GPU time: " << elapsedTime << endl;
	PrintMatrix(outputMatrix, outputFeaturesCeil, inputEntriesCeil);
	
	bool correct = true;
	for (uint32_t i = 0; i < outputMatrixCeilSize; i++)
	{
		if (abs(outputMatrix[i] - reference[i]) > 0.0001)
		{
			correct = false;
			break;
		}
	}
	cout << "Result is " << (correct ? "correct" : "incorrect") << endl;

	
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	cudaMemcpy(gpuInputMatrix, inputMatrix, inputMatrixCeilBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuWeightMatrix, weightMatrix, weightMatrixCeilBytes, cudaMemcpyHostToDevice);
	
	dim3 threads2 = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks2 = dim3(inputEntriesCeilBlocks, outputFeaturesCeilBlocks);
	MatrixMulGPU << <blocks2, threads2 >> > (gpuInputMatrix, gpuWeightMatrix, gpuOutputMatrix, inputFeaturesCeil, inputEntriesCeil, outputFeaturesCeil);
	cudaMemcpy(outputMatrix, gpuOutputMatrix, outputMatrixCeilBytes, cudaMemcpyDeviceToHost);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "GPU time unroll: " << elapsedTime << endl;
	PrintMatrix(outputMatrix, outputFeaturesCeil, inputEntriesCeil);
	
	correct = true;
	for (uint32_t i = 0; i < outputMatrixCeilSize; i++)
	{
		if (abs(outputMatrix[i] - reference[i]) > 0.0001)
		{
			correct = false;
			break;
		}
	}
	cout << "Result is " << (correct ? "correct" : "incorrect") << endl;
	
	return 0;
}