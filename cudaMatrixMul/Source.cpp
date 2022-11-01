#include "header.h"

void PrintStartup()
{
	cout << "[Matrix Multiply Using CUDA] - Starting...\n";

	int devID = 0;
	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		cout << "cudaGetDevice returned error code " << error << ", line(" << __LINE__ << ")\n";
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		cout << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n";
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		cout << "cudaGetDeviceProperties returned error code " << error << ", line(" << __LINE__ << ")\n";
	}
	else
	{
		cout << "GPU Device " << devID << ": \"" << deviceProp.name << "\" with compute capability " << deviceProp.major << "." << deviceProp.minor << "\n\n";
	}
}

void FillRandom(float* matrix, uint32_t rows, uint32_t columns, uint32_t totalRows, uint32_t totalColumns)
{
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < columns; j++)
		{
			matrix[i * totalColumns + j] = randoms.normalRand();
		}
		for (uint32_t j = columns; j < totalColumns; j++)
		{
			matrix[i * totalColumns + j] = 0.0f;
		}
	}
	for (uint32_t i = rows; i < totalRows; i++)
	{
		for (uint32_t j = 0; j < totalColumns; j++)
		{
			matrix[i * totalColumns + j] = 0.0f;
		}
	}
}

void FillZero(float* matrix, uint32_t sizeBytes)
{
	memset(matrix, 0, sizeBytes);
}

void PrintMatrix(float* matrix, uint32_t rows, uint32_t columns, uint32_t totalColumns)
{
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < columns; j++)
		{
			cout << matrix[i * totalColumns + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void MatrixMulMatrixCPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputFeatures, uint32_t inputEntries, uint32_t outputFeatures)
{
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			float sum = 0;
			for (uint32_t k = 0; k < inputFeatures; k++)
			{
				sum += inputMatrix[i * inputFeatures + k] * weightMatrix[k * outputFeatures + j];
			}
			outputMatrix[i * outputFeatures + j] = sum;
		}
	}
}

void MatrixMulMatrixTransposedCPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputFeatures, uint32_t inputEntries, uint32_t outputFeatures)
{
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			float sum = 0;
			for (uint32_t k = 0; k < inputFeatures; k++)
			{
				sum += inputMatrix[i * inputFeatures + k] * weightMatrix[j * inputFeatures + k];
			}
			outputMatrix[i * outputFeatures + j] = sum;
		}
	}
}

void MatrixTransposedMulMatrixCPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputFeatures, uint32_t inputEntries, uint32_t outputFeatures)
{
	for (uint32_t i = 0; i < inputFeatures; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			float sum = 0;
			for (uint32_t k = 0; k < inputEntries; k++)
			{
				sum += inputMatrix[k * inputFeatures + i] * weightMatrix[k * outputFeatures + j];
			}
			outputMatrix[i * outputFeatures + j] = sum;
		}
	}
}

__global__ void MatrixMulMatrixGPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputFeatures, uint32_t outputFeatures)
{
	uint32_t blockx = blockIdx.x;
	uint32_t blocky = blockIdx.y;
	uint32_t threadx = threadIdx.x;

	__shared__ float inputSubMatrix[BLOCK_SIZE * BLOCK_SIZE];
	float outputColumn[BLOCK_SIZE] = { };
	float* inputSubMatrixThreadPos;
	float* inputMatrixThreadPos;
	float* weightMatrixThreadPos;
	float* outputMatrixThreadPos;
	float weightValue;

	uint32_t inputStartIndex = blocky * (inputFeatures * BLOCK_SIZE);
	uint32_t inputEndIndex = inputStartIndex + inputFeatures;
	uint32_t weightStartIndex = blockx * BLOCK_SIZE;
	uint32_t weightIndexIncrement = outputFeatures * BLOCK_SIZE;

	for (uint32_t inputMatrixIndex = inputStartIndex, weightMatrixIndex = weightStartIndex; inputMatrixIndex < inputEndIndex; inputMatrixIndex += BLOCK_SIZE, weightMatrixIndex += weightIndexIncrement)
	{
		inputSubMatrixThreadPos = (inputSubMatrix)+(BLOCK_SIZE * threadx);
		inputMatrixThreadPos = (inputMatrix)+(inputMatrixIndex + threadx);

#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)  inputSubMatrixThreadPos[i] = inputMatrixThreadPos[inputFeatures * i];
		__syncthreads();

		inputSubMatrixThreadPos = inputSubMatrix;
		weightMatrixThreadPos = (weightMatrix)+(weightMatrixIndex + threadx);

#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)
		{
			weightValue = weightMatrixThreadPos[0];

#pragma unroll
			for (uint32_t j = 0; j < BLOCK_SIZE; j++) outputColumn[j] += inputSubMatrixThreadPos[j] * weightValue;
			inputSubMatrixThreadPos += BLOCK_SIZE;
			weightMatrixThreadPos += outputFeatures;
		}
		__syncthreads();
	}
	outputMatrixThreadPos = (outputMatrix)+(outputFeatures * BLOCK_SIZE * blocky) + (BLOCK_SIZE * blockx) + (threadx);

#pragma unroll
	for (uint32_t i = 0; i < BLOCK_SIZE; i++)
	{
		outputMatrixThreadPos[0] = outputColumn[i];
		outputMatrixThreadPos += outputFeatures;
	}
}

__global__ void MatrixMulMatrixTransposedGPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputFeatures, uint32_t outputFeatures)
{
	uint32_t blockx = blockIdx.x;
	uint32_t blocky = blockIdx.y;
	uint32_t threadx = threadIdx.x;

	__shared__ float inputSubMatrix[BLOCK_SIZE * BLOCK_SIZE];
	float outputColumn[BLOCK_SIZE] = { };
	float* inputSubMatrixThreadPos;
	float* inputMatrixThreadPos;
	float* weightMatrixThreadPos;
	float* outputMatrixThreadPos;
	float weightValue;

	uint32_t inputStartIndex = blocky * (inputFeatures * BLOCK_SIZE);
	uint32_t inputEndIndex = inputStartIndex + inputFeatures;
	uint32_t weightStartIndex = blockx * BLOCK_SIZE;

	for (uint32_t inputMatrixIndex = inputStartIndex, weightMatrixIndex = weightStartIndex; inputMatrixIndex < inputEndIndex; inputMatrixIndex += BLOCK_SIZE, weightMatrixIndex += BLOCK_SIZE)
	{
		inputSubMatrixThreadPos = (inputSubMatrix)+(BLOCK_SIZE * threadx);
		inputMatrixThreadPos = (inputMatrix)+(inputMatrixIndex + threadx);

#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)  inputSubMatrixThreadPos[i] = inputMatrixThreadPos[inputFeatures * i];
		__syncthreads();

		inputSubMatrixThreadPos = inputSubMatrix;
		weightMatrixThreadPos = (weightMatrix)+(weightMatrixIndex + threadx * outputFeatures);

#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)
		{
			weightValue = weightMatrixThreadPos[i];

#pragma unroll
			for (uint32_t j = 0; j < BLOCK_SIZE; j++) outputColumn[j] += inputSubMatrixThreadPos[j] * weightValue;
			inputSubMatrixThreadPos += BLOCK_SIZE;
		}
		__syncthreads();
	}
	outputMatrixThreadPos = (outputMatrix)+(outputFeatures * BLOCK_SIZE * blocky) + (BLOCK_SIZE * blockx) + (threadx);

#pragma unroll
	for (uint32_t i = 0; i < BLOCK_SIZE; i++)
	{
		outputMatrixThreadPos[0] = outputColumn[i];
		outputMatrixThreadPos += outputFeatures;
	}
}

__global__ void MatrixTransposedMulMatrixGPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputFeatures, uint32_t outputFeatures)
{
	uint32_t blockx = blockIdx.x;
	uint32_t blocky = blockIdx.y;
	uint32_t threadx = threadIdx.x;

	__shared__ float inputSubMatrix[BLOCK_SIZE * BLOCK_SIZE];
	float outputColumn[BLOCK_SIZE] = { };
	float* inputSubMatrixThreadPos;
	float* inputMatrixThreadPos;
	float* weightMatrixThreadPos;
	float* outputMatrixThreadPos;
	float weightValue;

	uint32_t inputStartIndex = blocky * (BLOCK_SIZE);
	uint32_t inputEndIndex = inputStartIndex + inputFeatures;
	uint32_t inputIndexIncrement = inputFeatures * BLOCK_SIZE;
	uint32_t weightStartIndex = blockx * BLOCK_SIZE;
	uint32_t weightIndexIncrement = outputFeatures * BLOCK_SIZE;

	for (uint32_t inputMatrixIndex = inputStartIndex, weightMatrixIndex = weightStartIndex; inputMatrixIndex < inputEndIndex; inputMatrixIndex += inputIndexIncrement, weightMatrixIndex += weightIndexIncrement)
	{
		inputSubMatrixThreadPos = (inputSubMatrix)+(BLOCK_SIZE * threadx);
		inputMatrixThreadPos = (inputMatrix)+(inputMatrixIndex + threadx * inputFeatures);

#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)  inputSubMatrixThreadPos[i] = inputMatrixThreadPos[i];
		__syncthreads();

		inputSubMatrixThreadPos = inputSubMatrix;
		weightMatrixThreadPos = (weightMatrix)+(weightMatrixIndex + threadx);

#pragma unroll
		for (uint32_t i = 0; i < BLOCK_SIZE; i++)
		{
			weightValue = weightMatrixThreadPos[0];

#pragma unroll
			for (uint32_t j = 0; j < BLOCK_SIZE; j++) outputColumn[j] += inputSubMatrixThreadPos[j] * weightValue;
			inputSubMatrixThreadPos += BLOCK_SIZE;
			weightMatrixThreadPos += outputFeatures;
		}
		__syncthreads();
	}
	outputMatrixThreadPos = (outputMatrix)+(outputFeatures * BLOCK_SIZE * blocky) + (BLOCK_SIZE * blockx) + (threadx);

#pragma unroll
	for (uint32_t i = 0; i < BLOCK_SIZE; i++)
	{
		outputMatrixThreadPos[0] = outputColumn[i];
		outputMatrixThreadPos += outputFeatures;
	}
}

int main() {
	PrintStartup();

	cudaEvent_t start;
	cudaEvent_t stop;
	float msecTotal;
	dim3 threads, blocks;

	uint32_t inputFeaturesUnrounded = 2;
	uint32_t inputEntriesUnrounded = 1;
	uint32_t outputFeaturesUnrounded = 3;

	uint32_t inputFeatureBlocks = ceil((float)inputFeaturesUnrounded / BLOCK_SIZE);
	uint32_t inputEntryBlocks = ceil((float)inputEntriesUnrounded / BLOCK_SIZE);
	uint32_t outputFeatureBlocks = ceil((float)outputFeaturesUnrounded / BLOCK_SIZE);

	uint32_t inputFeatures = inputFeatureBlocks * BLOCK_SIZE;
	uint32_t inputEntries = inputEntryBlocks * BLOCK_SIZE;
	uint32_t outputFeatures = outputFeatureBlocks * BLOCK_SIZE;

	uint32_t inputMatrixBytes = inputEntries * inputFeatures * sizeof(float);
	uint32_t weightMatrixBytes = inputFeatures * outputFeatures * sizeof(float);
	uint32_t outputMatrixBytes = inputEntries * outputFeatures * sizeof(float);

	float* inputMatrix = (float*)malloc(inputMatrixBytes);
	float* weightMatrix = (float*)malloc(weightMatrixBytes);
	float* outputMatrix = (float*)malloc(outputMatrixBytes);

	float* inputMatrixGPU;
	float* weightMatrixGPU;
	float* outputMatrixGPU;

	cudaMalloc((void**)&inputMatrixGPU, inputMatrixBytes);
	cudaMalloc((void**)&weightMatrixGPU, weightMatrixBytes);
	cudaMalloc((void**)&outputMatrixGPU, outputMatrixBytes);

	FillRandom(inputMatrix, inputEntriesUnrounded, inputFeaturesUnrounded, inputEntries, inputFeatures);
	FillRandom(weightMatrix, inputFeaturesUnrounded, outputFeaturesUnrounded, inputFeatures, outputFeatures);



	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	float* matrixMulMatrixRef = (float*)malloc(outputMatrixBytes);
	MatrixMulMatrixCPU(inputMatrix, weightMatrix, matrixMulMatrixRef, inputFeatures, inputEntries, outputFeatures);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "MatrixMulMatrix CPU time: " << msecTotal << " ms" << endl;
	PrintMatrix(matrixMulMatrixRef, inputEntriesUnrounded, outputFeaturesUnrounded, outputFeatures);



	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	FillZero(outputMatrix, outputMatrixBytes);
	cudaMemcpy(inputMatrixGPU, inputMatrix, inputMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(weightMatrixGPU, weightMatrix, weightMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(outputMatrixGPU, outputMatrix, outputMatrixBytes, cudaMemcpyHostToDevice);
	threads = dim3(BLOCK_SIZE);
	blocks = dim3(outputFeatureBlocks, inputEntryBlocks);
	MatrixMulMatrixGPU << <blocks, threads >> > (inputMatrixGPU, weightMatrixGPU, outputMatrixGPU, inputFeatures, outputFeatures);
	cudaMemcpy(outputMatrix, outputMatrixGPU, outputMatrixBytes, cudaMemcpyDeviceToHost);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "MatrixMulMatrix GPU time: " << msecTotal << " ms" << endl;
	PrintMatrix(outputMatrix, inputEntriesUnrounded, outputFeaturesUnrounded, outputFeatures);

	bool same = true;
	for (uint32_t i = 0; i < inputEntries * outputFeatures; i++) {
		if (abs(outputMatrix[i] - matrixMulMatrixRef[i]) > 0.001) {
			same = false;
			cout << i << " " << outputMatrix[i] << " " << matrixMulMatrixRef[i] << endl;
			break;
		}
	}
	cout << "The results are " << (same ? "the same" : "different") << endl;



	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	float* matrixMulMatrixTransposedRef = (float*)malloc(inputMatrixBytes);
	MatrixMulMatrixTransposedCPU(outputMatrix, weightMatrix, matrixMulMatrixTransposedRef, outputFeatures, inputEntries, inputFeatures);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "MatrixMulMatrixTransposed CPU time: " << msecTotal << " ms" << endl;
	PrintMatrix(matrixMulMatrixTransposedRef, inputEntriesUnrounded, inputFeaturesUnrounded, inputFeatures);


	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	FillZero(inputMatrix, inputMatrixBytes);
	cudaMemcpy(inputMatrixGPU, inputMatrix, inputMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(weightMatrixGPU, weightMatrix, weightMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(outputMatrixGPU, outputMatrix, outputMatrixBytes, cudaMemcpyHostToDevice);
	threads = dim3(BLOCK_SIZE);
	blocks = dim3(inputFeatureBlocks, inputEntryBlocks);
	MatrixMulMatrixTransposedGPU << <blocks, threads >> > (outputMatrixGPU, weightMatrixGPU, inputMatrixGPU, inputFeatures, outputFeatures);
	cudaMemcpy(inputMatrix, inputMatrixGPU, inputMatrixBytes, cudaMemcpyDeviceToHost);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "MatrixMulMatrixTransposed GPU time: " << msecTotal << " ms" << endl;
	PrintMatrix(inputMatrix, inputEntriesUnrounded, inputFeaturesUnrounded, inputFeatures);
	
	same = true;
	for (uint32_t i = 0; i < inputEntries * inputFeatures; i++) {
		if (abs(inputMatrix[i] - matrixMulMatrixTransposedRef[i]) > 0.001) {
			same = false;
			cout << i << " " << inputMatrix[i] << " " << matrixMulMatrixTransposedRef[i] << endl;
			break;
		}
	}
	cout << "The results are " << (same ? "the same" : "different") << endl;


	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	float* matrixTransposedMulMatrixRef = (float*)malloc(weightMatrixBytes);
	MatrixTransposedMulMatrixCPU(inputMatrix, outputMatrix, matrixTransposedMulMatrixRef, inputFeatures, inputEntries, outputFeatures);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "MatrixTransposedMulMatrix CPU time: " << msecTotal << " ms" << endl;
	PrintMatrix(matrixTransposedMulMatrixRef, inputFeaturesUnrounded, outputFeaturesUnrounded, outputFeatures);
	
	
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	FillZero(weightMatrix, weightMatrixBytes);
	cudaMemcpy(inputMatrixGPU, inputMatrix, inputMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(weightMatrixGPU, weightMatrix, weightMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(outputMatrixGPU, outputMatrix, outputMatrixBytes, cudaMemcpyHostToDevice);
	threads = dim3(BLOCK_SIZE);
	blocks = dim3(outputFeatureBlocks, inputFeatureBlocks);
	MatrixTransposedMulMatrixGPU <<<blocks, threads>>> (inputMatrixGPU, outputMatrixGPU, weightMatrixGPU, inputFeatures, outputFeatures);
	cudaMemcpy(weightMatrix, weightMatrixGPU, weightMatrixBytes, cudaMemcpyDeviceToHost);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cout << "MatrixTransposedMulMatrix GPU time: " << msecTotal << " ms" << endl;
	PrintMatrix(weightMatrix, inputFeaturesUnrounded, outputFeaturesUnrounded, outputFeatures);
	
	same = true;
	for (uint32_t i = 0; i < inputFeatures * outputFeatures; i++) {
		if (abs(weightMatrix[i] - matrixTransposedMulMatrixRef[i]) > 0.001) {
			same = false;
			cout << i << " " << weightMatrix[i] << " " << matrixTransposedMulMatrixRef[i] << endl;
			break;
		}
	}
	cout << "The results are " << (same ? "the same" : "different") << endl;
	
	free(inputMatrix);
	free(weightMatrix);
	free(outputMatrix);
	cudaFree(inputMatrixGPU);
	cudaFree(weightMatrixGPU);
	cudaFree(outputMatrixGPU);
	free(matrixMulMatrixRef);
	free(matrixMulMatrixTransposedRef);

	return 0;
}