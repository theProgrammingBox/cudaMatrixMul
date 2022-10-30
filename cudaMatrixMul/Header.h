#pragma once
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

const uint32_t BLOCK_SIZE = 16;
const uint32_t VECTOR_SIZE = 4;