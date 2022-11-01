#pragma once
#include <iostream>
#include "Randoms.h"
#include <cuda_runtime.h>

using std::cout;
using std::endl;

const uint32_t VECTOR_SIZE = 4;
const uint32_t BLOCK_SIZE = VECTOR_SIZE * 4;

static Random randoms;