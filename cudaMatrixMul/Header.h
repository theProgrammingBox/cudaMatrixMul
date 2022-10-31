#pragma once
#include <iostream>
#include "Randoms.h"
#include <cuda_runtime.h>

using std::cout;
using std::endl;

const uint32_t VECTOR_SIZE = 4;  //any number?
const uint32_t BLOCK_SIZE = 16;   //multiples of 8?

static Random randoms;