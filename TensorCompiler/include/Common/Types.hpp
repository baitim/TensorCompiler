#pragma once

#include <iostream>

#ifdef DEBUG_DUMP
#define dbgs std::cout
#else
#define dbgs if (false) std::cout
#endif

namespace tc {

enum class DataType {
    FLOAT, INT32, INT64, BOOL, STRING, UINT8, DOUBLE, UNKNOWN
};

enum class OpType {
    ADD, MUL, CONV, RELU, MATMUL, GEMM, RESHAPE, UNKNOWN
};

} // namespace tc