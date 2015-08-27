
#include "types.h"
#include "dot.h"

using namespace lpblas;

template float dot_dense<LPBLAS_i8>(const LPBLAS_i8 * const x, const LPBLAS_i8 * const y, int N);
template float dot_dense<LPBLAS_i16>(const LPBLAS_i16 * const x, const LPBLAS_i16 * const y, int N);
