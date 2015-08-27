
#include "types.h"
#include "dot.h"

using namespace lpblas;

float test_dot_i8(const LPBLAS_i8 * const x,
                  const LPBLAS_i8 * const y,
                                      int N) {

  return dot_dense(x, y, N);
}

float test_dot_i16(const LPBLAS_i16 * const x,
                   const LPBLAS_i16 * const y,
                                        int N) {

  return dot_dense(x, y, N);
}

