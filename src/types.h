
#ifndef _LPBLAS_TYPES_H
#define _LPBLAS_TYPES_H

#include <limits>

namespace lpblas {

typedef signed char LPBLAS_i8;
typedef signed short LPBLAS_i16;
typedef float LPBLAS_f32;

typedef bool LPBLAS_i1;	// TODO: THIS IS NOT STORED AS ONE-BIT!
						// type for single bit precision, need to double think after all others are done

template<typename LPBLAS_TYPE>
inline float MIN_VALUE();

template<typename LPBLAS_TYPE>
inline float MAX_VALUE();

template<> inline float MIN_VALUE<LPBLAS_i8>() {return -128;}
template<> inline float MAX_VALUE<LPBLAS_i8>() {return 127;}
template<> inline float MIN_VALUE<LPBLAS_i16>() {return -32768;}
template<> inline float MAX_VALUE<LPBLAS_i16>() {return 32767;}
template<> inline float MIN_VALUE<LPBLAS_f32>() {return -1;}
template<> inline float MAX_VALUE<LPBLAS_f32>() {return 1;}

template<typename LPBLAS_TYPE> class LPBLAS_UTIL;

template<> struct LPBLAS_UTIL<LPBLAS_i8> {
  typedef signed short EXPANDED;
  static const int VEC_SIZE = 16;
};

template<> struct LPBLAS_UTIL<LPBLAS_i16> {
  typedef signed int EXPANDED;
  static const int VEC_SIZE = 8;
};

template<> struct LPBLAS_UTIL<LPBLAS_f32> {
  typedef float EXPANDED;
  static const int VEC_SIZE = 4;
};

}

#endif
