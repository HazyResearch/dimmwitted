
#include "types.h"

#ifndef _LPBLAS_CV_H
#define _LPBLAS_CV_H

template<typename LPBLAS_TYPE1, typename LPBLAS_TYPE2>
void convert_dense_ceil(const LPBLAS_TYPE1 * const src,
                              LPBLAS_TYPE2 * const dst,
                                               int N);

/**
 * These functions might look stupid, but they actually get
 * > 9 GB/s single-core bandwidth on Haswell. 
 *
 * If we see the Assembly, the compoiler is actually
 * smart enough to use vmulps and vcvttps2dq. I am not any
 * better than this. 
 *
 * Note that we are assuming that the range of input is in
 * [-1, 1]. We should have an assert somewhere.
 **/
template<>
void convert_dense_ceil<LPBLAS_f32,LPBLAS_i8>(const LPBLAS_f32 * const src,
                                                     LPBLAS_i8 * const dst,
                                                                   int N){
    const float MAX = MAX_VALUE<LPBLAS_i8>();
    for(int i=0;i<N;i++){
        dst[i] = (LPBLAS_i8) (src[i] * MAX);
    }
}

template<>
void convert_dense_ceil<LPBLAS_f32,LPBLAS_i16>(const LPBLAS_f32 * const src,
                                                     LPBLAS_i16 * const dst,
                                                                    int N){
    const float MAX = MAX_VALUE<LPBLAS_i16>();
    for(int i=0;i<N;i++){
        dst[i] = (LPBLAS_i16) (src[i] * MAX);
    }
}

/**
 * The DIVIDEDBY variable is used to eliminate the 21-latency-13-throughput
 * Haswell division. I don't think it impacts the numerical stablity. But
 * we need to make sure it passes all tests. This should get > 8 GB/s single-core 
 * bandwidth.
 *
 **/
template<>
void convert_dense_ceil<LPBLAS_i8,LPBLAS_f32>(const LPBLAS_i8 * const src,
                                                   LPBLAS_f32 * const dst,
                                                                  int N){
    const float MAX = MAX_VALUE<LPBLAS_i8>();
    const float DIVIDEDBY = 1.0/MAX;
    for(int i=0;i<N;i++){
        dst[i] = DIVIDEDBY * src[i];
    }
}

template<>
void convert_dense_ceil<LPBLAS_i16,LPBLAS_f32>(const LPBLAS_i16 * const src,
                                                     LPBLAS_f32 * const dst,
                                                                    int N){
    const float MAX = MAX_VALUE<LPBLAS_i16>();
    const float DIVIDEDBY = 1.0/MAX;
    for(int i=0;i<N;i++){
        dst[i] = DIVIDEDBY * src[i];
    }
}

template<>
void convert_dense_ceil<LPBLAS_f32,LPBLAS_f32>(const LPBLAS_f32  * const src,
                                                     LPBLAS_f32  * const dst,
                                                                     int N){
    for(int i=0;i<N;i++){
        dst[i] = src[i];
    }
}

/**
 * I thought I need to specialize for each type, but turns out no.
 **/
//template<>
//void convert_dense<LPBLAS_f32,LPBLAS_i8>(const LPBLAS_f32 * const src,
//                                               LPBLAS_i8  * const dst,
//                                                            int N){
//    const float MAX = MAX_VALUE<LPBLAS_i8>();
//    for(int i=0;i<N;i++){
//        dst[i] = (LPBLAS_i8) (src[i] * MAX);
//    }
//}
//template<>
//void convert_dense<LPBLAS_i8,LPBLAS_f32>(const LPBLAS_i8 * const src,
//                                              LPBLAS_f32 * const dst,
//                                                             int N){
//    const float MAX = MAX_VALUE<LPBLAS_i8>();
//    const float DIVIDEDBY = 1.0/MAX;
//    for(int i=0;i<N;i++){
//        dst[i] = DIVIDEDBY * src[i];
//    }
//}

#endif





