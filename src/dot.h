
#ifndef _LPBLAS_DOT_H
#define _LPBLAS_DOT_H

#include <iostream>
#include "types.h"
#include "immintrin.h"

namespace lpblas {

template<typename LPBLAS_TYPE>
float dot_dense(const LPBLAS_TYPE * const x,
                const LPBLAS_TYPE * const y,
                                     int N);

#ifdef __AVX2__

/**
 * The following dot product algorithm is free from possible overflow.
 *
 * I am go much more crazy by optimizing variable dependencies.
 * But this is already at 11GB/s bandwidth on a single core Haswell,
 * so, probably better to have this cleaner version of the code.
 *
 **/
template<>
float dot_dense<LPBLAS_i8>(const LPBLAS_i8 * const x,
                           const LPBLAS_i8 * const y,
                           int N){

  const float MAX = MAX_VALUE<LPBLAS_i8>();
  const float DIVIDEDBY = 1.0 / MAX / MAX;
  const int n_remainder = N % 32;
  
  float rs[8];
  
  __m256i ymm0, ymm1, ymm2;

  __m256 ymm3;
  
  __m256i ymm_ones_16bit = _mm256_set_epi16(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
  
  __m256  ymm_aggregated_sum = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);

  for(int i=n_remainder;i<N;i+=32){ // 1 cache line per 5 cycle throughput. Looks OK
    ymm0 = _mm256_loadu_si256((__m256i const *)&x[i]);
    ymm1 = _mm256_loadu_si256((__m256i const *)&y[i]);
    
    ymm1 = _mm256_sign_epi8(ymm1, ymm0); // 1-latency 0.5-throughput
    ymm0 = _mm256_abs_epi8(ymm0); // 1-latency

    ymm2 = _mm256_maddubs_epi16(ymm0, ymm1);  // 5-latency 1-throughput
    ymm2 = _mm256_madd_epi16(ymm2, ymm_ones_16bit); // 5-latency 1-throughput
    ymm3 = _mm256_cvtepi32_ps(ymm2); // 3-latency 1-throughput

    ymm_aggregated_sum = _mm256_add_ps(ymm_aggregated_sum, ymm3); // 3-latency 1-throughput
  }

  _mm256_storeu_ps(rs, ymm_aggregated_sum);
  float toreturn = DIVIDEDBY*(rs[0]+rs[1]+rs[2]+rs[3]+rs[4]+rs[5]+rs[6]+rs[7]);
  for(int i=0;i<n_remainder;i++){
    toreturn += x[i] * y[i] * DIVIDEDBY;
  }
  return toreturn;
}

/**
 * 16-bit is very similar to 8-bit. Although we could automatically
 * generate this, but we only have 4 precision levels for now.
 *
 **/
template<>
float dot_dense<LPBLAS_i16>(const LPBLAS_i16 * const x,
                            const LPBLAS_i16 * const y,
                            int N){

  const float MAX = MAX_VALUE<LPBLAS_i16>();
  const float DIVIDEDBY = 1.0 / MAX / MAX;
  const int n_remainder = N % 16;
  
  float rs[8];
  
  __m256i ymm0, ymm1, ymm2;

  __m256 ymm3;

  __m256  ymm_aggregated_sum = _mm256_set_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);

  for(int i=n_remainder;i<N;i+=16){ // 1 cache line per 3 cycle throughput. Looks OK
                          // One interesting thing is that this is actually faster than 8-bit case
                          // but it is memory-bound, so who cares.
    ymm0 = _mm256_loadu_si256((__m256i const *)&x[i]);
    ymm1 = _mm256_loadu_si256((__m256i const *)&y[i]);
    
    ymm2 = _mm256_madd_epi16(ymm0, ymm1);  // 5-latency 1-throughput

    ymm3 = _mm256_cvtepi32_ps(ymm2); // 3-latency 1-throughput

    ymm_aggregated_sum = _mm256_add_ps(ymm_aggregated_sum, ymm3); // 3-latency 1-throughput
  }

  _mm256_storeu_ps(rs, ymm_aggregated_sum);
  float toreturn = DIVIDEDBY*(rs[0]+rs[1]+rs[2]+rs[3]+rs[4]+rs[5]+rs[6]+rs[7]);
  for(int i=0;i<n_remainder;i++){
    toreturn += x[i] * y[i] * DIVIDEDBY;
  }
  return toreturn;
}

#elif defined(__AVX__)

/**
 * The following dot product algorithm is free from possible overflow.
 *
 * I am go much more crazy by optimizing variable dependencies.
 * But this is already at 11GB/s bandwidth on a single core Haswell,
 * so, probably better to have this cleaner version of the code.
 *
 **/
template<>
float dot_dense<LPBLAS_i8>(const LPBLAS_i8 * const x,
                           const LPBLAS_i8 * const y,
                           int N){

  const float MAX = MAX_VALUE<LPBLAS_i8>();
  const float DIVIDEDBY = 1.0 / MAX / MAX;
  const int n_remainder = N % 16;
  
  float rs[4];
  
  __m128i ymm0, ymm1, ymm2;
  
  __m128 ymm3;
  
  __m128i ymm_ones_16bit = _mm_set_epi16(1,1,1,1,1,1,1,1);
  
  __m128  ymm_aggregated_sum = _mm_set_ps(0.0,0.0,0.0,0.0);

  for(int i = n_remainder; i < N; i += 16){
    ymm0 = _mm_loadu_si128((__m128i const *)&x[i]);
    ymm1 = _mm_loadu_si128((__m128i const *)&y[i]);
    
    ymm1 = _mm_sign_epi8(ymm1, ymm0);
    ymm0 = _mm_abs_epi8(ymm0);

    ymm2 = _mm_maddubs_epi16(ymm0, ymm1);
    ymm2 = _mm_madd_epi16(ymm2, ymm_ones_16bit);
    ymm3 = _mm_cvtepi32_ps(ymm2);

    ymm_aggregated_sum = _mm_add_ps(ymm_aggregated_sum, ymm3);
  }

  _mm_storeu_ps(rs, ymm_aggregated_sum);
  float toreturn = DIVIDEDBY*(rs[0]+rs[1]+rs[2]+rs[3]);
  for(int i = 0; i < n_remainder; i++){
    toreturn += x[i] * y[i] * DIVIDEDBY;
  }
  return toreturn;
}

/**
 * 16-bit is very similar to 8-bit. Although we could automatically
 * generate this, but we only have 4 precision levels for now.
 *
 **/
template<>
float dot_dense<LPBLAS_i16>(const LPBLAS_i16 * const x,
                            const LPBLAS_i16 * const y,
                            int N){

  const float MAX = MAX_VALUE<LPBLAS_i16>();
  const float DIVIDEDBY = 1.0 / MAX / MAX;
  const int n_remainder = N % 8;
  
  float rs[4];
  
  __m128i ymm0, ymm1, ymm2;

  __m128 ymm3;

  __m128 ymm_aggregated_sum = _mm_set_ps(0.0,0.0,0.0,0.0);

  for(int i = n_remainder; i < N; i+=8){
    ymm0 = _mm_loadu_si128((__m128i const *)&x[i]);
    ymm1 = _mm_loadu_si128((__m128i const *)&y[i]);
    
    ymm2 = _mm_madd_epi16(ymm0, ymm1);

    ymm3 = _mm_cvtepi32_ps(ymm2);

    ymm_aggregated_sum = _mm_add_ps(ymm_aggregated_sum, ymm3);
  }

  _mm_storeu_ps(rs, ymm_aggregated_sum);
  float toreturn = DIVIDEDBY*(rs[0]+rs[1]+rs[2]+rs[3]);
  for(int i = 0; i < n_remainder; i++){
    toreturn += x[i] * y[i] * DIVIDEDBY;
  }
  return toreturn;
}


void axpy(LPBLAS_i16 * const x,
           const LPBLAS_i16 * const y,
           float a,
           int N){

  const int n_remainder = N % 4;
  
  __m64 immx, immy;

  __m128 fmmx, fmmy, fmma;

  for(int i = n_remainder; i < N; i+=4){
    immx = _mm_cvtsi64_m64(*(__int64_t const *)&x[i]);
    immy = _mm_cvtsi64_m64(*(__int64_t const *)&y[i]);
    
    fmmx = _mm_cvtpi16_ps(immx);
    fmmy = _mm_cvtpi16_ps(immy);

    fmma = _mm_set_ps1(a);

    fmmy = _mm_mul_ps(fmmy, fmma);
    fmmx = _mm_add_ps(fmmx, fmmy);

    immx = _mm_cvtps_pi16(fmmx);

    *(__int64_t *)&x[i] = _mm_cvtm64_si64(immx);
  }

  for(int i = 0; i < n_remainder; i++){
    x[i] = (LPBLAS_i16)((float)x[i] + ((float)y[i] * a));
  }

  return;
}

#else

#error "need AVX support for lpblas library"

#endif

/**
 * TODO: This call should be change to OpenBLAS kernel to 
 * validate our point (i.e., we need to be fair and best effort
 * for 32-bit float.) However, I don't think OpenBLAS can be
 * any faster than this on Haswell given how memory-bandwidth
 * bound this kernel is.
 *
 **/
template<>
float dot_dense<LPBLAS_f32>(const LPBLAS_f32 * const x,
                            const LPBLAS_f32 * const y,
                                                 int N){

  const float MAX = MAX_VALUE<LPBLAS_f32>();
  const float n_remainder = N % 8;
  float rs[8];
  for(int i=n_remainder;i<N;i+=8){
    rs[0] += x[i] * y[i];
    rs[1] += x[i+1] * y[i+1];
    rs[2] += x[i+2] * y[i+2];
    rs[3] += x[i+3] * y[i+3];
    rs[4] += x[i+4] * y[i+4];
    rs[5] += x[i+5] * y[i+5];
    rs[6] += x[i+6] * y[i+6];
    rs[7] += x[i+7] * y[i+7];
  }
  float toreturn = (rs[0]+rs[1]+rs[2]+rs[3]+rs[4]+rs[5]+rs[6]+rs[7]);
  for(int i=0;i<n_remainder;i++){
    toreturn += x[i] * y[i];
  }
  return toreturn;
}

}

#endif





