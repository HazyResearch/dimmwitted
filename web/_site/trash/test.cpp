#include <xmmintrin.h>
#include <immintrin.h>
#include <avxintrin.h>
#include <math.h>

float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


double lr_update_sse (const double * __restrict__ const ex, double * __restrict__ const model, double label, int nfeat){
  
  __m256d aa, bb, cc, ss;
  ss = _mm256_set1_pd(0);
  double s[4];
  double dot;

  for(int i = 0 ; i < nfeat ; i += 4) {
     aa = _mm256_load_pd(ex + i);
     bb = _mm256_load_pd(model + i);
     cc = _mm256_mul_pd(aa, bb);
     ss = _mm256_add_pd(ss, cc);
  }

  return sum8(ss);

  /*
  _mm256_store_pd(s, ss);
  double dot = s[0] + s[1] + s[2] + s[3];

  const double d = exp(dot);
  const double Z = 0.0001 * (-label + d/(1.0+d));

  for(int i=0;i<nfeat;i++){
    model[i] -= ex[i] * Z;
  }
  */
}
