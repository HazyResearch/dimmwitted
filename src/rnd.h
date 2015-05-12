
#include "immintrin.h"

#ifndef _LPBLAS_RND_H
#define _LPBLAS_RND_H

struct rstate {
    __m256i z;
    __m256i w;
    __m256i jsr;
    __m256i jcong;
    __m256i a;
    __m256i b;
};

void init(rstate * pstate){
    
}



/*
#define znew  (z=36969*(z&65535)+(z>>16))
#define wnew  (w=18000*(w&65535)+(w>>16))
#define MWC   ((znew<<16)+wnew)
#define SHR3  (jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
#define CONG  (jcong=69069*jcong+1234567)
#define FIB   ((b=a+b),(a=b-a))
#define KISS  ((MWC^CONG)+SHR3)
#define LFIB4 (c++,t[c]=t[c]+t[UC(c+58)]+t[UC(c+119)]+t[UC(c+178)])
#define SWB   (c++,bro=(x<y),t[c]=(x=t[UC(c+34)])-(y=t[UC(c+19)]+bro))
#define UNI   (KISS*2.328306e-10)
#define VNI   ((long) KISS)*4.656613e-10
#define UC    (unsigned char) 
typedef unsigned long UL;
*/
/* Global static variables: */

//static UL z=362436069, w=521288629, jsr=123456789, jcong=380116160;
//static UL a=224466889, b=7584631, t[256], x=0, y=0, bro;
//static unsigned char c=0;
//
///* Example procedure to set the table, using KISS: */
//void settable(UL i1,UL i2,UL i3,UL i4,UL i5, UL i6)
//{ int i; z=i1; w=i2; jsr=i3; jcong=i4; a=i5; b=i6;
//for(i=0;i<256;i=i+1) t[i]=KISS; }

#endif