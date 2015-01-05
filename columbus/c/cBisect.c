#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


inline double _eval(double z, double q, double y, double LAMBDA){
	return -y + 1.0/(1.0+exp(-z)) + LAMBDA*(z-q);
}

double bisect(double q, double y, double LAMBDA){

	int maxIter = 5;
	double eps = 0.00000001;

	double lo = -10.0;
  	double hi = 10.0;
  	double flo = _eval(lo, q, y, LAMBDA);
  	double fhi = _eval(hi, q, y, LAMBDA);

	if(flo > fhi){
		lo = lo + hi;
		hi = lo - hi;
		lo = lo - hi;

		flo = flo + fhi;
		fhi = flo - fhi;
		flo = flo - fhi;
	}

	while(flo*fhi > 0){
		if(flo > 0){
      		assert(fhi > 0);
 	     	hi = lo;
  		    lo = lo*2;
    	}else{
	    	assert(flo < 0);
	      	lo = hi;
      		hi = hi * 2;
  		}
	    flo = _eval(lo, q, y, LAMBDA);
    	fhi = _eval(hi, q, y, LAMBDA);
	}

  	double mid  = (lo + hi)/2.0;
  	double fmid = _eval(mid, q, y, LAMBDA);
  	double its = 0;
  	while( fabs(fmid) > eps && its < maxIter ){
    	assert(flo*fhi <= 0);
    	if(fmid < 0){
    		lo = mid;
    		flo = fmid;
    	}else{
    		hi = mid;
    		fhi = fmid;
    	}
    	mid  = (lo + hi)/2.0;
    	fmid = _eval(mid, q, y, LAMBDA);
    	//printf("mid=%f, fmid=%f\n", mid, fmid);
    	its += 1;
    }

    //printf("------------\n\n");
	return mid;
}

double * cBISECT(long long int NEXAMPLE, double* AX, double* u, double* b, double LAMBDA){

	double * nzs = (double *)malloc(NEXAMPLE*sizeof(double));

	long long int i;
	for(i=0;i<NEXAMPLE;i++){
		nzs[i] = bisect(AX[i]+u[i], b[i], LAMBDA);
	}

	return nzs;
}