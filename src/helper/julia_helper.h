
#ifndef _JULIA_HELPER_H
#define _JULIA_HELPER_H

#include "common.h"
#include "engine/dimmwitted_dense.h"

class JuliaModle{
public:
  double * const p;
  long n;
  
  JuliaModle(int _n):
    n(_n), p(new double[_n]){}

  JuliaModle( const JuliaModle& other ) :
     n(other.n), p(new double[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }

};

extern "C" {

	void * DenseDimmWitted_Open(double **, long, long, double *, long, int, int, int);

	unsigned int DenseDimmWitted_Register_Row(void*, double (*F_ROW) (const DenseVector<double> * const, JuliaModle *));

	double DenseDimmWitted_Exec(void * p_dw, unsigned int fhandle);

	void Print(void *);

}



#endif