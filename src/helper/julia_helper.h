
#ifndef _JULIA_HELPER_H
#define _JULIA_HELPER_H

#define JL_DLLEXPORT DLLEXPORT

#include "assert.h"
#include "julia.h"
//#include "julia_internal.h"

//#include "builtin_proto.h"

#include "dimmwitted.h"

#include "engine/dimmwitted_dense_julia.h"

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

	void * DenseDimmWitted_Open2(jl_value_t *data_type, jl_value_t *model_type, \
								long data_nrows, long data_ncols, long model_nelems, \
								void * data, void * model, int, int, int, 
								jl_value_t *shared_data_type, int n_shared_data, void * shared_data);

	unsigned int DenseDimmWitted_Register_Row2(
		void *, double (*F_ROW) (const jl_array_t * const, jl_array_t *, jl_array_t *), int, int, int);

	unsigned int DenseDimmWitted_Register_Col2(
		void *, double (*F_COL) (const jl_array_t * const, jl_array_t *, jl_array_t *), int, int, int);

	unsigned int DenseDimmWitted_Register_C2R2(
		void *, double (*F_C2R) (const jl_array_t* const p_col, int i_col,
					const jl_array_t* const p_rows, jl_array_t *, jl_array_t *), int, int, int);

	double DenseDimmWitted_Exec2(void * p_dw, unsigned int fhandle, int, int, int);

	void * SparseDimmWitted_Open2(jl_value_t *data_type, jl_value_t *wrapped_el_type, jl_value_t *model_type, \
								long data_nrows, long data_ncols, long data_nzelems, long model_nelems, \
								void * data, long * data_rowval, long * data_colptr, void * model, int, int, int, 
								jl_value_t *shared_data_type, int n_shared_data, void * shared_data);


	void set_n_numa_node(void * p_dw, int n_numa_node, int model_repl_type, int data_repl_type, int data_access);

	void set_n_thread_per_node(void * p_dw, int n_thread_per_node, int model_repl_type, int data_repl_type, int data_access);

	void Hello();

	void DenseDimmWitted_Register_ModelAvg2(
		void*, unsigned int, void (*F_AVG) (jl_array_t* const p_models, int nreplicas, int ireplica), int, int, int);



	void * DenseDimmWitted_Open(double **, long, long, double *, long, int, int, int);

	unsigned int DenseDimmWitted_Register_Row(void*, double (*F_ROW) (const DenseVector<double> * const, JuliaModle *));

	double DenseDimmWitted_Exec(void * p_dw, unsigned int fhandle);

	void Print(void *);

}

#endif

