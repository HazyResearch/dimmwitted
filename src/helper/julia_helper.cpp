
#include "helper/julia_helper.h"


void * DenseDimmWitted_Open2(jl_value_t *data_type, jl_value_t *model_type,      \
							long data_nrows, long data_ncols, long model_nelems, \
							void * data, void * model, int a, int b, int c){

	// First, get the size of each data element
	jl_value_t *data_el_type = jl_tparam0(data_type);
	size_t data_el_sz = jl_datatype_size(data_el_type);

	// Second, get the size of each model element
	jl_value_t *model_el_type = jl_tparam0(model_type);
	size_t model_el_sz = jl_datatype_size(model_el_type);

	jl_array_t * mmodel = (jl_array_t*) ::operator new(model_nelems * ((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16));
	mmodel->type = model_type;
	mmodel->data = model;
	mmodel->length = model_nelems;
	mmodel->elsize = model_el_sz;
	mmodel->ptrarray = false;
	mmodel->ndims = 1;
	mmodel->isshared = 1;
	mmodel->isaligned = 0;
	mmodel->how = 0;
	mmodel->nrows = model_nelems;
	mmodel->maxsize = model_nelems;
	mmodel->offset = 0;

	DenseDimmWitted_Julia<jl_array_t, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW> * dw =
		new DenseDimmWitted_Julia<jl_array_t, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>
		(data, data_el_type, data_nrows, data_ncols, mmodel);

	return (void*) dw;

}

unsigned int DenseDimmWitted_Register_Row2(void * p_dw, double (*F_ROW) (const jl_array_t * const, jl_array_t *)){
	return ((DenseDimmWitted_Julia<jl_array_t, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>*)p_dw)
			->register_row(F_ROW);
}

void DenseDimmWitted_Register_ModelAvg2(
	void* p_dw, unsigned int f_handle, void (*F_AVG) (jl_array_t** const p_models, int nreplicas, int ireplica)){
	((DenseDimmWitted_Julia<jl_array_t, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>*)p_dw)
			->register_model_avg(f_handle, F_AVG);
}


double DenseDimmWitted_Exec2(void * p_dw, unsigned int fhandle){
	return ((DenseDimmWitted_Julia<jl_array_t, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>*)p_dw)
			->exec(fhandle);
}


void * DenseDimmWitted_Open(double ** _data, long _nrow, long _ncols, double * _model, long nelem, int MODELREPL, int DATAREPL, int ACCESS){

	double ** p = new double*[_nrow];
	long i;
	for(i=0;i<_nrow;i++){
		p[i] = & ( (double*) _data ) [i*_ncols];
	}

	JuliaModle * model = new JuliaModle(nelem);
	for(int i=0;i<nelem;i++){
		model->p[i] = _model[i];
	}

	DenseDimmWitted<double, JuliaModle, DW_MODELREPL_SINGLETHREAD_DEBUG, DW_DATAREPL_SHARDING, DW_ACCESS_ROW> * dw =
		new DenseDimmWitted<double, JuliaModle, DW_MODELREPL_SINGLETHREAD_DEBUG, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>(p, _nrow, _ncols, model);

	return (void*) dw;
}

unsigned int DenseDimmWitted_Register_Row(void * p_dw, double (*F_ROW) (const DenseVector<double> * const, JuliaModle *)){

	return ((DenseDimmWitted<double, JuliaModle, DW_MODELREPL_SINGLETHREAD_DEBUG, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>*)p_dw)
			->register_row(F_ROW);
}

double DenseDimmWitted_Exec(void * p_dw, unsigned int fhandle){
	return ((DenseDimmWitted<double, JuliaModle, DW_MODELREPL_SINGLETHREAD_DEBUG, DW_DATAREPL_SHARDING, DW_ACCESS_ROW>*)p_dw)
			->exec(fhandle);
}



























void Print(void * b){
	for(int i=0;i<100;i++){
		std::cout << "!" << ((double*)b)[i] << std::endl;
	}
}