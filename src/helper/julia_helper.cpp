
#include "helper/julia_helper.h"



void * DenseDimmWitted_Open(double ** _data, long _nrow, long _ncols, double * _model, long nelem, int MODELREPL, int DATAREPL, int ACCESS){

	//std::cout << "!!!!!!!!! " << _nrow << std::endl;

	double ** p = new double*[_nrow];
	long i;
	for(i=0;i<_nrow;i++){
		p[i] = & ( (double*) _data ) [i*_ncols];
		//std::cout << i << " -- " << p[i] << std::endl;
		//std::cout << i  << std::endl;
	}
	//std::cout << i << std::endl;
	//std::cout << p[i-1] << std::endl;

	JuliaModle * model = new JuliaModle(nelem);
	for(int i=0;i<nelem;i++){
		model->p[i] = _model[i];
	}

	DenseDimmWitted<double, JuliaModle, DW_STRAWMAN, DW_SHARDING, DW_ROW> * dw =
		new DenseDimmWitted<double, JuliaModle, DW_STRAWMAN, DW_SHARDING, DW_ROW>(p, _nrow, _ncols, model);

	return (void*) dw;
}

unsigned int DenseDimmWitted_Register_Row(void * p_dw, double (*F_ROW) (const DenseVector<double> * const, JuliaModle *)){

	return ((DenseDimmWitted<double, JuliaModle, DW_STRAWMAN, DW_SHARDING, DW_ROW>*)p_dw)
			->register_row(F_ROW);
}

double DenseDimmWitted_Exec(void * p_dw, unsigned int fhandle){
	return ((DenseDimmWitted<double, JuliaModle, DW_STRAWMAN, DW_SHARDING, DW_ROW>*)p_dw)
			->exec(fhandle);
}



























void Print(void * b){
	for(int i=0;i<100;i++){
		std::cout << "!" << ((double*)b)[i] << std::endl;
	}
}