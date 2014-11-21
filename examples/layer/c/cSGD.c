#include <R.h>
#include <Rinternals.h>

#include <stdio.h>

double * get_i_example(double * mat, int nrow, int ncol, int icol){
	return &mat[nrow*icol];
}

inline double dot(double * v1, double * v2, int len){
	double rs = 0;
	for(int i=0;i<len;i++){
		rs += v1[i]*v2[i];
	}
	return rs;
}

SEXP cSGD(SEXP X, SEXP y, SEXP _NEPOCH, SEXP stepsize){

	//printf("SEE?\n");

	int NEPOCH = (int)(*REAL(_NEPOCH));
	double LAMBDA = *(REAL(stepsize));

	printf("NEPOCH: %d\n", NEPOCH);
	printf("LAMBDA: %f\n", LAMBDA);

	int ncol,nrow;
	SEXP Rdim = getAttrib(X,R_DimSymbol);
	nrow = INTEGER(Rdim)[0];
	ncol = INTEGER(Rdim)[1];

	int NEXAMPLE = ncol;
	int NFEATURE = nrow;

	printf("NROW = %d\n", nrow);
	printf("NCOL = %d\n", ncol);

	double * mat = REAL(X);
	double * labels = REAL(y);
	SEXP _model = allocVector(REALSXP,NFEATURE);
	double * model = REAL(_model);

	// first init model
	for(int i=0;i<NFEATURE;i++){
		model[i] = 0.1;
	}

	for(int nepoch=0; nepoch<NEPOCH; nepoch++){
		for(int i=0;i<NEXAMPLE;i++){
			double * example = get_i_example(mat, nrow, ncol, i);
			double label = labels[i];
			//printf("label: %f\n", label);
			double wx = dot(example, model, NFEATURE);
			double prob = 1.0/(1.0+exp(-wx));
			double minus_y_plus_prob = -label + prob;

			//printf("prob: %f\n", prob);
			for(int j=0;j<NFEATURE;j++){
				model[j] -= LAMBDA * minus_y_plus_prob * example[j];
			}
		}
	}

	return _model;
}