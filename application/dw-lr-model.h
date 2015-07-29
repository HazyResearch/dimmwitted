
#ifndef _DW_LR_MODEL
#define _DW_LR_MODEL

#include "dimmwitted.h"

double stepsize = 0.01;
double epoches = 100;
double lambda = 0.001;

double EPS = 0.00001; // The EPS for float number comparison.

/**
 * See examples/logitic_regression_sparse_sgd.cpp for example.
 **/
class GLMModelExample_Sparse{
public:
  double * const p;
  int n;
  
  GLMModelExample_Sparse(int _n):
    n(_n), p(new double[_n]){}

  GLMModelExample_Sparse( const GLMModelExample_Sparse& other ) :
     n(other.n), p(new double[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }
};

void f_lr_modelavg(GLMModelExample_Sparse** const p_models, int nreplicas, int ireplica){
  GLMModelExample_Sparse * p_model = p_models[ireplica];
  double sum = 0.0;
  for(int i=0;i<p_model->n;i++){
    sum = 0.0;
    for(int j=0;j<nreplicas;j++){
      sum += p_models[j]->p[i];
    }
    (p_model->p)[i] = sum/nreplicas;
  }
}

double f_lr_dump_sparse(const SparseVector<double>* const ex, GLMModelExample_Sparse* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[ex->idxs[i]];
  }
  double prob = 1.0/(1.0+exp(-dot));
  if(prob > 0.5) return 1;
  else return -1;
}


double f_lr_accuracy_sparse(const SparseVector<double>* const ex, GLMModelExample_Sparse* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[ex->idxs[i]];
  }
  double prob = 1.0/(1.0+exp(-dot));
  double predict = prob > 0.5 ? 1.0 : 0.0;
  return fabs(predict-label)<EPS;
}

double f_lr_loss_sparse(const SparseVector<double>* const ex, GLMModelExample_Sparse* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[ex->idxs[i]];
  }
  return  - label * dot + log(exp(dot) + 1.0);
}

double f_lr_grad_sparse(const SparseVector<double>* const ex, GLMModelExample_Sparse* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];

  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[ex->idxs[i]];
  }

  const double d = exp(-dot);
  const double Z = stepsize * (-label + 1.0/(1.0+d));

  for(int i=0;i<ex->n-1;i++){
    model[ex->idxs[i]] -= ex->p[i] * Z;
    model[ex->idxs[i]] /= (1.0+lambda);
  }

  return 1.0;
}



#endif