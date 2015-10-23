// Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef _GLM_SPARSE_SGD_H
#define _GLM_SPARSE_SGD_H

#include "dimmwitted.h"

/**
 * \brief This file shows how to specify the same
 * synthetic model as in app/glm_dense_sgd.h
 * but store the data as sparse matrix instead
 * of dense matrix.
 *
 * See app/glm_dense_sgd.h for more detailed 
 * comments.
 */
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
  const double Z = 0.0001 * (-label + 1.0/(1.0+d));

  for(int i=0;i<ex->n-1;i++){
    model[ex->idxs[i]] -= ex->p[i] * Z;
  }

  return 1.0;
}

template<ModelReplType MODELREPL, DataReplType DATAREPL>
double test_glm_sparse_sgd(){

  long nexp = 100000;
  long nfeat = 1024;

  double * examples = new double[nexp*(nfeat+1)];
  long * cols = new long[nexp*(nfeat+1)];
  long * rows = new long[nexp];

  long ct = 0;
  for(long i=0;i<nexp;i++){
    rows[i] = ct;    
    for(int j=0;j<nfeat;j++){
      examples[ct] = 1;
      cols[ct] = j;
      ct ++;
    }
    examples[ct] = drand48() > 0.8 ? 0 : 1.0;
    cols[ct] = nfeat;
    ct ++;
  }

  GLMModelExample_Sparse model(nfeat);
  for(int i=0;i<model.n;i++){
    model.p[i] = 0.0;
  }

  SparseDimmWitted<double, GLMModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> 
    dw(examples, rows, cols, nexp, nfeat+1, nexp*(nfeat+1), &model);
  
  unsigned int f_handle_grad = dw.register_row(f_lr_grad_sparse);
  unsigned int f_handle_loss = dw.register_row(f_lr_loss_sparse);
  dw.register_model_avg(f_handle_grad, f_lr_modelavg);
  dw.register_model_avg(f_handle_loss, f_lr_modelavg);

  double sum = 0.0;
  for(int i_epoch=0;i_epoch<2;i_epoch++){
    double loss = dw.exec(f_handle_loss)/nexp;
    sum = 0.0;
    for(int i=0;i<nfeat;i++){
      sum += model.p[i];
    }
    std::cout.precision(8);
    std::cout << sum << "    loss=" << loss << std::endl;
    dw.exec(f_handle_grad);
  }

  return sum;
}

/**
 * \brief This is one example of running SGD for logistic regression
 * in DimmWitted. You can find more examples in test/glm_dense.cc
 * and test/glm_sparse.cc, and the documented code in 
 * app/glm_dense_sgd.h
 */
//int main(int argc, char** argv){
//  double rs = test_glm_sparse_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_SHARDING>();
//  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
//  return 0;
//}

#endif

