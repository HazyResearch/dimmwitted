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



#ifndef _GLM_DENSE_SCD_H
#define _GLM_DENSE_SCD_H

#include "common.h"
#include "engine/dimmwitted_dense.h"

/**
 * \brief This file shows how to specify the same
 * synthetic model as in app/glm_dense_sgd.h
 * but use SCD instead of SGD to solve the 
 * problem.
 *
 * See app/glm_dense_sgd.h for more detailed 
 * comments.
 */
class GLMModelExample_SCD{
public:
  double * const p;
  int n;
  
  GLMModelExample_SCD(int _n):
    n(_n), p(new double[_n]){}

  GLMModelExample_SCD( const GLMModelExample_SCD& other ) :
     n(other.n), p(new double[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }
};

void f_lr_modelavg(GLMModelExample_SCD** const p_models, int nreplicas, int ireplica){
  GLMModelExample_SCD * p_model = p_models[ireplica];
  double sum = 0.0;
  for(int i=0;i<p_model->n;i++){
    sum = 0.0;
    for(int j=0;j<nreplicas;j++){
      sum += p_models[j]->p[i];
    }
    (p_model->p)[i] = sum/nreplicas;
  }
}


double f_lr_loss(const DenseVector<double>* const ex, GLMModelExample_SCD* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[i];
  }
  return  - label * dot + log(exp(dot) + 1.0);
}

double f_lr_grad_c2r(const DenseVector<double>* const p_col, int i_col,
                     const DenseVector<double>* const p_rows, int n_rows, 
                     GLMModelExample_SCD* const p_model){

  if(n_rows == 0) return 1.0;
  if(p_rows[0].n-1 == i_col) return 1.0;
  double * model = p_model->p;
  double sum_term = 0.0;
  double pat_term = 0.0;
  for(int i_row=0;i_row<n_rows;i_row++){
    const DenseVector<double> & ex = p_rows[i_row];
    double label = ex.p[ex.n-1];
    double dot = 0.0;
    for(int i=0;i<ex.n-1;i++){
      dot += ex.p[i] * model[i];
    }
    sum_term += label * p_col->p[i_row];
    pat_term += label * p_col->p[i_row] * exp(label*dot) / (1.0 + exp(dot));
  }
  model[i_col] -= 0.00000001 * (-sum_term + pat_term);
  return 1.0;
}


template<ModelReplType MODELREPL, DataReplType DATAREPL>
double test_glm_dense_scd(){

  long nexp = 100000;
  long nfeat = 1024;
  double ** examples = new double* [nexp];
  double * content = new double[nexp*(nfeat+1)];
  for(long i=0;i<nexp;i++){
    examples[i] = &content[i*(nfeat+1)];
    for(int j=0;j<nfeat;j++){
      examples[i][j] = 1;
    }
    examples[i][nfeat] = drand48() > 0.8 ? 0 : 1.0;
  }

  GLMModelExample_SCD model(nfeat);
  for(int i=0;i<model.n;i++){
    model.p[i] = 0.0;
  }

  DenseDimmWitted<double, GLMModelExample_SCD, MODELREPL, DATAREPL, DW_C2R> 
    dw(examples, nexp, nfeat+1, &model);
  
  unsigned int f_handle_grad = dw.register_c2r(f_lr_grad_c2r);
  unsigned int f_handle_loss = dw.register_row(f_lr_loss);
  dw.register_model_avg(f_handle_grad, f_lr_modelavg);
  dw.register_model_avg(f_handle_loss, f_lr_modelavg);

  double sum = 0.0;
  double loss = 0.0;
  for(int i_epoch=0;i_epoch<10;i_epoch++){
    loss = dw.exec(f_handle_loss)/nexp;
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

#endif
