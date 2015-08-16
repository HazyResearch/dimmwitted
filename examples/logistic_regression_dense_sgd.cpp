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


#ifndef _GLM_DENSE_SGD_H
#define _GLM_DENSE_SGD_H

#include "dimmwitted.h"
#include "dot.h"

using namespace lpblas;

/**
 * \brief A model object. This model contains
 * two elements: (1) p: the pointers
 * to the paramters, and (2) n: the number
 * of paramters that this model contains.
 *
 * Note that, to use PerNode and PerCore
 * strategy, this object needs to have a copy
 * constructor.
 *
 */
class GLMModelExample{
public:
  LPBLAS_i16 * const p;
  int n;
  
  GLMModelExample(int _n):
    n(_n), p(new LPBLAS_i16[_n]){}

  GLMModelExample( const GLMModelExample& other ) :
     n(other.n), p(new LPBLAS_i16[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }

};

/**
 * \brief The function that takes input a series of models,
 * and update one of them according to others. You
 * need to register this one if you want to use
 * PerNode and PerCore strategy.
 */
void f_lr_modelavg(GLMModelExample** const p_models, /**< set of models*/
                   int nreplicas, /**< number of models in the above set */
                   int ireplica /**< id of the model that needs updates*/
                   ){
  std::cerr << "Model averaging function not implemented yet." << std::endl;
  exit(1);
  // GLMModelExample * p_model = p_models[ireplica];
  // float sum = 0.0;
  // for(int i=0;i<p_model->n;i++){
  //   sum = 0.0;
  //   for(int j=0;j<nreplicas;j++){
  //     sum += p_models[j]->p[i];
  //   }
  //   (p_model->p)[i] = sum/nreplicas; // update the ireplica'th model by
  //                                    // the average.
  // }
}

/**
 * \brief One example of the function that can be register to
 * Row-wise access (DW_ROW). This function takes as input
 * one row of the data (ex), and the current model,
 * returns the loss.
 */
float f_lr_loss(const DenseVector<LPBLAS_i16>* const ex, GLMModelExample* const p_model){
  LPBLAS_i16* model = p_model->p;
  float label = (float)(ex->p[ex->n-1]);
  float dot = (float)dot_dense(&ex->p[0], model, ex->n-1);
  return  - label * dot + log(exp(dot) + 1.0);
}

/**
 * \brief One example of the function that can be register to
 * Row-wise access (DW_ROW). This function takes as input
 * one row of the data (ex), and the current model (p_model),
 * and update the model with the gradient.
 * 
 */
float f_lr_grad(const DenseVector<LPBLAS_i16>* const ex, GLMModelExample* const p_model){

  LPBLAS_i16* model = p_model->p;
  const float label = (float)(ex->p[ex->n-1]);

  const float dot = dot_dense(&ex->p[0], model, ex->n-1);

  const float d = exp(-dot);
  const float Z = 0.005 * (-label + 1.0/(1.0+d));

  for(int i=0;i<ex->n-1;i++){
    model[i] -= (LPBLAS_i16)((float)ex->p[i] * Z);
  }

  //axpy(p_model->p, &ex->p[0], Z, ex->n-1);

  return 1.0;
}

/**
 * \brief One example main entry of how to use DimmWitted.
 * The application is Stochastic Gradient Descent (SGD)
 * with Row-wise Access.
 * 
 * \tparam MODELREPL Model replication strategy.
 * \tparam DATAREPL Data replication strategy.
 */
template<ModelReplType MODELREPL, DataReplType DATAREPL>
float test_glm_dense_sgd(){

  // First, create a synthetic data set. 
  // Given nexp examples and nfeat features,
  // this data set contains nexp rows, and
  // nfeat + 1 columns, where the last column
  // is the label that we want to train on.
  // 
  long nexp = 100000; // number of rows
  long nfeat = 1024;  // number of features
  LPBLAS_i16 ** examples = new LPBLAS_i16* [nexp];  // pointers to each row
  LPBLAS_i16 * content = new LPBLAS_i16[nexp*(nfeat+1)];  // buffer to actually hold objects
  for(long i=0;i<nexp;i++){
    examples[i] = &content[i*(nfeat+1)];
    for(int j=0;j<nfeat;j++){
      examples[i][j] = MAX_VALUE<LPBLAS_i16>() / 10;
    }
    examples[i][nfeat] = (LPBLAS_i16)(drand48() > 0.8 ? 0 : 1); // randomly generate labels 
                                                    // with 80% 1 and 20% 0.
  }

  // Second, create a model and initialize it
  // with all zeros.
  //
  GLMModelExample model(nfeat);
  for(int i=0;i<model.n;i++){
    model.p[i] = 0.0;
  }

  // Thrid, create a DenseDimmWitted object because the synthetic data set
  // we created is dense. This object has multiple templates,
  //    - float: the type of the data (type of elements in ``examples'')
  //    - GLMModelExample: the type of the model
  //    - MODELREPL: Model replication strategy
  //    - DATAREPL: Data replication strategy
  //    - DW_ROW: Access method
  //
  DenseDimmWitted<LPBLAS_i16, GLMModelExample, MODELREPL, DATAREPL, DW_ACCESS_ROW> 
    dw(examples, nexp, nfeat+1, &model);

  // Fourth, register functions.
  //
  unsigned int f_handle_grad = dw.register_row(f_lr_grad);
  unsigned int f_handle_loss = dw.register_row(f_lr_loss);
  dw.register_model_avg(f_handle_grad, f_lr_modelavg);
  dw.register_model_avg(f_handle_loss, f_lr_modelavg);

  // Last, run 10 epochs, for each epoch
  //   1. calculate the loss
  //   2. sum the model (only for getting statistics)
  //   3. update the model
  //
  float sum = 0.0;
  for(int i_epoch=0;i_epoch<2;i_epoch++){
    float loss = dw.exec(f_handle_loss)/nexp;
    sum = 0.0;
    for(int i=0;i<nfeat;i++){
      sum += model.p[i];
    }
    std::cout.precision(8);
    std::cout << sum << "    loss=" << loss << std::endl;

    Timer t;
    dw.exec(f_handle_grad);
    float data_byte = 1.0 * sizeof(LPBLAS_i16) * nexp * nfeat;
    float te = t.elapsed();
    float throughput_gb = data_byte / te / 1024 / 1024 / 1024;
    std::cout << "TIME=" << te << " secs" << " THROUGHPUT=" << throughput_gb << " GB/sec." << std::endl;
  }

  // Return the sum of the model. This value should be 
  // around 1.3-1.4 for this example.
  //
  return sum / MAX_VALUE<LPBLAS_i16>();
}

#endif
