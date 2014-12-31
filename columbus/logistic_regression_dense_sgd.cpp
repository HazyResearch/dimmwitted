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
  double * const p;
  int n;
  
  GLMModelExample(int _n):
    n(_n), p(new double[_n]){}

  GLMModelExample( const GLMModelExample& other ) :
     n(other.n), p(new double[other.n]){
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
  GLMModelExample * p_model = p_models[ireplica];
  double sum = 0.0;
  for(int i=0;i<p_model->n;i++){
    sum = 0.0;
    for(int j=0;j<nreplicas;j++){
      sum += p_models[j]->p[i];
    }
    (p_model->p)[i] = sum/nreplicas; // update the ireplica'th model by
                                     // the average.
  }
}

/**
 * \brief One example of the function that can be register to
 * Row-wise access (DW_ROW). This function takes as input
 * one row of the data (ex), and the current model,
 * returns the loss.
 */
double f_lr_loss(const DenseVector<double>* const ex, GLMModelExample* const p_model){
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[i];
  }
  return  - label * dot + log(exp(dot) + 1.0);
}

/**
 * \brief One example of the function that can be register to
 * Row-wise access (DW_ROW). This function takes as input
 * one row of the data (ex), and the current model (p_model),
 * and update the model with the gradient.
 * 
 */
double f_lr_grad(const DenseVector<double>* const ex, GLMModelExample* const p_model){

  double * model = p_model->p;
  double label = ex->p[ex->n-1];

  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[i];
  }

  const double d = exp(-dot);
  const double Z = 0.00001 * (-label + 1.0/(1.0+d));

  for(int i=0;i<ex->n-1;i++){
    model[i] -= ex->p[i] * Z;
  }

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
double test_glm_dense_sgd(){

  // First, create a synthetic data set. 
  // Given nexp examples and nfeat features,
  // this data set contains nexp rows, and
  // nfeat + 1 columns, where the last column
  // is the label that we want to train on.
  // 
  long nexp = 100000; // number of rows
  long nfeat = 1024;  // number of features
  double ** examples = new double* [nexp];  // pointers to each row
  double * content = new double[nexp*(nfeat+1)];  // buffer to actually hold objects
  for(long i=0;i<nexp;i++){
    examples[i] = &content[i*(nfeat+1)];
    for(int j=0;j<nfeat;j++){
      examples[i][j] = 1;
    }
    examples[i][nfeat] = drand48() > 0.8 ? 0 : 1.0; // randomly generate labels 
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
  //    - double: the type of the data (type of elements in ``examples'')
  //    - GLMModelExample: the type of the model
  //    - MODELREPL: Model replication strategy
  //    - DATAREPL: Data replication strategy
  //    - DW_ROW: Access method
  //
  DenseDimmWitted<double, GLMModelExample, MODELREPL, DATAREPL, DW_ACCESS_ROW> 
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
  double sum = 0.0;
  for(int i_epoch=0;i_epoch<2;i_epoch++){
    double loss = dw.exec(f_handle_loss)/nexp;
    sum = 0.0;
    for(int i=0;i<nfeat;i++){
      sum += model.p[i];
    }
    std::cout.precision(8);
    std::cout << sum << "    loss=" << loss << std::endl;

    Timer t;
    dw.exec(f_handle_grad);
    double data_byte = 1.0 * sizeof(double) * nexp * nfeat;
    double te = t.elapsed();
    double throughput_gb = data_byte / te / 1024 / 1024 / 1024;
    std::cout << "TIME=" << te << " secs" << " THROUGHPUT=" << throughput_gb << " GB/sec." << std::endl;
  }

  // Return the sum of the model. This value should be 
  // around 1.3-1.4 for this example.
  //
  return sum;
}

#endif
