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


#ifndef _CYC_SPARSE_SGD_H
#define _CYC_SPARSE_SGD_H

#include "dimmwitted.h"

/**
 * \brief This file shows how to specify the same
 * synthetic model as in app/cyc_dense_sgd.h
 * but store the data as sparse matrix instead
 * of dense matrix.
 *
 * See app/cyc_dense_sgd.h for more detailed 
 * comments.
 */
class CYCModelExample_Sparse{
public:
  double * const p;
  int n;
  
  CYCModelExample_Sparse(int _n):
    n(_n), p(new double[_n]){}

  CYCModelExample_Sparse( const CYCModelExample_Sparse& other ) :
     n(other.n), p(new double[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }

};

void f_lr_modelavg(CYCModelExample_Sparse** const p_models, int nreplicas, int ireplica){
  CYCModelExample_Sparse * p_model = p_models[ireplica];
  double sum = 0.0;
  for(int i=0;i<p_model->n;i++){
    sum = 0.0;
    for(int j=0;j<nreplicas;j++){
      sum += p_models[j]->p[i];
    }
    (p_model->p)[i] = sum/nreplicas;
  }
}


double f_lr_loss_sparse(const SparseVector<double>* const ex, CYCModelExample_Sparse* const p_model){

  /*
  double * model = p_model->p;
  double label = ex->p[ex->n-1];
  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[ex->idxs[i]];
  }
  return  - label * dot + log(exp(dot) + 1.0);
  */
  return 0.0;
}


double f_lr_grad_sparse(const SparseVector<double>* const ex, CYCModelExample_Sparse* const p_model){

  /*
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
  */
  return 1.0;
}

template<ModelReplType MODELREPL, DataReplType DATAREPL>
double test_cyc_sparse_sgd(){

//  numa_run_on_node(numa_node);
//  numa_set_localalloc();

  int NNUMA_NODE = numa_max_node() + 1;

  // First, create a synthetic data set. 
  // Given nexp examples and nfeat features,
  // this data set contains nexp rows, and
  // nfeat + 1 columns, where the last column
  // is the label that we want to train on.
  // 
  long nexp = 100; // number of rows
  long nfeat = 10;  // number of features
  long nbatches = 10; // number of batches
  long nepoch = 10;

  long nexp_per_batch = (int)ceil((double)nexp / (double)nbatches);
  long * batch_nexp = new long[nbatches];

  double ** examples_all = new double*[nbatches];
  long ** cols_all = new long*[nbatches];
  long ** rows_all = new long*[nbatches];

  std::vector<long> randperm;
  for (int i=0; i<nfeat; i++){
    randperm.push_back(i);
  }

  for (long ibatch = 0; ibatch < nbatches; ibatch++){
    // Randomly shuffle the features to be assigned to each core in this batch
    std::random_shuffle(randperm.begin(), randperm.end());

    // examples in this batch [batchBegIndex, batchEndIndex)
    long batchBegIndex = ibatch * nexp_per_batch;
    long batchEndIndex = batchBegIndex + nexp_per_batch;
    if (batchEndIndex >= nexp){
      batchEndIndex = nexp - 1;
    }
    batch_nexp[ibatch] = batchEndIndex - batchBegIndex;
    // Number of examples in the batch needs to be a multiple of number of NUMA nodes
    // We'll pad with noops to ensure this later
    double ndata_per_node_double = (double)(batch_nexp[ibatch]) / (double)NNUMA_NODE;
    long batchNumExamples = ceil(ndata_per_node_double) * NNUMA_NODE;
    double * examples = new double [batchNumExamples * (nfeat+1)];  // pointers to each row
    long * rows = new long[batchNumExamples];
    long * cols = new long[batchNumExamples * (nfeat+1)];
    examples_all[ibatch] = examples;
    rows_all[ibatch] = rows;
    cols_all[ibatch] = cols;

    long exampleIndex = 0;
    for(long inuma=0;inuma < NNUMA_NODE; inuma ++){
      numa_run_on_node(inuma);
      numa_set_localalloc();

      // double nodeBegIndex_double = (double) inuma    * ndata_per_node_double;
      double nodeEndIndex_double = (double)(inuma+1) * ndata_per_node_double;

      long ct = 0;

      long nodeExampleIndex = 0;
      for (; (double)exampleIndex < nodeEndIndex_double; exampleIndex++){
        rows[exampleIndex] = ct;
        for(int j=0;j<nfeat;j++){
          examples[ct] = drand48();
          cols[ct] = j;
          ct++;
        }
        examples[ct] = drand48() > 0.8 ? 0 : 1.0; // randomly generate labels 
        cols[ct] = nfeat;
        ct++;

        nodeExampleIndex ++;
      }
      for (; nodeExampleIndex < ceil(ndata_per_node_double); exampleIndex++){
        rows[exampleIndex] = ct;
        for(int j=0;j<nfeat;j++){
          examples[ct] = drand48();
          cols[ct] = j;
          ct++;
        }
        examples[ct] = -1; // indicate noop
        cols[ct] = nfeat;
        ct++;

        nodeExampleIndex++;
      }

    }
  }

  // Sanity check of examples assignments
  if (true){
    for (long ibatch = 0; ibatch < nbatches; ibatch++){
      for (long iexample = 0; iexample < batch_nexp[ibatch]; iexample++){
        std::cout << "Batch " << ibatch << ", example " << iexample << std::endl << "\t\t";
        for (long ifeat = 0; ifeat < nfeat+1; ifeat++){
          std::cout << cols_all[ibatch][rows_all[ibatch][iexample] + ifeat] << ":" << examples_all[ibatch][rows_all[ibatch][iexample] + ifeat] << "\t";
        }
        std::cout << std::endl;
      }
    }
  }


  CYCModelExample_Sparse model(nfeat);
  for(int i=0;i<model.n;i++){
    model.p[i] = 0.0;
  }

  // Create DW engines
  SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> ** dwEngines = new SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW>*[nbatches];
  unsigned int * f_handle_grads = new unsigned int[nbatches];
  unsigned int * f_handle_losss = new unsigned int[nbatches];

  for (long ibatch = 0; ibatch < nbatches; ibatch++){
    // Thrid, create a DenseDimmWitted object because the synthetic data set
    // we created is dense. This object has multiple templates,
    //    - double: the type of the data (type of elements in ``examples'')
    //    - CYCModelExample: the type of the model
    //    - MODELREPL: Model replication strategy
    //    - DATAREPL: Data replication strategy
    //    - DW_ROW: Access method
    //
    SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> *
      dw = new SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> (examples_all[ibatch], rows_all[ibatch], cols_all[ibatch], batch_nexp[ibatch], nfeat+1, batch_nexp[ibatch]*(nfeat+1), &model);
    unsigned int f_handle_grad = dw->register_row(f_lr_grad_sparse);
    unsigned int f_handle_loss = dw->register_row(f_lr_loss_sparse);
    dw->register_model_avg(f_handle_grad, f_lr_modelavg);
    dw->register_model_avg(f_handle_loss, f_lr_modelavg);
    dwEngines[ibatch] = dw;
    f_handle_grads[ibatch] = f_handle_grad;
    f_handle_losss[ibatch] = f_handle_loss;

    std::cout << f_handle_grads[ibatch]  << std::endl;
    std::cout << f_handle_losss[ibatch]  << std::endl;

    std::cout << "~~~~~~~~~" << std::endl;
    dw->exec(f_handle_loss);
    std::cout << "~~~~~~~~~" << std::endl;
  }


  double l2sum = 0.0;
  for (long iepoch = 0; iepoch < nepoch; iepoch++){
    // std::cout << "Running epoch " << iepoch << "\n";
    double loss = 0.0;
    for (long ibatch = 0; ibatch < nbatches; ibatch++){
      std::cout << "epoch " << iepoch << ",\tbatch " << ibatch << "\n";
      SparseDimmWitted<double, CYCModelExample_Sparse, MODELREPL, DATAREPL, DW_ACCESS_ROW> * dw = dwEngines[ibatch];
      // dw->exec(f_handle_grads[ibatch]);
      loss = dw->exec(f_handle_losss[ibatch]);
      std::cout << "\tloss = " << loss << std::endl;
      for (long ifeat = 0; ifeat < nfeat; ifeat++){
        std::cout << model.p[ifeat] << ", ";
      }
      std::cout << std::endl;
    }
    loss /= nexp;

    l2sum = 0.0;
    for (int i = 0; i<nfeat; i ++){
      l2sum += model.p[i] * model.p[i];
    }

    std::cout.precision(8);
    std::cout << l2sum << "    loss=" << loss << std::endl;
  }


  // Return the sum of the model. This value should be 
  // around 1.3-1.4 for this example.
  //
  return l2sum;

}

/**
 * \brief This is one example of running SGD for logistic regression
 * in DimmWitted. You can find more examples in test/cyc_dense.cc
 * and test/cyc_sparse.cc, and the documented code in 
 * app/cyc_dense_sgd.h
 */
//int main(int argc, char** argv){
//  double rs = test_cyc_sparse_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_SHARDING>();
//  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
//  return 0;
//}

#endif

