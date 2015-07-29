
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <iostream>

#include "dw-lr-helper.h"
#include "dw-lr-model.h"

/**
 * The main entry of dw-lr-train, whose goal
 * is to takes as input in the same format as
 * libsvm, and train a classifier.
 **/
int main(int argc, char ** argv){

  /**
   * First, parse command line input to get
   *    (1) step size
   *    (2) number of epoches to run
   *    (3) regularization (l2)
   **/  
  std::string filename;
  int flags, opt;
  int nsecs, tfnd;
  nsecs = 0;
  tfnd = 0;
  flags = 0;
  while ((opt = getopt(argc, argv, "s:e:r:")) != -1) {
    switch (opt) {
    case 's':
      stepsize = atof(optarg);
      break;
    case 'e':
      epoches = atof(optarg);
      break;
    case 'r':
      lambda = atof(optarg);
      break;
    default: 
      fprintf(stderr, "Usage: %s [-s stepsize] [-e epoches] [-r regularization] trainfile\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (optind >= argc) {
      fprintf(stderr, "Expected argument after options\n");
      exit(EXIT_FAILURE);
  }
  filename = std::string(argv[optind]);
  printf("stepsize=%f; epoches=%f; lambda=%f\n", stepsize, epoches, lambda);
  printf("trainfile = %s\n", filename.c_str());

  /**
   * Second, create corpus
   **/
  size_t n_elements, n_examples, n_features;
  double * p_examples;
  long * p_cols;
  long * p_rows;
  get_corpus_stats(filename.c_str(), &n_elements, &n_examples);
  n_features = create_dw_corpus(filename.c_str(), n_elements, n_examples, p_examples, p_cols, p_rows);
  printf("#elements=%zu; #examples=%zu; #n_features=%zu\n", n_elements, n_examples, n_features);

  /**
   * Third, create DimmWitted object, and let it run.
   **/
  GLMModelExample_Sparse model(n_features);
  for(int i=0;i<model.n;i++){
    model.p[i] = 0.0;
  }

  SparseDimmWitted<double, GLMModelExample_Sparse, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW> 
    dw(p_examples, p_rows, p_cols, n_examples, n_features+1, n_elements, &model);
  
  unsigned int f_handle_grad = dw.register_row(f_lr_grad_sparse);
  unsigned int f_handle_loss = dw.register_row(f_lr_loss_sparse);
  dw.register_model_avg(f_handle_grad, f_lr_modelavg);
  dw.register_model_avg(f_handle_loss, f_lr_modelavg);

  printf("Start training...\n");

  double sum = 0.0;
  for(int i_epoch=0;i_epoch<epoches;i_epoch++){
    double loss = dw.exec(f_handle_loss)/n_examples;
    std::cout << "loss=" << loss << std::endl;
    dw.exec(f_handle_grad);
  }

  /**
   * Forth, dump result.
   **/
   printf("Dumping training result to %s.model...\n", filename.c_str());
  std::ofstream fout((filename + ".model").c_str());
  fout << n_features << std::endl;
  for(int i=0;i<model.n;i++){
    fout << model.p[i] << std::endl;
  }
  fout.close();

  exit(EXIT_SUCCESS);
}










