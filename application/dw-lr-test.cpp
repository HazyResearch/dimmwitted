
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <iostream>

#include "dw-lr-model.h"
#include "dw-lr-helper.h"


/**
 * The main entry of dw-lr-test, whose goal
 * is to takes as input in the same format as
 * libsvm, and test a classifier.
 **/
int main(int argc, char ** argv){

  /**
   * First, parse command line input to get
   *    (1) step size
   *    (2) number of epoches to run
   *    (3) regularization (l2)
   **/  
  if(argc != 4){
    fprintf(stderr, "Usage: %s test_file model_file output_file\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  std::string test_file = std::string(argv[1]);
  std::string model_file = std::string(argv[2]);
  std::string output_file = std::string(argv[3]);

  /**
   * First, create corpus
   **/
  size_t n_elements, n_examples, n_features_test;
  double * p_examples;
  long * p_cols;
  long * p_rows;
  get_corpus_stats(test_file.c_str(), &n_elements, &n_examples);
  n_features_test = create_dw_corpus(test_file.c_str(), n_elements, n_examples, p_examples, p_cols, p_rows);

  /**
  * First, load the model.
   **/
  size_t n_features_train;
  std::ifstream fin(model_file.c_str());
  fin >> n_features_train;
  printf("#elements=%zu; #examples=%zu; #n_features_test=%zu; #n_features_train=%zu\n", n_elements, n_examples, n_features_test, n_features_train);
  size_t n_features = n_features_train > n_features_test ? n_features_train : n_features_test;
  printf("#n_features=%zu\n", n_features);
  GLMModelExample_Sparse model(n_features);
  for(int i=0;i<model.n;i++){
    fin >> model.p[i];
  }

  /**
   * Third, create DimmWitted object, and let it run.
   **/
  SparseDimmWitted<double, GLMModelExample_Sparse, DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING, DW_ACCESS_ROW> 
    dw(p_examples, p_rows, p_cols, n_examples, n_features+1, n_elements, &model);
  
  unsigned int f_handle_accuracy = dw.register_row(f_lr_accuracy_sparse);
  unsigned int f_handle_loss = dw.register_row(f_lr_loss_sparse);
  dw.register_model_avg(f_handle_accuracy, f_lr_modelavg);
  dw.register_model_avg(f_handle_loss, f_lr_modelavg);

  printf("Start testing...\n");

  double loss = dw.exec(f_handle_loss)/n_examples;
  std::cout << "Testing loss=" << loss << std::endl;

  double accuracy = dw.exec(f_handle_accuracy)/n_examples;
  std::cout << "Testing acc =" << accuracy << std::endl;

  /**
   * Forth, dump result.
   **/
  printf("Dumping the result to %s...\n", output_file.c_str());
  SparseDimmWitted<double, GLMModelExample_Sparse, DW_MODELREPL_SINGLETHREAD_DEBUG, DW_DATAREPL_SHARDING, DW_ACCESS_ROW> 
    dw_dumper(p_examples, p_rows, p_cols, n_examples, n_features+1, n_elements, &model);
  unsigned int f_handle_dumper = dw_dumper.register_row(f_lr_dump_sparse);
  dw_dumper.dump_row(f_handle_dumper, output_file);

  exit(EXIT_SUCCESS);
}










