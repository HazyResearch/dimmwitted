
#ifndef _DW_SVM_HELPER_H
#define _DW_SVM_HELPER_H

#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


/**
 * Given a file in LIBSVM's format, outputs the number of distinct
 * numbers (including both the label and features) in the whole file,
 * and also the number of examples.
 *
 * For LIBSVM, the first number is the number of ' ' and '\t', and for the 
 * second number is the number of '\n'.
 *
 **/
void get_corpus_stats(std::string filename, size_t * const n_elements, size_t * const n_examples);

/**
 * Given a file in LIBSVM's format and the number of elements
 * and examples, create the data structure that DW's sparse
 * engine can take as input as in examples/logistic_regression_sparse_sgd.cpp
 *
 * This function should be used after `get_corpus_stats` to get statistics.
 *
 **/
size_t create_dw_corpus(std::string filename, const size_t n_elements, const size_t n_examples,
	double * & p_examples, long * & p_cols, long * & p_rows);

#endif