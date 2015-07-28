
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
void get_corpus_stats(std::string filename, long * n_elements, long * n_examples);


#endif