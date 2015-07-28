
#include "dw-svm-helper.h"

/**
 * I can make this 10x faster with SIMD. But lets do a slower version first.
 **/
void get_corpus_stats(std::string filename, size_t * n_elements, size_t * n_examples){

  FILE *fp = fopen(filename.c_str(),"r");
  size_t elements, examples;
  char *endptr;
  char *idx, *val, *label;
  ssize_t read;

  if(fp == NULL){
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    exit(1);
  }

  elements = 0; 
  examples = 0;

  size_t max_line_len = 1024;
  char * line = new char[max_line_len];
  
  while ((read = getline(&line, &max_line_len, fp)) != -1) {

    char *p = strtok(line," \t"); // label

    while(1)
    {
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
        break;
      ++elements;
    }
    ++elements;
    ++examples;
  }

  delete[] line;
  fclose(fp);

  *n_elements = elements;
  *n_examples = examples;

}

