
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

/**
 * I can make this 10x faster with SIMD. But lets do a slower version first.
 **/
void create_dw_corpus(std::string filename, const size_t n_elements, const size_t n_examples,
  double * const p_examples, long * const p_cols, long * const p_rows){

  double * p_examples = new double[n_elements];
  long * p_cols = new long[n_elements];
  long * p_rows = new long[n_examples];

  std::ifstream fin(filename.c_str());

  



  fin.close();



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




}
