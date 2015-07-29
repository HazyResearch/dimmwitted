
#include "dw-lr-helper.h"

void exit_input_error(int line_num)
{
  fprintf(stderr,"Wrong input format at line %d\n", line_num);
  exit(1);
}


/**
 * This code is largely borrowed from LIBLINEAR.
 * TODO: This can be made faster with SIMD.
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
 * This code is largely borrowed from LIBLINEAR.
 * TODO: This can be made faster with SIMD.
 **/
size_t create_dw_corpus(std::string filename, const size_t n_elements, const size_t n_examples,
  double * & p_examples, long * & p_cols, long * & p_rows){

  p_examples = new double[n_elements];
  p_cols = new long[n_elements];
  p_rows = new long[n_examples];

  FILE *fp = fopen(filename.c_str(),"r");
  ssize_t read;
  size_t max_line_len = 1024;
  char * line = new char[max_line_len];
  char *endptr;
  char *idx, *val, *label;

  double * y = new double[n_examples];

  int input_idx;
  double input_label;
  double input_value;

  if(fp == NULL){
    fprintf(stderr,"can't open input file %s\n",filename.c_str());
    exit(1);
  }

  size_t maxidx = 0;
  int inst_max_index, j=0;
  size_t ct = 0;
  for(int i=0;i<n_examples;i++){
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    
    read = getline(&line, &max_line_len, fp);

    label = strtok(line," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);

    input_label = strtod(label,&endptr);
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);

    p_rows[i] = ct;

    while(1){
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");

      if(val == NULL)
        break;

      errno = 0;
      input_idx = (int) strtol(idx,&endptr,10);
      if(endptr == idx || errno != 0 || *endptr != '\0')
        exit_input_error(i+1);

      errno = 0;
      input_value = strtod(val,&endptr);
      if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
        exit_input_error(i+1);

      ++j;

      p_examples[ct] = input_value;
      p_cols[ct] = input_idx;
      ct ++;

      //std::cout << input_idx << std::endl;
      if(input_idx > maxidx){
        maxidx = input_idx;
      }
    }
    p_examples[ct] = (input_label+1)/2; // normalize +1/-1 to 1/0
    p_cols[ct] = -1;
    ct ++;

  }
  maxidx ++;

  return maxidx;
}
