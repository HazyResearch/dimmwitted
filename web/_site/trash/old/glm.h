
#include "common.h"
#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"
#include "engine/scheduler_pernode.h"

#include <xmmintrin.h>
#include <immintrin.h>
#include <avxintrin.h>

struct GLMData{
  long nexp;
  long nfeat;
  double ** examples;
  double *  labels;
  double *  _memory_buf;
  void (*update) (const double * const, double * const, double, int);
};

struct GLMModel{
  long nfeat;
  double * model;
};

void glm_map (long i_task, const GLMData * const rddata, GLMModel * const wrdata){
  //std::cout << i_task << std::endl;
  rddata->update(rddata->examples[i_task], wrdata->model, rddata->labels[i_task], rddata->nfeat);
}

void glm_comm (GLMModel * const a, GLMModel ** const b, int nreplicas){

  double sum = 0.0;
  for(int i=0;i<a->nfeat;i++){
    sum = 0.0;
    for(int j=0;j<nreplicas;j++){
      sum += b[j]->model[i];
    }
    a->model[i] = sum/nreplicas;
  }
}

void glm_finalize (GLMModel * const a, GLMModel ** const b, int nreplicas){
  
  for(int i=0;i<a->nfeat;i++){
    a->model[i] = 0.0;
  }

  for(int j=0;j<nreplicas;j++){
    for(int i=0;i<a->nfeat;i++){
      a->model[i] += b[j]->model[i]/nreplicas;
    }
  }
}

void glm_model_allocator (GLMModel ** const a, const GLMModel * const b){
  *a = new GLMModel();
  (*a)->nfeat = b->nfeat;
  (*a)->model = new double[b->nfeat];
  memcpy((*a)->model, b->model, sizeof(double)*b->nfeat);
}


void lr_update_naive (const double * __restrict__ const ex, double * __restrict__ const model, double label, int nfeat){
  
  double dot = 0;
  for(int i=0;i<nfeat;i++){
    dot += ex[i] * model[i];
  }

  const double d = exp(dot);
  const double Z = 0.0001 * (-label + d/(1.0+d));

  for(int i=0;i<nfeat;i++){
    model[i] -= ex[i] * Z;
  }

}

void lr_update_unroll (const double * __restrict__ const ex, double * __restrict__ const model, double label, int nfeat){
  
  double dot = 0;
  double dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0;
  for(int i=0;i<nfeat;i+=4){
    dot1 += ex[i] * model[i];
    dot2 += ex[i+1] * model[i+1];
    dot3 += ex[i+2] * model[i+2];
    dot4 += ex[i+3] * model[i+3];
  }
  dot = dot1 + dot2 + dot3 + dot4;

  const double d = exp(dot);
  const double Z = 0.0001 * (-label + d/(1.0+d));

  for(int i=0;i<nfeat;i++){
    model[i] -= ex[i] * Z;
  }

}

void lr_update_sse (const double * __restrict__ const ex, double * __restrict__ const model, double label, int nfeat){
  
  __m256d aa, bb, cc, ss;
  ss = _mm256_set1_pd(0);
  double s[4];

  for(int i = 0 ; i < nfeat ; i += 4) {
     aa = _mm256_load_pd(ex + i);
     bb = _mm256_load_pd(model + i);
     cc = _mm256_mul_pd(aa, bb);
     ss = _mm256_add_pd(ss, cc);
  }

  _mm256_store_pd(s, ss);
  double dot = s[0] + s[1] + s[2] + s[3];

  const double d = exp(-dot);
  const double Z = 0.0001 * (-label + 1.0/(1.0+d));

  for(int i=0;i<nfeat;i++){
    model[i] -= ex[i] * Z;
  }

}

void _glm(double * examples, double * labels, double * modelvec, long nexp, long nfeat, 
  void(* loss)(const double * const, double * const, double, int)){

  /*
  GLMData data;
  data.nexp = nexp;
  data.nfeat = nfeat;
  data.examples = new double*[nexp];
  data._memory_buf = examples;
  for(long i=0;i<nexp;i++){
    data.examples[i] = &data._memory_buf[i*nfeat];
  }
  data.labels = labels;
  data.update = loss;

  GLMModel model;
  model.nfeat = nfeat;
  model.model = modelvec;

  std::cout << "#####" << std::endl;

  long * tasks = new long[nexp];
  for(int i=0;i<nexp;i++){
    tasks[i] = i;
  }  

  std::cout << "*****" << std::endl;

  DWRun<GLMData, GLMModel, glm_map, glm_comm, glm_finalize, SCHED_STRAWMAN> 
    dw(&data, &model, tasks, nexp, glm_model_allocator);

  dw.prepare();
  dw.exec();
  */

}

/*
void glm_do(){

  long nexp = 100000;
  long nfeat = 1024;
  GLMData data;
  data.nexp = nexp;
  data.nfeat = nfeat;
  data.examples = new double*[nexp];
  data._memory_buf = new double[nexp*nfeat];
  for(long i=0;i<nexp;i++){
    data.examples[i] = &data._memory_buf[i*nfeat];
  }
  data.labels = new double[nexp];
  data.update = &lr_update_unroll;

  GLMModel model;
  model.nfeat = nfeat;
  model.model = new double[nfeat];
  for(long j=0;j<nfeat;j++){
    model.model[j] = 0;
  }

  for(long i=0;i<nexp;i++){
    for(long j=0;j<nfeat;j++){
      data.examples[i][j] = 1;
    }
    data.labels[i] = drand48() > 0.8 ? 0 : 1.0;
  }

  long * tasks = new long[nexp];
  for(int i=0;i<nexp;i++){
    tasks[i] = i;
  }  

  DWRun<GLMData, GLMModel, DW_STRAWMAN> 
    dw(&data, &model, glm_model_allocator);

  dw.prepare();

  double size_in_bit = nexp*nfeat*64;
  double size_in_byte= size_in_bit/8;

  for(int j=0;j<100;j++){
    Timer t;
    dw.exec(tasks, nexp, glm_map, glm_comm, glm_finalize);

    std::cout << "+++++" << std::endl;

    double ti = t.elapsed();
    double tp = size_in_byte/ti;
    std::cout << ti << " seconds!" << std::endl;
    std::cout << tp/1024/1024 << " MB/secs!" << std::endl;

    double sum = 0.0;
    for(int i=0;i<nfeat;i++){
      sum += model.model[i];
    }

    std::cout << sum << std::endl;

  }
  
}
*/















