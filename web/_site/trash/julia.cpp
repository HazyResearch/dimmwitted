
#include "julia.h"
#include "common.h"
#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"

#include "app/glm.h"











void glm_sgd(double * examples, double * labels, double * model, long nexp, long nfeat, 
  void(* loss)(const double * const, double * const, double, int)){
  
  
  _glm(examples, labels, model, nexp, nfeat, loss);

}

void p_map (long i_task, const double * const rddata, double * const wrdata){
  wrdata[0] += rddata[i_task];
}

void p_comm (double * const a, const double ** const b, int nreplicas){

}

void p_finalize (double * const a, double ** const b, int nreplicas){
  for(int i=0;i<nreplicas;i++){
    a[0] += b[i][0];
  }
}

void p_model_allocator (double ** const a, const double * const b){
  *a = (double *) malloc(sizeof(double));
  **a = *b;
}

void p_data_allocator (double ** const a, const double * const b){

}

void strawman_sum(double * numbers, double * output, long* indexes, int nnumbers){
  DWRun<double, double, p_map, p_comm, p_finalize, SCHED_STRAWMAN> dw(numbers, output, indexes, nnumbers, p_model_allocator);
  dw.prepare();
  dw.exec();
}

void hogwild_sum(double * numbers, double * output, long* indexes, int nnumbers){
  DWRun<double, double, p_map, p_comm, p_finalize, SCHED_HOGWILD> dw(numbers, output, indexes, nnumbers, p_model_allocator);
  dw.prepare();
  dw.exec();
}

void percore_sum(double * numbers, double * output, long* indexes, int nnumbers){
  DWRun<double, double, p_map, p_comm, p_finalize, SCHED_PERCORE> dw(numbers, output, indexes, nnumbers, p_model_allocator);
  dw.prepare();
  dw.exec();
}