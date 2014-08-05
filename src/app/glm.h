
#include "common.h"
#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"


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
  std::cout << i_task << std::endl;
  rddata->update(rddata->examples[i_task], wrdata->model, rddata->labels[i_task], rddata->nfeat);
}

void glm_comm (GLMModel * const a, const GLMModel ** const b, int nreplicas){

}

void glm_finalize (GLMModel * const a, GLMModel ** const b, int nreplicas){

}

void glm_model_allocator (GLMModel ** const a, const GLMModel * const b){
}


void lr_update (const double * const ex, double * const model, double label, int nfeat){
  
  double dot = 0;
  for(int i=0;i<nfeat;i++){
    dot += ex[i] * model[i];
  }
  double d = exp(dot);
  double Z = -label + d/(1.0+d);
  for(int i=0;i<nfeat;i++){
    model[i] -= 0.0001 * ex[i] * Z;
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


void glm_do(){


  /*
  long nexp = 10000;
  long nfeat = 10;
  GLMData data;
  data.nexp = nexp;
  data.nfeat = nfeat;
  data.examples = new double*[nexp];
  data._memory_buf = new double[nexp*nfeat];
  for(long i=0;i<nexp;i++){
    data.examples[i] = &data._memory_buf[i*nfeat];
  }
  data.labels = new double[nexp];
  data.update = &lr_update;

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

  DWRun<GLMData, GLMModel, glm_map, glm_comm, glm_finalize, SCHED_STRAWMAN> 
    dw(&data, &model, tasks, nexp, glm_model_allocator);

  dw.prepare();

  for(int j=0;j<100;j++){
    dw.exec();
    for(int i=0;i<nfeat;i++){
      std::cout << model.model[i] << std::endl;
    }
  }
  */
}
















