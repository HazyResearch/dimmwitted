
#include "common.h"
#include "engine/scheduler.h"
#include "engine/scheduler_strawman.h"
#include "engine/scheduler_hogwild.h"
#include "engine/scheduler_percore.h"

#include "app/glm.h"
#include "app/cnn.h"

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

int main(int argc, char** argv){
  std::cout << "Hello World!" << std::endl;

  cnn_do();

  //glm_do();

  /*
  double rddata[100000];
  for(int i=0;i<100000;i++){
    rddata[i] = i;
  }
  double wrdata[1] = {0.0f};
  long tasks[100000];
  for(int i=0;i<100000;i++){
    tasks[i] = i;
  }  

  std::cout << "START!" << std::endl;

  //DWRun<double, double, p_map, p_comm, p_finalize, SCHED_STRAWMAN> dw(rddata, wrdata, tasks, 100000);
  //DWRun<double, double, p_map, p_comm, p_finalize, SCHED_HOGWILD> dw(rddata, wrdata, tasks, 100000, p_model_allocator);
  DWRun<double, double, p_map, p_comm, p_finalize, SCHED_PERCORE> dw(rddata, wrdata, tasks, 100000, p_model_allocator);

  dw.prepare();
  dw.exec();

  std::cout << "SUM = " << wrdata[0] << std::endl;
  */

  return 0;
}