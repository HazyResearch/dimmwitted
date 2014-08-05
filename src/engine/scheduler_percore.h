#ifndef _SCHEDULER_PERCORE_H
#define _SCHEDULER_PERCORE_H

#include "engine/scheduler.h"


template<class RDTYPE, class WRTYPE>
void _percore_run_map(void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
  const RDTYPE * const RDPTR, 
              WRTYPE * const WRPTR, 
              const long * const tasks,
              long start, long end){
  for(long i=start;i<end;i++){
    p_map(tasks[i], RDPTR, WRPTR);
  }
}

template<class RDTYPE,
         class WRTYPE>
class DWRun<RDTYPE, WRTYPE, SCHED_PERCORE> {  
public:
  
  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  WRTYPE ** model_replicas;

  void (*p_model_allocator) (WRTYPE ** const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator)
  {}


  void prepare(){
    long n_sharding = 8;
    model_replicas = new WRTYPE*[n_sharding];
    for(int i=0;i<n_sharding;i++){
      p_model_allocator(&model_replicas[i], WRPTR);
    }
  }

  void exec(const long * const tasks, int ntasks,
    void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE * const, const WRTYPE ** const, int),
         void (*p_finalize) (WRTYPE * const, WRTYPE ** const, int)
    ){

    std::vector<std::thread> threads;

    long n_sharding = 4;
    for(int i_sharding=0;i_sharding<n_sharding;i_sharding++){
      long start = ((long)(ntasks/n_sharding)+1) * i_sharding;
      long end = ((long)(ntasks/n_sharding)+1) * (i_sharding+1);
      end = end >= ntasks ? ntasks : end;
      threads.push_back(std::thread(_percore_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, start, end));
    }

    for(int i=0;i<n_sharding;i++){
      threads[i].join();
    }

    p_finalize(WRPTR, model_replicas, n_sharding);


  }

};


#endif





