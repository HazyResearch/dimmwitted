#ifndef _SCHEDULER_HOGWILD_H
#define _SCHEDULER_HOGWILD_H

#include "engine/scheduler.h"


template<class RDTYPE, class WRTYPE>
void _hogwild_run_map(void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
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
class DWRun<RDTYPE, WRTYPE, SCHED_HOGWILD> {  
public:
  
  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  void (*p_model_allocator) (WRTYPE ** const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator)
  {}

  void prepare(){

  }

  void exec(const long * const tasks, int ntasks,
    void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE * const, const WRTYPE ** const, int),
         void (*p_finalize) (WRTYPE * const, WRTYPE ** const, int)
    ){

    std::vector<std::thread> threads;

    //int n_numa_nodes = numa_max_node();
    //long n_sharding = getNumberOfCores()/(n_numa_nodes+1);
 
    int n_sharding = 8;

    for(int i_sharding=0;i_sharding<n_sharding;i_sharding++){
      long start = ((long)(ntasks/n_sharding)+1) * i_sharding;
      long end = ((long)(ntasks/n_sharding)+1) * (i_sharding+1);
      end = end >= ntasks ? ntasks : end;
      threads.push_back(std::thread(_hogwild_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, WRPTR, tasks, start, end));
    }

    for(int i=0;i<n_sharding;i++){
      threads[i].join();
    }

  }

};


#endif

