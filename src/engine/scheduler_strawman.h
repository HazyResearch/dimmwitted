#ifndef _SCHEDULER_STRAWMAN_H
#define _SCHEDULER_STRAWMAN_H

#include "engine/scheduler.h"

template<class RDTYPE,
         class WRTYPE>
class DWRun<RDTYPE, WRTYPE, SCHED_STRAWMAN> {  
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
    std::cout << ntasks << std::endl;
    for(long i=0;i<ntasks;i++){
      p_map(tasks[i], RDPTR, WRPTR);
    }
  }

};


#endif