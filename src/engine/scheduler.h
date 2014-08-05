
#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include "common.h"

enum ScheduleType{
  SCHED_STRAWMAN,
  SCHED_HOGWILD,
  SCHED_PERCORE
};

/*! \p transpose : transpose a matrix
 *
 * Design document can be found at https://github.com/zhangce/dw/wiki/(Design-Doc)-Scheduler
 *
 */
template<class RDTYPE,
         class WRTYPE, 
         ScheduleType SCHEDTYPE>
class DWRun {
public:

  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  void (*p_model_allocator) (WRTYPE * const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator)
  {}


  void prepare();

  void exec(const long * const tasks, int ntasks,
         void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE * const, const WRTYPE ** const, int),
         void (*p_finalize) (WRTYPE * const, WRTYPE ** const, int)
    );

};



#endif






