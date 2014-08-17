// Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#ifndef _SCHEDULER_HOGWILD_H
#define _SCHEDULER_HOGWILD_H

#include "engine/scheduler.h"

template<class RDTYPE, class WRTYPE>
double _hogwild_run_map(double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
  const RDTYPE * const RDPTR, 
              WRTYPE * const WRPTR, 
              const long * const tasks,
              long start, long end){
  double rs = 0.0;
  for(long i=start;i<end;i++){
    rs += p_map(tasks[i], RDPTR, WRPTR);
  }
  return rs;
}

/**
 * \brief A specialization of DWRun that maintains a single
 * model replica per machine and one thread per core.
 */
template<class RDTYPE,
         class WRTYPE,
         DataReplType DATAREPL>
class DWRun<RDTYPE, WRTYPE, DW_MODELREPL_PERMACHINE,
        DATAREPL> {  
public:
  
  bool isjulia;

  int n_numa_node;

  int n_thread_per_node;

  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  void (*p_model_allocator) (WRTYPE ** const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator),
    n_numa_node( numa_max_node() + 1),
    n_thread_per_node(getNumberOfCores()/(numa_max_node() + 1))
  {}

  void prepare(){

  }

  double exec(const long * const tasks, int ntasks,
    double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE ** const, int, int),
         void (*p_finalize) (WRTYPE * const, int, int)
    ){

    std::vector<std::future<double>> futures;

    int n_sharding = n_numa_node * n_thread_per_node;
    std::cout << "| Running on " << n_sharding << " Cores..." << std::endl;

    //int n_sharding = 2;

    for(int i_sharding=0;i_sharding<n_sharding;i_sharding++){
      long start = ((long)(ntasks/n_sharding)+1) * i_sharding;
      long end = ((long)(ntasks/n_sharding)+1) * (i_sharding+1);
      end = end >= ntasks ? ntasks : end;
      if(DATAREPL == DW_DATAREPL_FULL){
        futures.push_back(std::async(std::launch::async, _hogwild_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, WRPTR, tasks, 0, ntasks));
      }else{
        futures.push_back(std::async(std::launch::async, _hogwild_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, WRPTR, tasks, start, end));
      }
    }

    double rs = 0.0;

    for(int i=0;i<n_sharding;i++){
      rs += futures[i].get();
    }

    return rs;

  }

};


#endif

