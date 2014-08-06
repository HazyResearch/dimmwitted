#ifndef _SCHEDULER_PERNODE_H
#define _SCHEDULER_PERNODE_H

#include "engine/scheduler.h"


template<class RDTYPE, class WRTYPE>
void _pernode_run_map(void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
    const RDTYPE * const RDPTR, 
    WRTYPE * const WRPTR, 
    const long * const tasks,
    long start, long end, int numa_node){

  //numa_run_on_node(numa_node);
  //numa_set_localalloc();

  for(long i=start;i<end;i++){
    p_map(tasks[i], RDPTR, WRPTR);
  }
}

template<class RDTYPE,
         class WRTYPE>
class DWRun<RDTYPE, WRTYPE, SCHED_PERNODE> {  
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
 
    int n_numa_nodes = numa_max_node();
    int n_thread_per_numa = getNumberOfCores()/(n_numa_nodes+1);
    n_numa_nodes ++;

    // create one copy for each numa node
    int n_sharding = n_numa_nodes;

    model_replicas = new WRTYPE*[n_sharding];
    for(int i=0;i<n_sharding;i++){
      numa_run_on_node(i);
      numa_set_localalloc();
      std::cout << "| Allocating models on NUMA Node " << i << std::endl;
      p_model_allocator(&model_replicas[i], WRPTR);
    }

  }

  void exec(const long * const tasks, int ntasks,
    void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE * const, const WRTYPE ** const, int),
         void (*p_finalize) (WRTYPE * const, WRTYPE ** const, int)
    ){

    std::vector<std::thread> threads;

    int n_numa_nodes = numa_max_node();
    int n_thread_per_numa = getNumberOfCores()/(n_numa_nodes+1);
    n_numa_nodes ++;
    int n_sharding = n_numa_nodes;

    int ct = -1;
    int total = n_numa_nodes * n_thread_per_numa;

    for(int i_sharding=0;i_sharding<n_sharding;i_sharding++){
      for(int i_thread=0;i_thread<n_thread_per_numa;i_thread++){
        ct ++;
        long start = ((long)(ntasks/total)+1) * ct;
        long end = ((long)(ntasks/total)+1) * (ct+1);
        end = end >= ntasks ? ntasks : end;
        threads.push_back(std::thread(_pernode_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, start, end, i_sharding));
        std::cout << "| Start worker " << i_thread << " on NUMA node " << i_sharding << std::endl;
      }
    }

    for(int i=0;i<total;i++){
      threads[i].join();
    }

    p_finalize(WRPTR, model_replicas, n_sharding);

  }

};


#endif





