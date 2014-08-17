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

#ifndef _SCHEDULER_PERNODE_H
#define _SCHEDULER_PERNODE_H

#include "engine/scheduler.h"

template<class RDTYPE, class WRTYPE>
double _pernode_run_map(double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
    const RDTYPE * const RDPTR, 
    WRTYPE * const WRPTR, 
    const long * const tasks,
    long start, long end, int numa_node){

  numa_run_on_node(numa_node);
  numa_set_localalloc();

  double rs = 0.0;
  for(long i=start;i<end;i++){
    rs += p_map(tasks[i], RDPTR, WRPTR);
  }
  return rs;
}

template<class RDTYPE, class WRTYPE>
void _pernode_comm(void (*p_comm) (WRTYPE ** const, int, int),
  WRTYPE * const myself,
  WRTYPE ** const allothers,
  int nreplicas, 
  int numa_node
  ){

  numa_run_on_node(numa_node);
  numa_set_localalloc();

  p_comm(allothers, nreplicas, numa_node);

}

template<class RDTYPE, class WRTYPE>
void _pernode_comm_julia(void (*p_comm) (WRTYPE * const, int, int),
  WRTYPE * myself,
  int nreplicas, 
  int numa_node
  ){

  numa_run_on_node(numa_node);
  numa_set_localalloc();

  p_comm(myself, nreplicas, numa_node);

}

/**
 * \brief A specialization of DWRun that maintains a single
 * model replica per socket and one thread per core. Each
 * core updates the model replica on the same socket, and model replicas
 * are averaged as soon as possible.
 */
template<class RDTYPE,
         class WRTYPE,
         DataReplType DATAREPL>
class DWRun<RDTYPE, WRTYPE, DW_MODELREPL_PERNODE,
         DATAREPL> {  
public:
  
  bool isjulia;

  int n_numa_node;

  int n_thread_per_node;
   
  const RDTYPE * const RDPTR;

  WRTYPE * const WRPTR;

  WRTYPE ** model_replicas;

  void (*p_model_allocator) (WRTYPE ** const, const WRTYPE * const);

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator),
    n_numa_node( numa_max_node() + 1),
    n_thread_per_node(getNumberOfCores()/(numa_max_node() + 1)),
    isjulia(false)
  {}

#ifdef _JULIA
  jl_array_t* modelptr;
#endif

  void prepare(){
 
    int n_numa_nodes = n_numa_node - 1;
    int n_thread_per_numa = n_thread_per_node;
    n_numa_nodes ++;

    // create one copy for each numa node
    int n_sharding = n_numa_nodes;

    model_replicas = new WRTYPE*[n_sharding+1];
    for(int i=0;i<n_sharding;i++){
      numa_run_on_node(i);
      numa_set_localalloc();
      std::cout << "| Allocating models on NUMA Node " << i << std::endl;
      p_model_allocator(&model_replicas[i], WRPTR);
    }

    model_replicas[n_sharding] = WRPTR;

#ifdef _JULIA
    modelptr = (jl_array_t*) ::operator new(((sizeof(jl_array_t)+jl_array_ndimwords(1)*sizeof(size_t)+15)&-16));

    modelptr->type = jl_typeof(&model_replicas[0]);
    modelptr->data = (void*) &model_replicas[0];
    modelptr->length = n_sharding + 1;
    modelptr->elsize = sizeof(void*);
    modelptr->ptrarray = true;
    modelptr->ndims = 1;
    modelptr->isshared = 1;
    modelptr->isaligned = 0;
    modelptr->how = 0;
    modelptr->nrows = n_sharding + 1;
    modelptr->maxsize = n_sharding + 1;
    modelptr->offset = 0;
#endif

  }

  double exec(const long * const tasks, int ntasks,
    double (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE ** const, int, int),
         void (*p_finalize) (WRTYPE * const, int, int)
    ){

    std::vector<std::future<double>> futures;
    std::vector<std::thread> comm_threads;

    int n_numa_nodes = n_numa_node - 1;
    int n_thread_per_numa = n_thread_per_node;
    n_numa_nodes ++;
    int n_sharding = n_numa_nodes;

    int ct = -1;
    int total = n_numa_nodes * n_thread_per_numa;

    std::cout << "| Running on " << n_sharding << " Nodes with " << n_thread_per_numa << " Cores Each..." << std::endl;

    double rs = 0.0;

    for(int i_sharding=0;i_sharding<n_sharding;i_sharding++){
      for(int i_thread=0;i_thread<n_thread_per_numa;i_thread++){
        ct ++;
        long start = ((long)(ntasks/total)+1) * ct;
        long end = ((long)(ntasks/total)+1) * (ct+1);
        end = end >= ntasks ? ntasks : end;
        
        if(DATAREPL == DW_DATAREPL_FULL){
          futures.push_back(std::async(std::launch::async, _pernode_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, 0, ntasks, i_sharding));
        }else{
          futures.push_back(std::async(std::launch::async, _pernode_run_map<RDTYPE, WRTYPE>, p_map, RDPTR, model_replicas[i_sharding], tasks, start, end, i_sharding));
        }
        //std::cout << "| Start worker " << i_thread << " on NUMA node " << i_sharding << std::endl;
      }
      //std::cout << "| Start communicator on NUMA node " << i_sharding << std::endl;
      if(isjulia == true){
#ifdef _JULIA
        comm_threads.push_back(std::thread(_pernode_comm_julia<RDTYPE, WRTYPE>, p_finalize, modelptr, n_sharding, i_sharding));
#endif
      }else{
        comm_threads.push_back(std::thread(_pernode_comm<RDTYPE, WRTYPE>, p_comm, model_replicas[i_sharding], model_replicas, n_sharding, i_sharding));

      }
    }

    for(int i=0;i<total;i++){
      rs += futures[i].get();
    }

    for(int i=0;i<n_sharding;i++){
      comm_threads[i].join();
    }

    if(isjulia == true){
#ifdef _JULIA
      p_finalize(modelptr, n_sharding, n_sharding);
#endif
    }else{
      p_comm(model_replicas, n_sharding, n_sharding);
    }

    return rs;
  }

};


#endif





