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


#ifndef _SCHEDULER_H
#define _SCHEDULER_H

#include "util.h"

#ifdef _JULIA
#include "julia.h"
//#include "julia_internal.h"
#endif


/**
 * \brief This is the actual execution engine of DimmWitted.
 * It is specialized by SCHEDTYPE and DATAREPL, and
 * is implemented in engine/scheduler_hogwild.h,
 * engine/percore.h, and engine/scheduler_pernode.h
 *
 * The semantic of this exeuction is simple, it is
 * a ``map'' over a read-only area with each map funciton
 * has access to a region of read&write area. For
 * concessness of the documentation, we call the read&write
 * area model throughout this documentation.
 *
 * \tparam RDTYPE Type of a read-only area of memory.
 * \tparam WRTYPE Type of a read&write area of memory.
 * \tparam SCHEDTYPE Model replication strategy.
 * \tparam DATAREPL Data replication strategy.
 */
template<class RDTYPE,
         class WRTYPE, 
         ModelReplType SCHEDTYPE,
         DataReplType DATAREPL>
class DWRun {
public:

  bool isjulia;

  int n_numa_node;

  int n_thread_per_node;

  const RDTYPE * const RDPTR; /**<\brief Pointer to the read-only area.*/

  WRTYPE * const WRPTR; /**<\brief Pointer to the read&write area.*/

  void (*p_model_allocator) (WRTYPE * const, const WRTYPE * const);
      /**<\brief Function pointer to the function that allocate model
          replica from a given model.*/

  DWRun(const RDTYPE * const _RDPTR, WRTYPE * const _WRPTR,
        void (*_p_model_allocator) (WRTYPE ** const, const WRTYPE * const)
    ):
    RDPTR(_RDPTR), WRPTR(_WRPTR),
    p_model_allocator(_p_model_allocator)
  {}

  /**
   * \brief Do things like replicate models etc. Can only be called once before
   * DWRun::exec().
   */
  void prepare();

  /**
   * \brief Execute one pass over the read-only data. 
   *
   * \param tasks A list of long numbers, each of which will
   * be used to call p_map.
   * \param ntasks Number of long numbers for the above array.
   * \param p_map For each long number in tasks, this function
   * takes as input that long number, and the pointer
   * to the read-only area and write area, and do something
   * that the engine does not care.
   * \param p_comm Given a set of models, number of models, and
   * an id to one model, this function reads the other models
   * and update the model with the given id. See f_lr_modelavg()
   * in app/glm_dense_sgd.h for an example.
   * \param p_finalize Ignore this guy...
   */
  double exec(const long * const tasks, int ntasks,
         void (*p_map) (long, const RDTYPE * const, WRTYPE * const),
         void (*p_comm) (WRTYPE ** const, int, int),
         void (*p_finalize) (WRTYPE * const, int, int)
    );

};


#endif






