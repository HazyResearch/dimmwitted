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


#ifndef _COMMON_H
#define _COMMON_H

#include <iostream>
#include <thread>
#include <vector>
#include <map>
#include <sstream>
#include <sys/sysctl.h>
#include <assert.h>
#include <iomanip>  
#include <future>
#include <unistd.h>

/**
 * \brief Class for dense vector of type A.
 */
template<class A>
class DenseVector{
public:
    A * p;
    long n;
    DenseVector(A * _p, int _n) :
        p(_p), n(_n){}
};

/**
 * \brief Class for sparse vector of type A.
 *
 * For example, if we want to store a vector
 * \verbatim
   pos1=a pos2=b pos3=c
   \endverbatim
 * Then 
 * \verbatim
    p    = [a b c]
    idxs = [pos1 pos2 pos3]
    n    = 3
   \endverbatim
 */
template<class A>
class SparseVector{
public:
    A * p;
    long * idxs;
    long n;
    SparseVector(A * _p, long * _idxs, int _n) :
        p(_p), idxs(_idxs), n(_n){}
};

template<class A, class B>
class Pair{
public:
    A first;
    B second;
};

enum SparsityType{
  DW_SPARSE,
  DW_DENSE
};

enum ModelReplType{
  DW_STRAWMAN,
  DW_HOGWILD,
  DW_PERCORE,
  DW_PERNODE
};

enum DataReplType{
  DW_FULL,
  DW_SHARDING
  //,DW_IMPORTANCE
};

enum AccessMode{
  DW_ROW,
  DW_COL,
  DW_C2R
};

int getNumberOfCores() {
  #ifdef __MACH__
    int nm[2];
    size_t len = 4;
    uint32_t count;

    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);

    if(count < 1) {
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        if(count < 1) { count = 1; }
    }
    return count;
  #else
    return sysconf(_SC_NPROCESSORS_ONLN);
  #endif
}

#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <sys/time.h>

//clock_gettime is not implemented on OSX
int clock_gettime(int /*clk_id*/, struct timespec* t) {
    struct timeval now;
    int rv = gettimeofday(&now, NULL);
    if (rv) return rv;
    t->tv_sec  = now.tv_sec;
    t->tv_nsec = now.tv_usec * 1000;
    return 0;
}

#define CLOCK_MONOTONIC 0
#endif

#include <time.h>

class Timer {
public:
    
    struct timespec _start;
    struct timespec _end;
    
    Timer(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    
    virtual ~Timer(){}
    
    inline void restart(){
        clock_gettime(CLOCK_MONOTONIC, &_start);
    }
    
    inline float elapsed(){
        clock_gettime(CLOCK_MONOTONIC, &_end);
        return (_end.tv_sec - _start.tv_sec) + (_end.tv_nsec - _start.tv_nsec) / 1000000000.0;
    }
    
    
};
         

#ifndef __MACH__
    #include <numa.h>
    #include <numaif.h>
#endif

#ifdef __MACH__
    #include <math.h>
    #include <stdlib.h>
    #define numa_alloc_onnode(X,Y) malloc(X)
    #define numa_max_node() 0
    #define numa_run_on_node(X) 0
    #define numa_set_localalloc() 0
#endif

#include <math.h>

#define LOG_2   0.693147180559945
#define MINUS_LOG_THRESHOLD   -18.42

inline bool fast_exact_is_equal(double a, double b){
    return (a <= b && b <= a);
}

inline double logadd(double log_a, double log_b){
    if (log_a < log_b){
        double tmp = log_a;
        log_a = log_b;
        log_b = tmp;
    } else if (fast_exact_is_equal(log_a, log_b)) {
        return LOG_2 + log_a;
    }
    double negative_absolute_difference = log_b - log_a;
    if (negative_absolute_difference < MINUS_LOG_THRESHOLD)
        return (log_a);
    return (log_a + log1p(exp(negative_absolute_difference)));
}

#endif
