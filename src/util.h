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
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

/**
 * Return the number of logical cores
 * on the current machine. Compatible
 * for both Mac and Linux.
 */
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


/**
 * Mac does not provide NUMA interface, so
 * we reload NUMA-related function for Mac
 * to do nothing.
 */
#ifndef __MACH__
    #include <numa.h>
    #include <numaif.h>
#endif

#ifdef __MACH__
    #define numa_alloc_onnode(X,Y) malloc(X)
    #define numa_max_node() 0
    #define numa_run_on_node(X) 0
    #define numa_set_localalloc() 0
#endif


/**
 * Following is a timer that works for
 * both Mac and Linux.
 */
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#include <sys/time.h>
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

#endif
