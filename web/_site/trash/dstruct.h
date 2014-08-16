#ifndef _DSTRUCT_H
#define _DSTRUCT_H

#include "common.h"

template<class A, SparsityType SPARSITY>
class Array{
};

template<class A>
class Array<A, DW_DENSE>{
public:
	const A* const p;
	int n;
	Array(const A * const _p, int _n):
		p(_p), n(_n){}
};

#endif