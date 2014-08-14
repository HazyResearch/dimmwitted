
#ifndef _DIMMWITTED_DSTRUCT_H
#define _DIMMWITTED_DSTRUCT_H

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

/**
 * A pair consists of type A and Type B.
 */
template<class A, class B>
class Pair{
public:
    A first;
    B second;
};


#endif