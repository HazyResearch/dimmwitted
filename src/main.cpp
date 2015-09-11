
#include "types.h"
#include "cv.h"
#include "timer.h"
#include "dot.h"
#include <stdlib.h>
#include <iostream>

using namespace lpblas;

void report_range(){
	std::cout << MIN_VALUE<LPBLAS_i8>() << " ~ " << MAX_VALUE<LPBLAS_i8>() << std::endl;
	std::cout << MIN_VALUE<LPBLAS_i16>() << " ~ " << MAX_VALUE<LPBLAS_i16>() << std::endl;
	std::cout << MIN_VALUE<LPBLAS_f32>() << " ~ " << MAX_VALUE<LPBLAS_f32>() << std::endl;
}

template<typename LPBLAS_TYPE>
void test_dot(){
	const int N = 100000000;
	LPBLAS_TYPE  * dst = new LPBLAS_TYPE[N];
	LPBLAS_TYPE  * dst3= new LPBLAS_TYPE[N];
	LPBLAS_f32 * src = new LPBLAS_f32[N];
	LPBLAS_f32 * dst2= new LPBLAS_f32[N];
	double dot_ans = 0.0;
	for(int i=0;i<N;i++){
		src[i] = 2*drand48()-1;
		dot_ans += src[i];
		dst2[i] = 0.0;
		dst[i] = 0;
		dst3[i] = MAX_VALUE<LPBLAS_TYPE>();	// this is 1 in our encoding
	}
	Timer t;
	double te;
	double size;

	t.restart();
	convert_dense_ceil(src, dst, N);
	te = t.elapsed();
	size = 1.0*N*(sizeof(LPBLAS_TYPE)+sizeof(LPBLAS_f32))/1024/1024;
	//std::cout << "    TIME = " << te << std::endl;
	//std::cout << "    SIZE = " << size << " MB" << std::endl;
	//std::cout << "    BAND = " << size/te << " MB/s" << std::endl;
	std::cout << "| Convert (f32->" << sizeof(LPBLAS_TYPE)*8 << ") = " << size/te << " MB/s" << "   t = " << te << " seconds." << std::endl;

	t.restart();
	convert_dense_ceil(dst, dst2, N);
	te = t.elapsed();
	size = 1.0*N*(sizeof(LPBLAS_TYPE)+sizeof(LPBLAS_f32))/1024/1024;
	//std::cout << "    TIME = " << te << std::endl;
	//std::cout << "    SIZE = " << size << " MB" << std::endl;
	//std::cout << "    BAND = " << size/te << " MB/s" << std::endl;
	std::cout << "| Convert (" << sizeof(LPBLAS_TYPE)*8 << "->f32) = " << size/te << " MB/s" << "   t = " << te << " seconds." << std::endl;

	for(int i=0;i<5;i++){
		std::cout << "|    Approximate: "  << src[i] << " -> " << dst2[i] << std::endl;
	}

	t.restart();
	float dot = dot_dense(dst, dst3, N);
	te = t.elapsed();
	size = 1.0*N*(sizeof(LPBLAS_TYPE)+sizeof(LPBLAS_TYPE))/1024/1024;
	//std::cout << "    TIME = " << te << std::endl;
	//std::cout << "    SIZE = " << size << " MB" << std::endl;
	//std::cout << "    BAND = " << size/te << " MB/s" << std::endl;
	std::cout << "| Dot (" << sizeof(LPBLAS_TYPE)*8 << ") = " << size/te << " MB/s" << "   t = " << te << " seconds." << std::endl;
	std::cout << "|    Approximate: " << dot << " -> " << dot_ans << std::endl;
	std::cout << std::endl;

}

int main(int argc, char ** argv){
	//report_range();
	test_dot<LPBLAS_i8>();
	test_dot<LPBLAS_i16>();
	test_dot<LPBLAS_f32>();
	return 0;
}
