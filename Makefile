
GCC = gcc
ICC = icc

all:
	$(GCC) -Ofast -fopenmp -mavx -Wa,-q -o lpblas_avx.out src/main.cpp -L /usr/local/lib -lstdc++
	$(GCC) -Ofast -fopenmp -mavx2 -Wa,-q -o lpblas_avx2.out src/main.cpp -L /usr/local/lib -lstdc++
	$(GCC) -Ofast -fopenmp -mavx -Wa,-q -o lpblas_auto_avx.out -D LPBLAS_AUTOVEC src/main.cpp -L /usr/local/lib -lstdc++
	$(GCC) -O2 -mavx -Wa,-q -o lpblas_novec.out -D LPBLAS_AUTOVEC src/main.cpp -L /usr/local/lib -lstdc++

icc:
	$(ICC) -Ofast -openmp -march=corei7-avx -o lpblas_avx.out src/main.cpp
	$(ICC) -Ofast -openmp -march=core-avx2 -o lpblas_avx2.out src/main.cpp
	$(ICC) -Ofast -openmp -march=corei7-avx -o lpblas_auto_avx.out -D LPBLAS_AUTOVEC src/main.cpp
	$(ICC) -O2 -march=corei7-avx -o lpblas_novec.out -D LPBLAS_AUTOVEC src/main.cpp

source:
	$(GCC) -Ofast -fopenmp -mavx -c -S -o lpblas_avx.s src/test.cpp
	$(GCC) -Ofast -fopenmp -mavx2 -c -S -o lpblas_avx2.s src/test.cpp
	$(GCC) -Ofast -fopenmp -mavx -c -S -o lpblas_auto_avx.s -D LPBLAS_AUTOVEC src/test.cpp
	$(GCC) -O2 -mavx -c -S -o lpblas_novec.s -D LPBLAS_AUTOVEC src/main.cpp

iccsource:
	$(ICC) -Ofast -openmp -openmp-report2 -march=corei7-avx -S -o lpblas_avx.s src/test.cpp
	$(ICC) -Ofast -openmp -openmp-report2 -march=core-avx2 -S -o lpblas_avx2.s src/test.cpp
	$(ICC) -Ofast -openmp -openmp-report2 -march=corei7-avx -S -o lpblas_auto_avx.s -D LPBLAS_AUTOVEC src/test.cpp
	$(ICC) -O2 -march=corei7-avx -S -o lpblas_novec.s -D LPBLAS_AUTOVEC src/test.cpp
