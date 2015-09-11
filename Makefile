all:
	g++ -Ofast -mavx -o lpblas_avx.out src/main.cpp
	g++ -Ofast -mavx2 -o lpblas_avx2.out src/main.cpp
	g++ -Ofast -mavx -o lpblas_auto_avx.out -D LPBLAS_AUTOVEC src/main.cpp
	g++ -O2 -mavx -o lpblas_novec.out -D LPBLAS_AUTOVEC src/main.cpp

icc:
	icc -Ofast -march=corei7-avx -o lpblas_avx.out src/main.cpp
	icc -Ofast -march=core-avx2 -o lpblas_avx2.out src/main.cpp
	icc -Ofast -march=corei7-avx -o lpblas_auto_avx.out -D LPBLAS_AUTOVEC src/main.cpp
	icc -O2 -march=corei7-avx -o lpblas_novec.out -D LPBLAS_AUTOVEC src/main.cpp

source:
	g++ -Ofast -mavx -S -o lpblas_avx.s src/test.cpp
	g++ -Ofast -mavx2 -S -o lpblas_avx2.s src/test.cpp
	g++ -Ofast -mavx -S -o lpblas_auto_avx.s -D LPBLAS_AUTOVEC src/test.cpp
	g++ -O2 -mavx -S -o lpblas_novec.s -D LPBLAS_AUTOVEC src/main.cpp

iccsource:
	icc -Ofast -march=corei7-avx -S -o lpblas_avx.s src/test.cpp
	icc -Ofast -march=core-avx2 -S -o lpblas_avx2.s src/test.cpp
	icc -Ofast -march=corei7-avx -S -o lpblas_auto_avx.s -D LPBLAS_AUTOVEC src/test.cpp
	icc -O2 -march=corei7-avx -S -o lpblas_novec.s -D LPBLAS_AUTOVEC src/test.cpp
