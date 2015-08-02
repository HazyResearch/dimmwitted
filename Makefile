all:
	g++ -Ofast -mavx -o lpblas_avx.out src/main.cpp
	g++ -Ofast -mavx2 -o lpblas_avx2.out src/main.cpp
