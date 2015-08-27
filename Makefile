all:
	g++ -Ofast -mavx -ffast-math -o lpblas_avx.out src/main.cpp
	g++ -Ofast -mavx2 -ffast-math -o lpblas_avx2.out src/main.cpp
	g++ -Ofast -mavx -ffast-math -o lpblas_auto_avx.out -D LPBLAS_AUTOVEC src/main.cpp
