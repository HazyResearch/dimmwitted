
all:
	clang++ -O3 -std=c++11 -stdlib=libc++ -I./src src/main.cpp

julia:
	clang++ -O3 -std=c++11 -stdlib=libc++ -I./src -dynamiclib src/julia.cpp -o libdw_julia.dylib
	nm -gm libdw_julia.dylib
