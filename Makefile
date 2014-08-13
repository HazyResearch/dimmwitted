
all:
	clang++ -O3 -std=c++11 -stdlib=libc++ -I./src src/main.cpp

julia:
	clang++ -O3 -std=c++11 -stdlib=libc++ -I./src -dynamiclib src/julia.cpp -o libdw_julia.dylib
	nm -gm libdw_julia.dylib

test_dep:

	clang++ -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/  -c ./lib/gtest-1.7.0/src/gtest_main.cc

	clang++ -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/  -c ./lib/gtest-1.7.0/src/gtest-all.cc
	
test:

	clang++ -O3 -std=c++11 -stdlib=libc++ -I./test -I./src -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/  -c test/glm.cc

	clang++ gtest_main.o  glm.o gtest-all.o -o run_test
