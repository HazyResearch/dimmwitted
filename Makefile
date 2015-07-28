UNAME := $(shell uname)

### LINUX ###
ifeq ($(UNAME), Linux)

ifndef CXX
CXX=g++
endif

CPP_FLAG = -O3 -std=c++11 -I./lib/libunwind-1.1/include -L./lib/numactl-2.0.9 -I./lib/numactl-2.0.9
CPP_INCLUDE = -I./src
CPP_JULIA_LIBRARY = -fPIC -lnuma -shared src/helper/julia_helper.cpp -o libdw_julia.so
CPP_LAST = -lrt -lnuma -l pthread

endif

### MAC ###
ifeq ($(UNAME), Darwin)

ifndef CXX
CXX=clang++
endif

CPP_FLAG = -O3 -std=c++11 -stdlib=libc++ 
CPP_INCLUDE = -I./src
CPP_JULIA_LIBRARY = -dynamiclib src/helper/julia_helper.cpp -o libdw_julia.dylib
CPP_LAST = 

endif

exp:
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) examples/example.cpp -o example $(CPP_LAST)

svm-help.o: application/dw-svm-helper.cpp
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) application/dw-svm-helper.cpp -c -o svm-help.o $(CPP_LAST)

dep:
	cd ./lib/numactl-2.0.9; CXX=$(CXX) make; cd ../..

test_dep:

	$(CXX) -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c ./lib/gtest-1.7.0/src/gtest_main.cc

	$(CXX) -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c ./lib/gtest-1.7.0/src/gtest-all.cc
	
runtest:

	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) -I./test -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -I./examples/ -c test/glm.cc $(CPP_LAST)

	$(CXX) $(CPP_FLAG) gtest_main.o  glm.o gtest-all.o -o run_test $(CPP_LAST)

	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):./lib/numactl-2.0.9 ./run_test

julia:

	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) -I./src -I./lib/julia/src/ -I./lib/libsupport/ -I./lib/libuv/include/ -D _JULIA \
			$(CPP_JULIA_LIBRARY) $(CPP_LAST)

