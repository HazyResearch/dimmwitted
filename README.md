DimmWitted [![Build Status](https://travis-ci.org/HazyResearch/dimmwitted.svg?branch=master)](https://travis-ci.org/HazyResearch/dimmwitted)
==

DimmWitted is a high-performance execution engine for statistical analytics
and is the core of [DeepDive](deepdive.stanford.edu).To see the type of analytics that is enabled by DimmmWitted, you can find examples in our paper
http://arxiv.org/abs/1403.7550.

# Installation

We first describe how to install DimmWitted.

##Dependencies

DimmWitted is designed to be compiled on Linux
or MacOS. Before downloading the system, you need
one C++ compiler, and we have successfully compiled
DimmWitted using

  - clang++ (Apple LLVM version 6.0, clang-600.0.45.3)
  - g++ 4.8.2 (On Linux)

Note that DimmWitted needs a compiler with C++0x support.
To let DimmWitted know which compiler you are using during compilation, you can use options
like

    CXX=g++-4.8 ...

##Downloading and Compiling DimmWitted

Now you are ready to install DimmWitted. First, you can
check out the most recent version of the code by

    git clone https://github.com/HazyResearch/dimmwitted

By default, this will create a folder called dimmwitted, from now
on, lets use the name `DW_HOME` to refer to this folder.

There are two ways that DimmWitted can be used. We write
a logsitic regression application with DimmWitted that
thats as input the same format as LIBLINEAR and output the
same format. DimmWitted can also be used to write different 
other applications. We describe these two use cases as follows.

###Logistic Regression Application

To compile the logistic regression application, you can compile
DimmWitted as

    cd DW_HOME
    make lr
    
This produces two binary files, `dw-lr-train` and `dw-lr-test`.
To train a model, you can try to run

    ./dw-lr-train -s 0.01 -e 100 -r 0.0001 ./test/a6a  

where ./test/a6a is one example input from LIBLINEAR. `-s`
specifies the stepsize, `-e` specifies the number of epoches
to run, and `-r` specifies the regularization parameter (l2).
This will produces a model file `test/a6a.model`.

To test this model on test data, you can run

    ./dw-lr-test ./test/a6a.t ./test/a6a.model ./test/a6a.output
    
where `./test/a6a.t` is one example test input of LIBLINEAR,
`./test/a6a.model` is the trained model, and `./test/a6a.output`
is the output that is of the same format as LIBLINEAR. Running
this command will give you

    #elements=317325; #examples=21341; #n_features_test=124; #n_features_train=123
    #n_features=124
    Start testing...
    | Running on 8 Cores...
    [DimmWitted FUNC=1] TIME=0.000622 secs THROUGHPUT=  7.86 GB/sec.
    Testing loss=0.33
    | Running on 8 Cores...
    [DimmWitted FUNC=0] TIME=0.000459 secs THROUGHPUT=  10.6 GB/sec.
    Testing acc =0.847
    Dumping the result to ./test/a6a.output...

where you can see the testing loss and testing accuracy. Running LIBLINER
on the same dataset obtains similar accuracy.

You can find more data at http://ntucsu.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
Currently, this example application only supports binary logistic
regression (with label +1 and -1). You can write other applications
by yourself, which will be the topic as you keep reading this document.

###DimmWitted as Library

DimmWitted can be used to write different applications, and
we include an example one in the repositary. To compile
this example, you can type in

    cd DW_HOME
    make

This will generate a binary file with the name `example`, to
validate whether the installation is successful, you can try
to run this binary file by typing in

    ./example

This should output:

    TIME=0.062495001 secs THROUGHPUT=12.208008 GB/sec.
    1.2478489    loss=0.50147917
    TIME=0.063390002 secs THROUGHPUT=12.035643 GB/sec.
    1.2431277    loss=0.50159193
    TIME=0.063591003 secs THROUGHPUT=11.9976 GB/sec.
    1.2588885    loss=0.50123058
    TIME=0.062996998 secs THROUGHPUT=12.110727 GB/sec.
    1.2576657    loss=0.50125708
    TIME=0.062763996 secs THROUGHPUT=12.155686 GB/sec.
    SUM OF MODEL (Should be ~1.3-1.4): 1.2576657

This binary contains one example of using DimmWitted to train
a logistic regression model on a synthetic data set. To check
whether DimmWitted works properly on your machine for this application, 
you should see the last line to be similar to 

    SUM OF MODEL (Should be ~1.3-1.4): 1.2576657

Note that the number 1.26 might vary, but it should not be too far
away from 1.3-1.4.

##Testing

DimmWitted contains a set of unit tests to better make sure
it works properly on your machine. To run our suite of test
cases, you first need to compile googletest by typing in

    make test_dep

And then to run test case, you can type in

    make runtest

Please allow couple minutes for the test to run, and ideally you
should see

    ...
    [----------] Global test environment tear-down
    [==========] 12 tests from 2 test cases ran. (91775 ms total)
    [  PASSED  ] 12 tests.

##What's Next?

Now you have successfully installed DimmWitted. The next step
is to learn how to write an application inside DimmWitted.
You can find examples in the `examples` folder and the walkthrough
[here](https://github.com/HazyResearch/dimmwitted/wiki).

