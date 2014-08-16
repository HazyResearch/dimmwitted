---
layout: page
title: Installation
permalink: /installation/
---

This page contains instructions of how to install 
DimmWitted. After finishing this step, you can
read [this page](/walkthrough/) to see how to write
an application inside DimmWitted.

##Dependencies

DimmWitted is designed to be compiled on Linux
or MacOS. Before downloading the system, you need
one C++ compiler, and we have successfully compiled
DimmWitted using

  - clang++ (Apple LLVM version 6.0, clang-600.0.45.3)
  - g++ 4.8.2 (On Linux)

##Downloading and Compiling DimmWitted

Now you are ready to install DimmWitted. First, you can
check out the most recent version of the code by

    git clone https://github.com/zhangce/dw

By default, this will create a folder called dw, from now
on, lets use the name `DW_HOME` to refer to this folder.

DimmWitted can be used to write different applications, and
we include an example one in the repositary. To compoile
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
is to learn how to write an application inside DimmWitted
in [this page](/walkthrough/).


