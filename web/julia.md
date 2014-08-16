---
layout: page
title: Julia Support
permalink: /julia/
---

DimmWitted provides a [Julia](http://julialang.org/) interface to make it
easy for you to write DimmWitted application in Julia. In this way,
you can take advantage of both the high-level language in Julia and
high-performance backend in DimmWitted at the same time for your
analytics task. In this tutorial, we will go through (1) how to
compile the DimmWitted support for Julia using stochastic gradient descent
(SGD), and (2) how to write a simple application for logistic regression. 

**See Also...** You probably have more questions about writting Julia applications
in DimmWitted that is not covered by this tutorial, following is a list of pages 
that you might also be interested in.

  1. [How to write other access methods in Julia for DimmWitted?](/julia_scd/) We will
  show you an example of writing SCD with a column-to-row access method.
  2. [I am getting a Segmentation Fault! What should I do?](/julia_segfault/) We will show you
  the set of assumptions that we made on your Julia functions, how to use
  a simple tool provided by DimmWitted to sanity check these assumptions, and how
  to use the debugging mode in DimmWitted to diagnose the problem.
  3. [Can I use non-primative data type, e.g., structure, in my data?](/julia_immutable/) Sure, you
  can, but make sure they are immutable.
  4. [Can my gradient function accesses some global variables, e.g., stepsize?](/julia_global/) Yes, but you need to see this tutorial.
  5. [Can I use sparse input matrix?](/julia_sparse/) Yes, you can.
  6. [Miscellaneous](/julia_misc/). We will document some tips we found in our experience 
  that we hope you also found useful.
  7. [Cheat Sheet](/julia_cheetsheet/). 

**Pre-requisites...** To understand this tutorial, we assume that you already went through the
[installation guideline](/installation/) and have all test passed.

##Compile the DimmWitted Interface for Julia

Recall from our [installation guideline](/installation/) that you already checked out
the code of DimmWitted by

    git clone https://github.com/zhangce/dw

and lets still assume DW_HOME to be the name of the folder that contains the code (where
the file `Makefile` sits). Compiling the DimmWitted Interface for Julia
contains two steps: (1) check out dependencies, and (2) compile DimmWitted Interface.

###Dependencies

We first need to checkout three dependencies, including

  1. [Julia (source code)](https://github.com/JuliaLang/julia.git)
  2. [libsupport](https://github.com/JeffBezanson/libsupport)
  3. [libuv](https://github.com/joyent/libuv)

We first go to the lib folder under DW_HOME

    cd DW_HOME/lib
    git clone https://github.com/JuliaLang/julia.git
    git clone https://github.com/JeffBezanson/libsupport
    git clone https://github.com/joyent/libuv

###Compile DimmWitted Interface

Now we can compile the DimmWitted interface:

    cd DW_HOME
    make julia

You should see a new file with the name `libdw_julia.dylib` in the DW_HOME folder.


###Validation

Let's do some simple sanity check to make sure compilation is OK. Open your julia
shell, and first run (Remeber to replace [DW_HOME] with the real path)
    
{% highlight julia linenos%}
push!(LOAD_PATH, "[DW_HOME]/julialib/")
import DimmWitted
DimmWitted.set_libpath("[DW_HOME]/libdw_julia")
{% endhighlight %}

These three lines set up the DimmWitted module that you can use to communicate
with DimmWitted. To validate whether it works or not, type in

    DimmWitted.hi()

You should see

    Hi! -- by DimmWitted

##Writing a simple Julia application

Let's start writing a logistic regression application in Julia.
The code can be found [here](https://github.com/zhangce/dw/blob/master/examples/julia_lr.jl)
but we will walkthrough it together.

The first thing you need to do is to create a Julia program,
let's say with the name `julia_lr.jl`. The first
three lines of the code is the same as the validation
step

{% highlight julia linenos%}
push!(LOAD_PATH, "[DW_HOME]/julialib/")
import DimmWitted
DimmWitted.set_libpath("[DW_HOME]/libdw_julia")
{% endhighlight %}

####Prepare the Data

We will generate a synthetic data set to play with. The following code
creates a synthetic classifcation problem with 100000 examples, each of
which has 10 features and 1 boolean prediction in 0/1.

{% highlight julia linenos%}
nexp = 100000
nfeat = 100
examples = Array(Cdouble, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = 1
	end
	if rand() > 0.8
		examples[row, nfeat+1] = 0
	else
		examples[row, nfeat+1] = 1
	end
end
model = Cdouble[0 for i = 1:nfeat]
{% endhighlight %}

We see that this piece of code creates a two-dimensional
array `examples`, each row of which is an example, and
the first 100 columns are features (all equals to 1 here),
and the last column is the prediction (80% are 1, 20% are 0).
We also created a one-dimensional array `model`, each element
of which corresponds to the weight for each feature.

####Define Loss Function and Gradient Function

After we specify the data, we can write Julia functions
to define how to calculate the loss and gradient. Note that,
in this application, we will use `ROW_ACCESS` in DimmWitted,
which means that DimmWitted will call these functions
for each row with the current state of the model.
Thesefore, these function have the following signature

{% highlight julia linenos%}
(row::Array{Cdouble,1}, model::Array{Cdouble,1}) -> Cdouble
{% endhighlight %}

where `row` and `model` are the row and the current state
of the model, respectively.

>> **Where does the ''Cdouble'' for ''Array{Cdouble,1}'' in the 
function signature comes from?**
>> When you define the data structure `examples` and `models`,
they are of the type Array{Cdouble,2} and Array{Cdouble,1}. DimmWitted
will get their types automatically. You can also use other
primitive types (e.g., Cint) or composite types ([See here](julia_immutable))--
just to make sure you change the signature of the function accordingly.

Let's now define the loss function with this signature.

{% highlight julia linenos%}
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	return (-label * d + log(exp(d) + 1.0))
end
{% endhighlight %}

We can see that this function contains three components:

  1. Line 2-3: We get the label for the given row by picking the
  last element in the `row`, and the total number of features
  by the length of the `model`.
  2. Line 4-7: Calculate the dot product and store it in the variable
  `d`.
  3. Line 8: Calculate the loss for each row and returns it.

Similary, we can write the gradient function

{% highlight julia linenos%}
function grad(row::Array{Cdouble,1}, model::Array{Cdouble,1})
	const label = row[length(row)]
	const nfeat = length(model)
	d = 0.0
	for i = 1:nfeat
		d = d + row[i]*model[i]
	end
	d = exp(-d)
		Z = 0.00001 * (-label + 1.0/(1.0+d))
  	for i = 1:nfeat
  		model[i] = model[i] - row[i] * Z
  	end
	return 1.0
end
{% endhighlight %}

We can see that this `grad` function is similar to `loss`, with the
difference that in Line 10-12, we update the model.

####Run!

We will now create a DimmWitted object to training our logistic
regressor defined by the function `grad` and `loss` on the data
`examples` and `models`. We first create an object with
the specification of the data and how we want to access the data:

{% highlight julia linenos%}
dw = DimmWitted.open(examples, model, 
                DimmWitted.MR_SINGLETHREAD_DEBUG,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW)
{% endhighlight %}

This command creates a DimmWitted object `dw` by using
the `open()` function. Line 1 specifies the data and model,
and line 2-4 specifies how the model will be accessed, here

  - DimmWitted.MR_SINGLETHREAD_DEBUG means that we will
  have one model replica and one thread processing this model.
  (This is slow, but we will show how to make that faster
  in a minute!)
  - DimmWitted.DR_SHARDING means that each thread will
  process a partition of the data instead of the whole data set.
  - DimmWitted.AC_ROW means that we are going to access
  the data (`example`) in a row-wise way.

If this function runs correctly, you should see the following output (Note 
that the address @0x00000001067957a0 might vary for each run--it is the
address of the DimmWitted object created in C++):

    [JULIA-DW] Created DimmWitted Object: Ptr{Void} @0x00000001067957a0

For a complete list of these parameters, see [Cheat Sheet](/julia_cheatsheet/).

After we create this `dw` object, we need to let it know
about the two functions, i.e., `loss` and `grad`, that we
defined. We can do it by

{% highlight julia linenos%}
handle_loss = DimmWitted.register_row(dw, loss)
handle_grad = DimmWitted.register_row(dw, grad)
{% endhighlight %}

Each function call will register the function to DimmWitted
and returns a handle that can be used later. Here, because
both `loss` and `grad` are row-access functions, we
use `register_row` here. (See [Cheat Sheet](/julia_cheatsheet/)
if you want to register other types of functions.) If these
run successfully, you should see in the output:

    [JULIA-DW] Registered Row Function loss Handle=0
    [JULIA-DW] Registered Row Function grad Handle=1

Now lets run a function! Lets first see what is the loss
we can get given the model that we initialized with all zeros:

{% highlight julia linenos%}
rs = DimmWitted.exec(dw, handle_loss)
println("LOSS: ", rs/nexp)
{% endhighlight %}

You should see in the output

    LOSS: 0.6931471805587225

We can then run a gradient step:

{% highlight julia linenos%}
rs = DimmWitted.exec(dw, handle_grad)
{% endhighlight %}

Lets re-calculate the loss, and this time we will
get

    LOSS: 0.5029576555246331

We see that it gets smaller! 

Now we can run ten iterations:

{% highlight julia linenos%}
for iepoch = 1:10
	rs = DimmWitted.exec(dw, handle_loss)
	println("LOSS: ", rs/nexp)
	rs = DimmWitted.exec(dw, handle_grad)
end
{% endhighlight %}

and get the final loss

    LOSS: 0.5029576555246331

####Use All the Cores!

Now we have built a simple logsitic regression model,
but we can make it better because currently it only uses
a single thread. One advantage of DimmWitted is
to run statistial analytics workload efficenlty in
main memory by taking advantage of massive parallelism.

To speed-up our toy example, we only need to do one single
twist

{% highlight julia linenos%}
dw = DimmWitted.open(examples, model, 
                DimmWitted.MR_PERMACHINE,    
                DimmWitted.DR_SHARDING,      
                DimmWitted.AC_ROW)
{% endhighlight %}

If we compare the Line 2, we can see that we are using
a different strategy called `DimmWitted.MR_PERMACHINE`,
which will maintain a single model in main memory and
use all possible threads to update it in a lock-free
way. This approach is also known as Hogwild!

After making this changes, you can then register the
function, and run ten iterations of the gradient step.













