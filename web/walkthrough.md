---
layout: page
title: C++ Support
permalink: /cpp/
---

This tutorial will walkthrough step-by-step
of how to creating a DimmWitted application
in C++. This tutorial assumes that you
already went through the 
[installation guideline](/installation/).
Apart from C++, you can also write your
application Julia, and you can find the
corresponding tutorial [here](/walkthrough#/).

The application that we are going to
build is to train a logistic regression 
model on dense data set. You can find
the code [here](https://github.com/zhangce/dw/blob/master/src/app/glm_dense_sgd.h), but we are going to
walk through it step-by-step.

##A Primer for Logistic Regression

Before we start writing C++ code, lets go over
some basic concepts of logistic regression
to make sure we are on the same page.

In this example, we will encode a logistic regression
over Boolean random variables, that can take
values from 0 and 1. We will have a set of
random variables $$\{y_1,...,y_n\}$$. For
each random variable $$\{y_i\}$$, we have a set of 
features denoted as $$\mathcal{X_i}=\{x_{i1},...,x_{im}\}$$.
We also have a set of real-value weights $$\Omega=\{w_i,...,w_m\}$$.
Given these settings, we can define the
probability distribution of each $$y_i$$ 
equal to a certain value (0 or 1) as

$$\Pr\left[ y_i = y ; \mathcal{X_i}, \Omega \right] = \frac{\exp\left\{y\sum_j w_{j}x_{wj}\right\}}{1 + \exp\left\{\sum_j w_{j}x_{wj}\right\}}$$

Assume that we already observe the value that each $$y_i$$
should take, denoted as $$\hat{y_i}$$, training a logistic
regression model is to find the set of weight
that minimizes the negative log likelihood, which is defined as

$$\mathcal{L}\left(\Omega\right) = \sum_i - \log \Pr\left[ y_i = \hat{y_i} ; \mathcal{X_i}, \Omega \right]$$

To solve this mathematical optimization problem, we
will implement an approach called _Stochastic Gradient
Descent (SGD)_. It contains multiple steps as follows

  1. Pick an $$i$$;
  2. Calculate the gradient $$\nabla^{(i)}_j = \frac{\partial}{\partial w_j} \left(- \log \Pr\left[y_i=\hat{y_i};\mathcal{X_i}, \Omega\right] \right)$$;
  3. Update $$w_j$$ to be $$w_j + \lambda \nabla^{(i)}_j$$, where $$\lambda$$ is a constant step size.
  4. Repeat 1.

We will then show how to write this simple SGD algorithm inside DimmWitted.

##Implementing Logistic Regression in DimmWitted

Before we start writing any code, we need to include
the header file that contains DimmWitted-related functions
by

{% highlight C++ linenos%}
#include "dimmwitted.h"
{% endhighlight %}

####Define the Workspace

In DimmWitted, the workspace contains a set of objects
that will be changed during execution, in our case, this
workspace contains $$\Omega$$, which is a set of weights
that we are going to update. In DimmWitted, we
need to define a class for the workspace

{% highlight C++ linenos%}
class GLMModelExample{
public:
  double * const p;
  int n;
  
  GLMModelExample(int _n):
    n(_n), p(new double[_n]){}

  GLMModelExample( const GLMModelExample& other ) :
     n(other.n), p(new double[other.n]){
    for(int i=0;i<n;i++){
      p[i] = other.p[i];
    }
  }
};
{% endhighlight %}

We can see that this class contains four components:

  1. Line 3-4: These two lines define a double-typed pointer,
  and the number of elements in this pointer. One can think
  about each double number here corresponds to one $$w_j \in \Omega$$.
  2. Line 6-7: These two lines define a constructor for the 
  workspace. In this simple example, we take as input the
  number of elements, and allocate the memory space.
  3. Line 9-14: These six lines define a copy constructor.
  This function is highly recommended to implement because it
  will be used when DimmWitted decides to replicate your
  workspace for better performance.


####Prepare the Data and Create a DimmWitted Object








####Define the Gradient Function

We then define the gradient function. 

{% highlight C++ linenos%}
double f_lr_grad(const DenseVector<double>* const ex,
				 GLMModelExample* const p_model){

  double * model = p_model->p;
  double label = ex->p[ex->n-1];

  double dot = 0.0;
  for(int i=0;i<ex->n-1;i++){
    dot += ex->p[i] * model[i];
  }

  const double d = exp(-dot);
  const double Z = 0.00001 * (-label + 1.0/(1.0+d));

  for(int i=0;i<ex->n-1;i++){
    model[i] -= ex->p[i] * Z;
  }

  return 1.0;
}
{% endhighlight %}

####Create a DimmWitted Object and Execute

{% highlight C++ linenos%}
DenseDimmWitted<double, GLMModelExample, DW_DEBUG, DW_SHARDING, DW_ROW> 
    dw(examples, nexp, nfeat+1, &model);
unsigned int f_handle_grad = dw.register_row(f_lr_grad);
dw.exec(f_handle_grad);
{% endhighlight %}

##Extensions

#### More efficient way of calculating the gradient

#### Use Stocastic Coordinate Descent instead of Stocastic Gradient Descent














