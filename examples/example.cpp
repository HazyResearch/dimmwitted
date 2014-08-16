
#include "logistic_regression_dense_sgd.cpp"

/**
 * \brief This is one example of running SGD for logistic regression
 * in DimmWitted. You can find more examples in test/glm_dense.cc
 * and test/glm_sparse.cc, and the documented code in 
 * app/glm_dense_sgd.h
 */
int main(int argc, char** argv){
  double rs = test_glm_dense_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING>();
  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
  return 0;
}