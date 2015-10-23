// Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "app/glm_dense_sgd.h"
#include "app/glm_sparse_sgd.h"
#include "app/glm_dense_scd.h"

/**
 * \brief This is one example of running SGD for logistic regression
 * in DimmWitted. You can find more examples in test/glm_dense.cc
 * and test/glm_sparse.cc, and the documented code in 
 * app/glm_dense_sgd.h
 */
int main(int argc, char** argv){
  double rs = test_glm_dense_sgd<DW_MODELREPL_SINGLETHREAD_DEBUG, DW_SHARDING>();
  std::cout << "SUM OF MODEL (Should be ~1.3-1.4): " << rs << std::endl;
  return 0;
}

