/**
Copyright 2014 Hazy Research (http://i.stanford.edu/hazy)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
**/

#include <limits.h>
#include "gtest/gtest.h"
#include "app/glm_sparse_sgd.h"

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERCORE_DATAFULL) {
	double rs;
  rs = test_glm_sparse_sgd<DW_PERCORE, DW_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERNODE_DATAFULL) {
	double rs;
  rs = test_glm_sparse_sgd<DW_PERNODE, DW_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_HOGWILD_DATAFULL) {
	double rs;
  rs = test_glm_sparse_sgd<DW_HOGWILD, DW_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERCORE_SHARDING) {
	double rs;
  rs = test_glm_sparse_sgd<DW_PERCORE, DW_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERNODE_SHARDING) {
	double rs;
  rs = test_glm_sparse_sgd<DW_PERNODE, DW_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_HOGWILD_SHARDING) {
	double rs;
  rs = test_glm_sparse_sgd<DW_HOGWILD, DW_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

