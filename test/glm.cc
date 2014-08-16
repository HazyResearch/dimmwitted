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
#include "logistic_regression_dense_scd.cpp"
#include "logistic_regression_dense_sgd.cpp"
#include "logistic_regression_sparse_sgd.cpp"

TEST(GLMTEST_DENSE_SGD, DENSE_PERCORE_DATAFULL) {
	double rs;
  rs = test_glm_dense_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_DENSE_SGD, DENSE_PERNODE_DATAFULL) {
	double rs;
  rs = test_glm_dense_sgd<DW_MODELREPL_PERNODE, DW_DATAREPL_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_DENSE_SGD, DENSE_HOGWILD_DATAFULL) {
	double rs;
  rs = test_glm_dense_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_DENSE_SGD, DENSE_PERCORE_SHARDING) {
	double rs;
  rs = test_glm_dense_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_DENSE_SGD, DENSE_PERNODE_SHARDING) {
	double rs;
  rs = test_glm_dense_sgd<DW_MODELREPL_PERNODE, DW_DATAREPL_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_DENSE_SGD, DENSE_HOGWILD_SHARDING) {
	double rs;
  rs = test_glm_dense_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

/*
TEST(GLMTEST_SPARSE_SGD, SPARSE_PERCORE_DATAFULL) {
  double rs;
  rs = test_glm_sparse_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERNODE_DATAFULL) {
  double rs;
  rs = test_glm_sparse_sgd<DW_MODELREPL_PERNODE, DW_DATAREPL_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_HOGWILD_DATAFULL) {
  double rs;
  rs = test_glm_sparse_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_FULL>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERCORE_SHARDING) {
  double rs;
  rs = test_glm_sparse_sgd<DW_MODELREPL_PERCORE, DW_DATAREPL_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_PERNODE_SHARDING) {
  double rs;
  rs = test_glm_sparse_sgd<DW_MODELREPL_PERNODE, DW_DATAREPL_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}

TEST(GLMTEST_SPARSE_SGD, SPARSE_HOGWILD_SHARDING) {
  double rs;
  rs = test_glm_sparse_sgd<DW_MODELREPL_PERMACHINE, DW_DATAREPL_SHARDING>();
  EXPECT_GT(rs, 1.0);
  EXPECT_LT(rs, 2.0);
}
*/