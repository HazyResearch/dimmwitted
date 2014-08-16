

extern "C" {

  void strawman_sum(double *, double *, long* , int);
  void hogwild_sum(double *, double *, long* , int);
  void percore_sum(double *, double *, long* , int);

  void glm_sgd(double *, double *, double *, long, long, void(*)(const double * const, double * const, double, int));

}
