#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


extern double _time_sum=0;
/// define start timing macro.
#define _TIMING_START_                            \
  {                                               \
    struct timeval _timing_start_, _timing_stop_; \
    gettimeofday(&_timing_start_, NULL);

/// define stop timing macro.
#define _TIMING_STOP_(ntimes)                                    \
  gettimeofday(&_timing_stop_, NULL);                            \
  double _time_used_ =                                           \
      1000.0 * (_timing_stop_.tv_sec - _timing_start_.tv_sec) +  \
      (_timing_stop_.tv_usec - _timing_start_.tv_usec) / 1000.0; \
      _time_sum+=_time_used_;   \
  }
  //printf("Time Used: %f ms\n", _time_used_ / (ntimes));          
  
  

double get_helper_time_sum();




/// random init data
template <typename T>
void random_init_data(const int num, T *data, const int min = -100,
                      const int max = 100) {
  clock_t ct = clock();
  srand((unsigned int)ct);
  T i_part, f_part;
  for (int i = 0; i < num; ++i) {
    i_part = (T)(rand() % (max - min) + min);
    f_part = (T)rand() / RAND_MAX;
    data[i] = i_part + f_part;
  }
}

/// constant init data
template <typename T>
void constant_init_data(const int num, T *data, const T value = 0) {
  if (value == 0) {
    memset(data, 0, num * sizeof(T));
  } else {
    for (int i = 0; i < num; ++i) {
      data[i] = value;
    }
  }
}

/// check result
template <typename T>
void check_result(const int num, const T *data1, const T *data2) {
  double min_diff = DBL_MAX;
  double max_diff = DBL_MIN;
  double temp_diff = 0.0;
  double ave_diff = 0.0;
  double ave_data = 0.0;

  for (int i = 0; i < num; ++i) {
    // printf("%d: %f %f\n", i, data1[i], data2[i]);
    temp_diff = fabs(data1[i] - data2[i]);
    ave_diff += temp_diff;
    max_diff = (temp_diff > max_diff) ? temp_diff : max_diff;
    min_diff = (temp_diff > min_diff) ? min_diff : temp_diff;
    ave_data += fabs(data1[i]);
  }
  ave_diff = ave_diff / num;
  ave_data = ave_data / num;
  double max_loss =
      ((ave_data > 0 && ave_data < 1e-8) || (ave_data < 0 && ave_data > -1e-8))
          ? 0.0
          : (max_diff / ave_data);
  printf(
      "Check Result:\nMin Diff: %f\nMax Diff: %f\nAve Diff: %f\nMax Prescision "
      "Loss: %f\n",
      min_diff, max_diff, ave_diff, max_loss);
}


#endif  // HELPER_HPP_