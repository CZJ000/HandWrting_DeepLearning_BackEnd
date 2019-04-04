#include "caffe/layers/helper.cpp"

namespace caffe {
double _time_sum=0;

double get_helper_time_sum()
{
    return _time_sum;
}


}