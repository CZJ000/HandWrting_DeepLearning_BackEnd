#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layers/helper.hpp"
#include <arm_neon.h>
namespace caffe {


void matrix_mul_vector_neon( 
     const int M, 
     const int N,
     const int K,   
     const float alpha, 
     const float *A, 
     const float *B, 
     const float beta,    
      float *C) {
  int i, j,e;
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta  = vdupq_n_f32(beta);
  for ( i = 0; i <=M-4; i+=4) 
  {
   for(e=0;e<=N-4;e+=4)
  {
    float32x4_t vc0 = vdupq_n_f32(0.0f);
    float32x4_t vc1 = vdupq_n_f32(0.0f);
    float32x4_t vc2 = vdupq_n_f32(0.0f);
    float32x4_t vc3 = vdupq_n_f32(0.0f);
    for ( j= 0; j < K; j++) 
    {
                  //B K*N bb 
        float32x4_t vb =vld1q_f32(B+j*N+e); // vget(&B[k][4]);   
        //vfmaq_f32 混合   c=a*b+c  
        vc0=vmlaq_f32( vc0,vdupq_n_f32(A[i*K+j]), vb);  
        vc1=vmlaq_f32(vc1,vdupq_n_f32(A[(i+1)*K+j]), vb);
        vc2=vmlaq_f32(vc2,vdupq_n_f32(A[(i+2)*K+j]), vb);
        vc3=vmlaq_f32(vc3,vdupq_n_f32(A[(i+3)*K+j]), vb);
    }  
        // C M*N
      // C = alpha*op( A )*op( B ) + beta*C
         // C M*N
        float32x4_t c0 =vld1q_f32(C+i*N+e);
        float32x4_t c1 = vld1q_f32(C+(i+1)*N+e);
        float32x4_t c2 = vld1q_f32(C+(i+2)*N+e);
        float32x4_t c3 =vld1q_f32(C+(i+3)*N+e);

        float32x4_t c0_b =vmulq_f32(c0, vbeta);
        
        float32x4_t c1_b = vmulq_f32(c1, vbeta);
      
        float32x4_t c2_b = vmulq_f32(c2, vbeta);
      
        float32x4_t c3_b =vmulq_f32(c3, vbeta);

      float32x4_t c0_r= vaddq_f32(vmulq_f32(vc0, valpha),c0_b);
      float32x4_t c1_r= vaddq_f32(vmulq_f32(vc1, valpha),c1_b);
      float32x4_t c2_r= vaddq_f32(vmulq_f32(vc2, valpha),c2_b);
      float32x4_t c3_r= vaddq_f32(vmulq_f32(vc3, valpha),c3_b);



        // vst1q_f32(C+i*N+e,vc0);
        // vst1q_f32(C+(i+1)*N+e,vc1);
        // vst1q_f32(C+(i+2)*N+e,vc2);
        // vst1q_f32(C+(i+3)*N+e,vc3);

         vst1q_f32(C+i*N+e,c0_r);
        vst1q_f32(C+(i+1)*N+e,c1_r);
        vst1q_f32(C+(i+2)*N+e,c2_r);
        vst1q_f32(C+(i+3)*N+e,c3_r);
  }
    if(e<N)
    {
        int p=0,l,q;
        for(l=0;l<4;l++)
        {
          for(q=e;q<N;q++)
          {
            float sum=0;
            for(p=0;p<K;p++)
            {
                sum+=A[(i+l)*K+p]*B[p*N+q];
            }
            C[(i+l)*N+q]= C[(i+l)*N+q]*beta+(sum*alpha);
          }     
        } 
    }
  }

 int p=0;
  for(;i<M;i++)
  {
    for(e=0;e<N;e++)
    {
      float sum=0;
        for(p=0;p<K;p++)
        {
          sum+=A[i*K+p]*B[p*N+e];

        }
        C[i*N+e]= C[i*N+e]*beta+(sum*alpha);
    }
  }
}
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  //  LOG_IF(INFO, Caffe::root_solver())<< "M:"<<M<<"  N:"<<N<<"  K:"<<K;
  //  LOG_IF(INFO, Caffe::root_solver())<<(TransA == CblasNoTrans);
 // _TIMING_START_   
  int i=0,j=0;
    float* mc=(float*)malloc(M * N * sizeof(float));
    for(i=0;i<M;i++)
    {
      for(j=0;j<N;j++)
      {
        mc[i*N+j]=C[i*N+j];
      }
    }

    for(i=0;i<M;i++)
    {
      for(j=0;j<N;j++)
      {
        if(mc[i*N+j]!=C[i*N+j])
        {
          re=0;
          break;
        }
      }
      if(!re)
      {
        LOG_IF(INFO, Caffe::root_solver())<<"false posi"<<": i:"<<i<<" j:"<<j;
        LOG_IF(INFO, Caffe::root_solver())<<"C:"<<C[i*N+j];
         LOG_IF(INFO, Caffe::root_solver())<<"mc:"<<mc[i*N+j];
        break;
      }
    }

   cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
            ldb, beta, C, N);
    matrix_mul_vector_neon(M, N, K,1.0f, A,B,0.0f,mc);
    // int re=1;

    //  for(i=0;i<M;i++)
    // {
    //   for(j=0;j<N;j++)
    //   {
    //     if(mc[i*N+j]!=C[i*N+j])
    //     {
    //       re=0;
    //       break;
    //     }
    //   }
    //   if(!re)
    //   {
    //     LOG_IF(INFO, Caffe::root_solver())<<"false posi"<<": i:"<<i<<" j:"<<j;
    //     LOG_IF(INFO, Caffe::root_solver())<<"C:"<<C[i*N+j];
    //      LOG_IF(INFO, Caffe::root_solver())<<"mc:"<<mc[i*N+j];
    //     break;
    //   }
    // }
   // if(re) LOG_IF(INFO, Caffe::root_solver())<<"true";

      // if(TransA == CblasNoTrans)
      // {
      //      matrix_mul_vector_neon(M, N, K,1.0f, A,B,0.0f,C);
      // }
      // else
      // {
      //      cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      //       ldb, beta, C, N);
      // }
  //_TIMING_STOP_(1)
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
 
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
     
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
        
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe
