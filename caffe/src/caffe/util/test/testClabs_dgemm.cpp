#include <vector>
#include <iostream>
#include "caffe/util/math_functions.hpp"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <arm_neon.h>	
#include "helper.hpp"

using namespace std;

#define TEST_NTIMES 100;



// int main() {
//     const int M=4;
//     const int N=2;
//     const int K=3;
//     const float alpha=1;
//     const float beta=0;
//     const int lda=K;
//     const int ldb=N;
//     const int ldc=N;
//     const float A[M*K]={1,2,3,4,5,6,7,8,9,8,7,6};
//     const float B[K*N]={5,4,3,2,1,0};
//     float C[M*N];
   
// } 

int main(const int argc, const char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s M K N\n", argv[0]);
    printf( "M*K,   K* N  ,  M* N\n");
    return 0;
  }
  const int M = atoi(argv[1]);
  const int K = atoi(argv[1]);
  const int N = atoi(argv[2]);

  // malloc matrix and vector
  float* matrix_A_data = (float*)malloc(M * K * sizeof(float));
  float* matrix_B_data = (float*)malloc(K * N * sizeof(float));

  float* matrix_C_data = (float*)malloc(M * N * sizeof(float));





  //float* vector_data = (float*)malloc(N * sizeof(float));
//   float* result_data_1 = (float*)malloc(M * sizeof(float));
//   float* result_data_2 = (float*)malloc(M * sizeof(float));
//   float* result_data_3 = (float*)malloc(M * sizeof(float));
//   float* result_data_4 = (float*)malloc(M * sizeof(float));
//   float* result_data_5 = (float*)malloc(M * sizeof(float));
  // init matrix and vector data
  random_init_data(M * K, matrix_A_data);
  random_init_data(K * N, matrix_B_data);
  random_init_data(M * N, matrix_C_data);



  //random_init_data(N, vector_data);

//   constant_init_data(M, result_data_1);
//   constant_init_data(M, result_data_2);
//   constant_init_data(M, result_data_3);
//   constant_init_data(M, result_data_4);
//   constant_init_data(M, result_data_5);

  // timing
  int i;
  _TIMING_START_
  for (i = 0; i < 100; ++i) {
    matrix_mul_vector_neon(M, N, matrix_data, vector_data, result_data_1);
  }
  _TIMING_STOP_(100)
 

  
   _TIMING_START_
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
   _TIMING_STOP_(100)
    // for(int i=0;i<M;i++)
    // {
    //    for(int j=0;j<N;j++)
    //    {
    //        cout<<C[i*N+j]<<" ";
    //    }   
    //    cout<<endl;
    // }  

}

void matrix_mul_vector_neon(const CBLAS_TRANSPOSE TransA, 
     const CBLAS_TRANSPOSE TransB, 
     const int M, 
     const int N,
     const int K,   
     const float alpha, 
     const float *A, 
     const float *B, 
     const float beta,    
     float *C) {
  int i, j;
//  for (i = 0; i < m; ++i) {
//    float32x4_t v0 = vdupq_n_f32(0.f);
//    for (j = 0; j < n - 4; j += 4) {
//      float32x4_t v1 = vld1q_f32(matrix + i * n + j);
//      float32x4_t v2 = vld1q_f32(vector + j);
//      v0 = vmlaq_f32(v0, v1, v2);
//    }
//    float temp[4];
//    vst1q_f32(temp, v0);
//    result[i] = temp[0] + temp[1] + temp[2] + temp[3];
//    for (; j < n; ++j) {
//      result[i] += matrix[i * n + j] * vector[j];
//    }

// float32x4_t valpha = vdup(alpha);
// float32x4_t vbeta  = vdup(beta);
float32x4_t vc0 = vdupq_n_f32(0.0f);
float32x4_t vc1 = vdupq_n_f32(0.0f);
float32x4_t vc2 = vdupq_n_f32(0.0f);
float32x4_t vc3 = vdupq_n_f32(0.0f);
 
for ( i = 0; i < K-4; i+=4) {
  
    for ( j= 0; j < K; j++) {
        //B K*N
    float32x4_t vb =vld1q_f32(B+j*N+i); // vget(&B[k][4]);   
    //vfmaq_f32 混合   c=a*b+c  
    vfmaq_f32(vc0, vld1q_f32(A+i*K+j), vb, vc0);  
    vfmaq_f32(vc1, vld1q_f32(A+(i+1)*K+j), vb, vc1);
    vfmaq_f32(vc2, vld1q_f32(A+(i+2)*K+j), vb, vc2);
    vfmaq_f32(vc3, vld1q_f32(A+(i+3)*K+j), vb, vc3);

    }  
     // C M*N
    vst1q_f32(C+j*N+i,vaddq_f32(vmul(vc0, alpha), vmulq_f32(vld1q_f32(C+j*N+i), beta)));
    vst1q_f32(C+(j+1)*N+i,vaddq_f32(vmul(vc1, alpha), vmulq_f32(vld1q_f32(C+(j+1)*N+i), beta)));
    vst1q_f32(C+(j+2)*N+i,vaddq_f32(vmul(vc2, alpha), vmulq_f32(vld1q_f32(C+(j+2)*N+i), beta)));
    vst1q_f32(C+(j+3)*N+i,vaddq_f32(vmul(vc3, alpha), vmulq_f32(vld1q_f32(C+(j+3)*N+i), beta)));
  
}


}







 


