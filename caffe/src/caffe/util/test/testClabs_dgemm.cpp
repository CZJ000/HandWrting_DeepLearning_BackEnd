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
#include  <cblas.h>
using namespace std;

#define TEST_NTIMES 100;

void matrix_mul_vector_neon(const int M, 
     const int N,
     const int K,   
     const float alpha, 
     const float *A, 
     const float *B, 
     const float beta,    
      float *C);

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
  //float* matrix_A_data = (float*)malloc(M * K * sizeof(float));

  float matrix_A_data[]={1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
float matrix_B_data[]={1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
float matrix_C_data[]={1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
 // float* matrix_B_data = (float*)malloc(K * N * sizeof(float));

 // float* matrix_C_data = (float*)malloc(M * N * sizeof(float));
  //float* vector_data = (float*)malloc(N * sizeof(float));
//   float* result_data_1 = (float*)malloc(M * sizeof(float));
//   float* result_data_2 = (float*)malloc(M * sizeof(float));
//   float* result_data_3 = (float*)malloc(M * sizeof(float));
//   float* result_data_4 = (float*)malloc(M * sizeof(float));
//   float* result_data_5 = (float*)malloc(M * sizeof(float));
  // init matrix and vector data
 // random_init_data(M * K, matrix_A_data);
 // random_init_data(K * N, matrix_B_data);
  //random_init_data(M * N, matrix_C_data);

float* c= (float*)malloc(M * N * sizeof(float));
  int i,j;
for (i = 0; i < M * N; ++i) {
   c[i]=matrix_C_data[i];
  }


  //random_init_data(N, vector_data);

//   constant_init_data(M, result_data_1);
//   constant_init_data(M, result_data_2);
//   constant_init_data(M, result_data_3);
//   constant_init_data(M, result_data_4);
//   constant_init_data(M, result_data_5);

  // timing


  _TIMING_START_
  for (i = 0; i < 1; ++i) {
   
    matrix_mul_vector_neon( M, N, K, 1, matrix_A_data,matrix_B_data,0,c);
     
  }
  _TIMING_STOP_(1)
 

 for( i=0;i<M;i++)
    {
       for( j=0;j<N;j++)
       {
           cout<<c[i*N+j]<<" ";
       }   
       cout<<endl;
    }  
 

   _TIMING_START_
   for (i = 0; i < 1; ++i) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, matrix_A_data, K, matrix_B_data, N, 0, matrix_C_data, N);
   }
   _TIMING_STOP_(1)
   
   for( i=0;i<M;i++)
    {
       for( j=0;j<N;j++)
       {
           cout<<matrix_C_data[i*N+j]<<" ";
       }   
       cout<<endl;
    }  

 

}

void matrix_mul_vector_neon( 
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
    
 float32x4_t valpha = vdupq_n_f32(alpha);
 float32x4_t vbeta  = vdupq_n_f32(alpha);
float32x4_t vc0 = vdupq_n_f32(0.0f);
float32x4_t vc1 = vdupq_n_f32(0.0f);
float32x4_t vc2 = vdupq_n_f32(0.0f);
float32x4_t vc3 = vdupq_n_f32(0.0f);
 
for ( i = 0; i <=K-4; i+=4) {
 
    for ( j= 0; j < K; j++) {
              //B K*N bb 
    float32x4_t vb =vld1q_f32(B+j*N+i); // vget(&B[k][4]);   
    //vfmaq_f32 混合   c=a*b+c  
    vc0=vmlaq_f32( vc0,vdupq_n_f32(A[i*K+j]), vb);  
    //  cout<<"vc0 temp:"<<endl;
    //  float temp[4];
   // vst1q_f32(temp, vc0);
    // for(i=0;i<4;i++)
    // {
    //   cout<<temp[i]<<" ";
    // }
    // cout<<endl;
     vc1=vmlaq_f32(vc1,vdupq_n_f32(A[(i+1)*K+j]), vb);
     vc2=vmlaq_f32(vc2,vdupq_n_f32(A[(i+2)*K+j]), vb);
     vc3=vmlaq_f32(vc3,vdupq_n_f32(A[(i+3)*K+j]), vb);

     }  
   
     // C M*N
    float32x4_t c0 =vld1q_f32(C+j*N+i);
    float32x4_t c1 = vld1q_f32(C+(j+1)*N+i);
    float32x4_t c2 = vld1q_f32(C+(j+2)*N+i);
    float32x4_t c3 =vld1q_f32(C+(j+3)*N+i);

   

    vst1q_f32(C+j*N+i,vaddq_f32(vmulq_f32(vc0, valpha), vmulq_f32(c0, vbeta)));
    vst1q_f32(C+(j+1)*N+i,vaddq_f32(vmulq_f32(vc1, valpha), vmulq_f32(c1, vbeta)));
    vst1q_f32(C+(j+2)*N+i,vaddq_f32(vmulq_f32(vc2, valpha), vmulq_f32(c2, vbeta)));
    vst1q_f32(C+(j+3)*N+i,vaddq_f32(vmulq_f32(vc3, valpha), vmulq_f32(c3, vbeta)));
  
}


}







 


