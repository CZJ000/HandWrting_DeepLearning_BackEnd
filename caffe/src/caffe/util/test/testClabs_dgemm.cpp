#include <vector>
#include <iostream>
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
#include<pthread.h>
using namespace std;

#define TEST_NTIMES 100;

int M,N,K,K_full;
float alpha=1.0f,beta=0.0f;

 float *A,*B,*C;
int mc=0,kc=0;
// #define mc 128
// #define kc 128



struct mypara
{
      //   int M;//参数1
      //   int N;//参数2
      //   int K;
      //   float alpha;
      //   // float *A;
      //   // float *B;
      //   float beta;
      //  float *C;
       int i;
       int e;
       int k;
};
pthread_t t[100];  

void matrix_mul_vector_neon_optimize( 
     const int M, 
     const int N,
     const int K,   
     const float alpha, 
      float *A, 
      float *B, 
     const float beta,    
      float *C) ;
void matrix_mul_vector_neon_thread(  int M, 
      int N,
      int K,   
      float alpha, 
      float *A, 
      float *B, 
      float beta,    
      float *C);
void* matrix_mul_vector_neon_rownot4_thread(void *arg);
void*  matrix_mul_vector_neon_colnot4_thread(void *arg);
void* matrix_mul_vector_neon_4by4_thread(void *arg);
void matrix_mul_vector_neon(const int M, 
     const int N,
     const int K,   
     const float alpha, 
     const float *A, 
     const float *B, 
     const float beta,    
      float *C);

void matrix_normal(const int M, 
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
    printf("Usage: %s M   N  K\n", argv[0]);
    printf( "M*K,   K* N  ,  M* N\n");
    return 0;
  }
    M = atoi(argv[1]); 
    N = atoi(argv[2]); 
    K = atoi(argv[3]); 
  K_full=K;
  // malloc matrix and vector
  int blo[8]={512,1024,2048};
  int blo_size[6]={16,32,64,128,256,512};

    float* matrix_A_data = (float*)malloc(M * K * sizeof(float));
    float* matrix_B_data = (float*)malloc(K * N * sizeof(float));
    float* c = (float*)malloc(M * N * sizeof(float));
    random_init_data(M * K, matrix_A_data);
    random_init_data(K * N, matrix_B_data);
    //constant_init_data(M * N, matrix_C_data,0);
    random_init_data(M * N, c);
int i=0,j=0;
    float* c1= (float*)malloc(M * N * sizeof(float));
    for (i = 0; i < M * N; ++i) {
    c1[i]=c[i];
    }

//
  //random_init_data(N, vector_data);

//   constant_init_data(M, result_data_1);
//   constant_init_data(M, result_data_2);
//   constant_init_data(M, result_data_3);
//   constant_init_data(M, result_data_4);
//   constant_init_data(M, result_data_5);

  // timing
// cout<<"before A"<<endl;

//  for( i=0;i<M;i++)
//     {
//        for( j=0;j<K;j++)
//        {
//            cout<<matrix_A_data[i*K+j]<<" ";
//        }   
//        cout<<endl;
//     }  
 
// cout<<"before B"<<endl;

//  for( i=0;i<K;i++)
//     {
//        for( j=0;j<N;j++)
//        {
//            cout<<matrix_B_data[i*N+j]<<" ";
//        }   
//        cout<<endl;
//     }  


     
// cout<<"before C"<<endl;

//  for( i=0;i<K;i++)
//     {
//        for( j=0;j<N;j++)
//        {
//            cout<<c[i*N+j]<<" ";
//        }   
//        cout<<endl;
//     }  



  _TIMING_START_
  for (i = 0; i < 1; ++i) {
   
    matrix_normal( M, N, K,0.0001f, matrix_A_data,matrix_B_data,0.005f,c);
     
  }
  _TIMING_STOP_(1)


  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 0.1f, matrix_A_data, K, matrix_B_data, N,0.005f, c1, N);



int re=1;

     for(i=0;i<M;i++)
    {
      for(j=0;j<N;j++)
      {
        if(abs(c[i*N+j]-c1[i*N+j]) > 1e-6)
        {
          re=0;
          break;
        }
      }
      if(!re)
      {
        cout<<"false posi"<<": i:"<<i<<" j:"<<j<<endl;
        cout<<"c:"<<c[i*N+j]<<endl;
        cout<<"mc:"<<c1[i*N+j]<<endl;
        break;
      }
    }

//  for( i=0;i<M;i++)
//     {
//        for( j=0;j<N;j++)
//        {
//            cout<<c[i*N+j]<<" ";
//        }   
//        cout<<endl;
//     }  
//  cout<<endl;
//   cout<<endl;
//    cout<<endl;


//      for( i=0;i<M;i++)
//     {
//        for( j=0;j<N;j++)
//        {
//            cout<<c1[i*N+j]<<" ";
//        }   
//        cout<<endl;
//     }  


/*
lenet 规模计算  
M       N       K
20      576     25
50      64      25
500     1       800
10      500     1
*/

// vector<int*> size_v(4);
// size_v.push_back({20,576,25});
// size_v.push_back({50,64,25});
// size_v.push_back({500,1,800});
// size_v.push_back({10,500,1});

// int e=0;
// for(e=0;e<4;e++)
// {
//     M=size_v.at(e)[0];
//     N=size_v.at(e)[1];
//     K=size_v.at(e)[2];
//     cout<<"M:   "<<M<<" N:   "<<N<<" K:   "<<K<<endl;
//     float* matrix_A_data = (float*)malloc(M * K * sizeof(float));
//     float* matrix_B_data = (float*)malloc(K * N * sizeof(float));
//     float* matrix_C_data = (float*)malloc(M * N * sizeof(float));
//     random_init_data(M * K, matrix_A_data);
//     random_init_data(K * N, matrix_B_data);
//     constant_init_data(M * N, matrix_C_data,0);
//     constant_init_data(M * N, c,0);
//     cout<<"matrix_mul_vector_neon cost time"<<endl;
//       _TIMING_START_
//        for (i = 0; i < 20;++i) {

//          matrix_mul_vector_neon( M, N, K, 1.0f, matrix_A_data,matrix_B_data,0.0f,matrix_C_data);
        
//        }
//        _TIMING_STOP_(10)


//     cout<<"cblas_sgemm cost time"<<endl;

//             _TIMING_START_
//         for (i = 0; i < 20; ++i) {
//             cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, matrix_A_data, K, matrix_B_data, N, 0, c, N);
//         }
//         _TIMING_STOP_(10)

// }























///不同规模测试


// int e=0;

// for(e=0;e<8;e++)
// {
//   int i=0,j=0;
//   M=blo[e];
//   N=blo[e];
//   K=blo[e];
//   float* matrix_A_data = (float*)malloc(M * K * sizeof(float));
//     //   float matrix_A_data[]={1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
//     // float matrix_B_data[]={1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
//     // float matrix_C_data[]={1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
//   float* matrix_B_data = (float*)malloc(K * N * sizeof(float));
//   float* matrix_C_data = (float*)malloc(M * N * sizeof(float));
//   random_init_data(M * K, matrix_A_data);
//   random_init_data(K * N, matrix_B_data);
//   constant_init_data(M * N, matrix_C_data,0);

//   float* c= (float*)malloc(M * N * sizeof(float));
//   constant_init_data(M * N, c,0);

//   float* c1= (float*)malloc(M * N * sizeof(float));
//   constant_init_data(M * N, c1,0);


//     cout<<"size: "<<blo[e]<<endl;


//     cout<<"matrix_normal cost time"<<endl;

//   _TIMING_START_
//   for (i = 0; i < 1;++i) {
   
//     matrix_normal( M, N, K, 1.0f, matrix_A_data,matrix_B_data,0.0f,c1);
     
//   }
//    _TIMING_STOP_(1)

//     cout<<"matrix_mul_vector_neon cost time"<<endl;

//     //   _TIMING_START_
//     //    for (i = 0; i < 10;++i) {

//     //      matrix_mul_vector_neon( M, N, K, 1.0f, matrix_A_data,matrix_B_data,0.0f,matrix_C_data);
        
//     //    }
//     //    _TIMING_STOP_(10)

//     //分块cache

//     cout<<"matrix_mul_vector_neon_optimize cost time"<<endl;
   
//     int v=0;
//     for(v=0;v<6;v++)
//     {   

//         cout<<"block size "<<blo_size[v]<<endl;
//         mc=blo_size[v];kc=blo_size[v];
//         _TIMING_START_
//         for (i = 0; i < 1; ++i) {

//             // int r, p, pb, ib; 
//             // for (p = 0; p < k; p += kc) {
//             // pb = min(k - p, kc);
//             // for (r = 0; r < m; r += mc) {
//             //   ib = min(m - r, mc);     //每次取256块，小于256时，取小的值
//             //   InnerKernel(ib, n, pb, &A(r, p), lda, &B(p, 0), ldb, &C(r, 0), ldc);
//             // }
//             int r, p, pb, ib; 
//             for (p = 0; p < K; p += kc) {
//             pb = K-p>kc?kc:K-p;//min(k - p, kc);
//             for (r = 0; r < M; r += mc) {
//             ib = M-r>mc?mc:M-r;//min(m - r, mc);     //每次取256块，小于256时，取小的值
//             matrix_mul_vector_neon_optimize( ib, N, pb, 1.0f, matrix_A_data+r*K+p,matrix_B_data+p*N,0.0f,matrix_C_data+r*N);
//             // int q,a;
//             //  for( q=0;q<M;q++)
//             //   {
//             //     for( a=0;a<N;a++)
//             //     {
//             //         cout<<c[q*N+a]<<" ";
//             //     }   
//             //     cout<<endl;
//             //   }  

//             //    cout<<endl;

//                 }

//             }
  
//         }
//          _TIMING_STOP_(1)
//     //     matrix_mul_vector_neon_optimize( M, N, K, 1.0f, matrix_A_data,matrix_B_data,0.0f,c);
      
//     }


///不同规模测试








//     cout<<"cblas_sgemm cost time"<<endl;

//     _TIMING_START_
//    for (i = 0; i < 1; ++i) {
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, matrix_A_data, K, matrix_B_data, N, 0, c, N);
//    }
//    _TIMING_STOP_(1)


// }





//  for( i=0;i<M;i++)
//     {
//        for( j=0;j<N;j++)
//        {
//            cout<<c1[i*N+j]<<" ";
//        }   
//        cout<<endl;
//     }  
 

  // _TIMING_START_
  //  for (i = 0; i < 1; ++i) {
    
  //    matrix_mul_vector_neon_thread( M, N, K, 1.0f, A,B,0.0f,c1);
      
  //  }
  //  _TIMING_STOP_(1)



//  for( i=0;i<M;i++)
//     {
//        for( j=0;j<N;j++)
//        {
//            cout<<c1[i*N+j]<<" ";
//        }   
//        cout<<endl;
//     }  

  // _TIMING_START_
  //  for (i = 0; i < 1; ++i) {
    
  //    matrix_normal( M, N, K, 1.0f, A,B,0.0f,c);
      
  //  }
  //  _TIMING_STOP_(1)






  //  for( i=0;i<M;i++)
  //   {
  //      for( j=0;j<N;j++)
  //      {
  //          cout<<matrix_C_data[i*N+j]<<" ";
  //      }   
  //      cout<<endl;
  //   }  

  //    cout<<endl;
  //     cout<<endl;
//分块cache
//    _TIMING_START_
//    for (i = 0; i < 1; ++i) {

//     // int r, p, pb, ib; 
//     // for (p = 0; p < k; p += kc) {
//     // pb = min(k - p, kc);
//     // for (r = 0; r < m; r += mc) {
//     //   ib = min(m - r, mc);     //每次取256块，小于256时，取小的值
//     //   InnerKernel(ib, n, pb, &A(r, p), lda, &B(p, 0), ldb, &C(r, 0), ldc);
//     // }

//     int r, p, pb, ib; 
//     for (p = 0; p < K; p += kc) {
//       pb = K-p>kc?kc:K-p;//min(k - p, kc);
//     for (r = 0; r < M; r += mc) {
//       ib = M-r>mc?mc:M-r;//min(m - r, mc);     //每次取256块，小于256时，取小的值
//       matrix_mul_vector_neon_optimize( ib, N, pb, 1.0f, matrix_A_data+r*K+p,matrix_B_data+p*N,0.0f,c+r*N);
//       // int q,a;
//       //  for( q=0;q<M;q++)
//       //   {
//       //     for( a=0;a<N;a++)
//       //     {
//       //         cout<<c[q*N+a]<<" ";
//       //     }   
//       //     cout<<endl;
//       //   }  

//       //    cout<<endl;

//     }
//   }
    
// //     matrix_mul_vector_neon_optimize( M, N, K, 1.0f, matrix_A_data,matrix_B_data,0.0f,c);
      
//    }
//  _TIMING_STOP_(100)

   
  //  for( i=0;i<M;i++)
  //   {
  //      for( j=0;j<N;j++)
  //      {
  //          cout<<c[i*N+j]<<" ";
  //      }   
  //      cout<<endl;
  //   }  
  


  
   
  //  for( i=0;i<M;i++)
  //   {
  //      for( j=0;j<N;j++)
  //      {
  //          cout<<c1[i*N+j]<<" ";
  //      }   
  //      cout<<endl;
  //   }  
  

 

}


void matrix_normal(const int M, 
     const int N,
     const int K,   
     const float alpha, 
     const float *A, 
     const float *B, 
     const float beta,    
      float *C)
{
    int i,j,k;

    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
        {   
          float sum=0;
            for(k=0;k<K;k++)
            {
              sum+=A[i*K+k]*B[k*N+j]*alpha;
            }
            C[i*N+j]=sum+C[i*N+j]*beta;
        }
    }


}




void* matrix_mul_vector_neon_4by4_thread(void *arg)
{
 
      struct  mypara *pstru;
       pstru = (struct mypara*) arg;
      //  pstru->;//参数1
      //  pstru->para2;//参数2 
      // int M,N,K;
      // M=pstru->M;
      // N=pstru->N;
      // K=pstru->K;
      // float alpha=pstru->alpha;
      //  float beta=pstru->beta;
      // float* A=pstru->A;
      // float* B=pstru->B;
      // float* C=pstru->C;


    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta  = vdupq_n_f32(beta);

    int i, j,e;

    i=pstru->i;
    e=pstru->e;

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


        vst1q_f32(C+i*N+e,c0_r);
      // cout<<"c1 temp:"<<endl;
      //  float temp[4];
      //vst1q_f32(temp, vaddq_f32(vmulq_f32(vc1, valpha), vmulq_f32(c1, vbeta)));
        vst1q_f32(C+(i+1)*N+e,c1_r);
      // for(i=0;i<4;i++)
      //   {
      //     cout<<C[i]<<" ";
      //   }
      //   cout<<endl;
        vst1q_f32(C+(i+2)*N+e,c2_r);
        vst1q_f32(C+(i+3)*N+e,c3_r);



}


void*  matrix_mul_vector_neon_colnot4_thread(void *arg)
{

     struct  mypara   *pstru;
       pstru = ( struct mypara *) arg;
      //  pstru->;//参数1
      //  pstru->para2;//参数2 
      // int M,N,K;
      // M=pstru->M;
      // N=pstru->N;
      // K=pstru->K;
      // float alpha=pstru->alpha;
      //  float beta=pstru->beta;
      // float* A=pstru->A;
      // float* B=pstru->B;
      // float* C=pstru->C;


    int i,e;

    i=pstru->i;
    
    e=pstru->e;

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
            C[(i+l)*N+q]=sum;
          }     
        } 

 
}



void* matrix_mul_vector_neon_rownot4_thread(void *arg)
{

       struct  mypara  *pstru;
       pstru = (struct mypara*) arg;
      //  pstru->;//参数1
      //  pstru->para2;//参数2 
    //   int M,N,K;
    //   M=pstru->M;
    //   N=pstru->N;
    //   K=pstru->K;
    // float alpha=pstru->alpha;
    // float beta=pstru->beta;
    //  float* A=pstru->A;
    //  float* B=pstru->B;
    //   float* C=pstru->C;


    int i,e;

    i=pstru->i;
  
    e=pstru->e;

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
        C[i*N+e]=sum;
    }
  }

 
}




void matrix_mul_vector_neon_thread(  int M, 
      int N,
      int K,   
      float alpha, 
      float *A, 
      float *B, 
      float beta,    
      float *C)
{
    int a=0;
    int i, j,e;
        for ( i = 0; i <=M-4; i+=4) 
    {
        for(e=0;e<=N-4;e+=4)
        {         
            struct mypara pstru;        
            // pstru.M=M;      
            // pstru.N=N;    
            // pstru.K=K;
            
            // pstru.alpha=alpha;
            
            // pstru.A=A;
            
            // pstru.B=B;
            
            // pstru.C=C;
            
            pstru.i=i;       
            pstru.e=e;       
            int re =pthread_create(&t[a], NULL, matrix_mul_vector_neon_4by4_thread,&(pstru));
            
            pthread_join(t[a],NULL);
            a++;
            
        }
        if(e<N)
        {
            struct mypara pstru;
            // pstru.M=M;
            // pstru.N=N;
            // pstru.K=K;
            // pstru.alpha=alpha;
            // pstru.A=A;
            // pstru.B=B;
            // pstru.C=C;
            pstru.i=i;
            pstru.e=e;
            pthread_create(&t[a], NULL, matrix_mul_vector_neon_colnot4_thread,&(pstru));
            pthread_join(t[a],NULL);
            a++;
        }
}

          struct mypara pstru;
          // pstru.M=M;
          // pstru.N=N;
          // pstru.K=K;
          // pstru.alpha=alpha;
          // pstru.A=A;
          // pstru.B=B;
          // pstru.C=C;
          pstru.i=i;
          pstru.e=e;
          pthread_create(&t[a], NULL, matrix_mul_vector_neon_rownot4_thread,&(pstru));
          pthread_join(t[a],NULL);
          a++;
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
  int i, j,e;
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

      //    int z,x;
      //  cout<<endl;
      //   cout<<endl;
      // for(z=0;z<M;z++)
      //   {
           
      //     for(x=0;x<N;x++)
      //       cout<<C[z*N+x]<<" ";
      //     cout<<endl;
      //   }
      //   cout<<endl;
       
      
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
            C[(i+l)*N+q]=C[(i+l)*N+q]*beta+(sum*alpha);;
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
        C[i*N+e]=C[i*N+e]*beta+(sum*alpha);
    }
  }
}



void matrix_mul_vector_neon_optimize( 
     const int M, 
     const int N,
     const int K,   
     const float alpha, 
      float *A, 
      float *B, 
     const float beta,    
      float *C) {
  int i, j,e;
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
for ( i = 0; i <=M-4; i+=4) 
  {
   for(e=0;e<=N-4;e+=4)
  {
    float32x4_t vc0 = vdupq_n_f32(0.0f);
    float32x4_t vc1 = vdupq_n_f32(0.0f);
    float32x4_t vc2 = vdupq_n_f32(0.0f);
    float32x4_t vc3 = vdupq_n_f32(0.0f);

    // float 
    // /* Point to the current elements in the four rows of A */
    // *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

    // a_0p_pntr = A+i*K;
    // a_1p_pntr = A+(i+1)*K;
    // a_2p_pntr = A+(i+2)*K;
    // a_3p_pntr = A+(i+3)*K;

    // register float
    // a_0p_reg,
    // a_1p_reg,   
    // a_2p_reg,
    // a_3p_reg;

    for ( j= 0; j < K; j++) 
    {


      // a_0p_reg = *a_0p_pntr++;
      // a_1p_reg = *a_1p_pntr++;
      // a_2p_reg = *a_2p_pntr++;
      // a_3p_reg = *a_3p_pntr++;
                  //B K*N bb 
        float32x4_t vb =vld1q_f32(B+j*N+e); // vget(&B[k][4]);   
        //vfmaq_f32 混合   c=a*b+c  
      //   vc0=vmlaq_n_f32( vc0, vb,a_0p_reg);  
      //   //  cout<<"vc0 temp:"<<endl;
      //   //  float temp[4];
      // // vst1q_f32(temp, vc0);
      //   // for(i=0;i<4;i++)
      //   // {
      //   //   cout<<temp[i]<<" ";
      //   // }
      //   // cout<<endl;
      //   vc1=vmlaq_n_f32(vc1, vb,a_1p_reg);
      //   vc2=vmlaq_n_f32(vc2, vb,a_2p_reg);
      //   vc3=vmlaq_n_f32(vc3, vb,a_3p_reg);


      vc0=vmlaq_f32( vc0,vdupq_n_f32(A[i*K_full+j]), vb);  
        //  cout<<"vc0 temp:"<<endl;
        //  float temp[4];
      // vst1q_f32(temp, vc0);
        // for(i=0;i<4;i++)
        // {
        //   cout<<temp[i]<<" ";
        // }
        // cout<<endl;
        vc1=vmlaq_f32(vc1,vdupq_n_f32(A[(i+1)*K_full+j]), vb);
        vc2=vmlaq_f32(vc2,vdupq_n_f32(A[(i+2)*K_full+j]), vb);
        vc3=vmlaq_f32(vc3,vdupq_n_f32(A[(i+3)*K_full+j]), vb);

    }  
        float32x4_t c0 =vld1q_f32(C+i*N+e);
        float32x4_t c1 = vld1q_f32(C+(i+1)*N+e);
        float32x4_t c2 = vld1q_f32(C+(i+2)*N+e);
        float32x4_t c3 =vld1q_f32(C+(i+3)*N+e);

      //   float32x4_t c0_b =vmulq_f32(c0, vbeta);
        
      //   float32x4_t c1_b = vmulq_f32(c1, vbeta);
      
      //   float32x4_t c2_b = vmulq_f32(c2, vbeta);
      
      //   float32x4_t c3_b =vmulq_f32(c3, vbeta);

      float32x4_t c0_r= vaddq_f32(vc0,c0);
      float32x4_t c1_r= vaddq_f32(vc1,c1);
      float32x4_t c2_r= vaddq_f32(vc2,c2);
      float32x4_t c3_r= vaddq_f32(vc3,c3);


        vst1q_f32(C+i*N+e,c0_r);
        vst1q_f32(C+(i+1)*N+e,c1_r);
        vst1q_f32(C+(i+2)*N+e,c2_r);
        vst1q_f32(C+(i+3)*N+e,c3_r);
      // cout<<"c1 temp:"<<endl;
      //  float temp[4];
      //vst1q_f32(temp, vaddq_f32(vmulq_f32(vc1, valpha), vmulq_f32(c1, vbeta)));
      // int z,x;
      //  cout<<endl;
      //   cout<<endl;
      // for(z=0;z<M;z++)
      //   {
           
      //     for(x=0;x<N;x++)
      //       cout<<C[z*N+x]<<" ";
      //     cout<<endl;
      //   }
      //   cout<<endl;
       
        
      
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
                sum+=A[(i+l)*K_full+p]*B[p*N+q];
            }
            C[(i+l)*N+q]+=sum;
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
          sum+=A[i*K_full+p]*B[p*N+e];

        }
        C[i*N+e]+=sum;
    }
  }
}













 


