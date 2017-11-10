// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

      register const unsigned int TW = 32;
      register const unsigned int Tw[8] = {0,TW/8, TW/4, 3*TW/8, TW/2, 5*TW/8, 3*TW/4, 7*TW/8};
      const unsigned int edge_limit = N/TW;
      const unsigned int ty = threadIdx.y, tx = threadIdx.x;
      const unsigned int by = blockIdx.y, bx = blockIdx.x;
      const unsigned int I = by*TW + ty, J = bx*TW + tx;

      __shared__ _DOUBLE_ As[TW][TW], Bs[TW][TW];

     register _DOUBLE_ Cs[8] = {0};

    //  #pragma unroll
    //  for(int i = 0; i < 4; i++)
    //  {
    //     As[ty+Tw[i]][tx] = 0;
    //  }
    //  #pragma unroll
    //  for(int i = 0; i < 4; i++)
    //  {
    //     Bs[ty+Tw[0]][tx] = 0;
    //  }
     As[ty][tx] = 0;
     As[ty+Tw[1]][tx] = 0;
     As[ty+Tw[2]][tx] = 0;
     As[ty+Tw[3]][tx] = 0;
     As[ty+Tw[4]][tx] = 0;
     As[ty+Tw[5]][tx] = 0;
     As[ty+Tw[6]][tx] = 0;
     As[ty+Tw[7]][tx] = 0;

     Bs[ty][tx] = 0;
     Bs[ty+Tw[1]][tx] = 0;
     Bs[ty+Tw[2]][tx] = 0;
     Bs[ty+Tw[3]][tx] = 0;
     Bs[ty+Tw[4]][tx] = 0;
     Bs[ty+Tw[5]][tx] = 0;
     Bs[ty+Tw[6]][tx] = 0;
     Bs[ty+Tw[7]][tx] = 0;

     if(N%TW !=0)
     {
       for(int Ai =I; (Ai < N) && (Ai < I + TW); Ai +=Tw[1])
       {
         if(edge_limit*TW+tx < N)
          As[ty+Ai-I][tx] = A[(Ai)*N+(edge_limit*TW+tx)];
       }
       int b_start = edge_limit*TW+ty;
       for(int Bi = b_start;(Bi<N) && (Bi < b_start+TW) ; Bi+=Tw[1])
       {
        if(J < N)
          Bs[Bi-edge_limit*TW][tx] = B[Bi*N+J];

       }
       __syncthreads();

       for(int k = 0; k < TW; k++)
       {
         Cs[0] += As[ty][k] * Bs[k][tx];
         Cs[1] += As[ty+Tw[1]][k] * Bs[k][tx];
         Cs[2] += As[ty+Tw[2]][k] * Bs[k][tx];
         Cs[3] += As[ty+Tw[3]][k] * Bs[k][tx];
         Cs[4] += As[ty+Tw[4]][k] * Bs[k][tx];
         Cs[5] += As[ty+Tw[5]][k] * Bs[k][tx];
         Cs[6] += As[ty+Tw[6]][k] * Bs[k][tx];
         Cs[7] += As[ty+Tw[7]][k] * Bs[k][tx];
       }

     }
      __syncthreads();

    for(int kk = 0; kk < edge_limit; kk++)
    {
      // #pragma unroll
      // for(int i = 0; i < 4; i++)
      // {
      //   As[ty+Tw[i]][tx] = A[(I+Tw[i])*N+(kk*TW+tx)];
      // }
      // #pragma unroll
      // for(int i = 0; i < 4; i++)
      // {
      //   Bs[ty+Tw[i]][tx] = B[(kk*TW+ty+Tw[i])*N+J];
      // }
      As[ty][tx] = A[I*N+(kk*TW+tx)];
      As[ty+Tw[1]][tx] = A[(I+Tw[1])*N+(kk*TW+tx)];
      As[ty+Tw[2]][tx] = A[(I+Tw[2])*N+(kk*TW+tx)];
      As[ty+Tw[3]][tx] = A[(I+Tw[3])*N+(kk*TW+tx)];
      As[ty+Tw[4]][tx] = A[(I+Tw[4])*N+(kk*TW+tx)];
      As[ty+Tw[5]][tx] = A[(I+Tw[5])*N+(kk*TW+tx)];
      As[ty+Tw[6]][tx] = A[(I+Tw[6])*N+(kk*TW+tx)];
      As[ty+Tw[7]][tx] = A[(I+Tw[7])*N+(kk*TW+tx)];
      Bs[ty][tx] = B[(kk*TW+ty)*N+J];
      Bs[ty+Tw[1]][tx] = B[(kk*TW+ty+Tw[1])*N+J];
      Bs[ty+Tw[2]][tx] = B[(kk*TW+ty+Tw[2])*N+J];
      Bs[ty+Tw[3]][tx] = B[(kk*TW+ty+Tw[3])*N+J];
      Bs[ty+Tw[4]][tx] = B[(kk*TW+ty+Tw[4])*N+J];
      Bs[ty+Tw[5]][tx] = B[(kk*TW+ty+Tw[5])*N+J];
      Bs[ty+Tw[6]][tx] = B[(kk*TW+ty+Tw[6])*N+J];
      Bs[ty+Tw[7]][tx] = B[(kk*TW+ty+Tw[7])*N+J];
      __syncthreads();


      for(int k = 0; k < TW; k++)
      {
        Cs[0] += As[ty][k] * Bs[k][tx];
        Cs[1] += As[ty+Tw[1]][k] * Bs[k][tx];
        Cs[2] += As[ty+Tw[2]][k] * Bs[k][tx];
        Cs[3] += As[ty+Tw[3]][k] * Bs[k][tx];
        Cs[4] += As[ty+Tw[4]][k] * Bs[k][tx];
        Cs[5] += As[ty+Tw[5]][k] * Bs[k][tx];
        Cs[6] += As[ty+Tw[6]][k] * Bs[k][tx];
        Cs[7] += As[ty+Tw[7]][k] * Bs[k][tx];
      }

      __syncthreads();
    }

// not helping.

    // for(int Ci = I,csub=0; (Ci<N)&&(Ci<I+TW); Ci += Tw[1],csub += 1) {
    //   if(J<N) C[Ci*N+J] = Cs[csub];
    // }

    // for(int Ci = 0; Ci <4 && (I+Tw[1]*Ci) < N; Ci++)
    //       {
    //         if(J<N)
    //         C[(I+Tw[1]*Ci)*N + J] = Cs[Ci];
    //       }

    if((I<N)&&(J<N))
    {
      C[I*N+J] = Cs[0];
      if(I+Tw[1] < N)
        C[(I+Tw[1])*N+J] = Cs[1];
      if(I+Tw[2] < N)
        C[(I+Tw[2])*N+J] = Cs[2];
      if(I+Tw[3] < N)
        C[(I+Tw[3])*N+J] = Cs[3];
      if(I+Tw[4] < N)
        C[(I+Tw[4])*N+J] = Cs[4];
      if(I+Tw[5] < N)
        C[(I+Tw[5])*N+J] = Cs[5];
      if(I+Tw[6] < N)
        C[(I+Tw[6])*N+J] = Cs[6];
      if(I+Tw[7] < N)
        C[(I+Tw[7])*N+J] = Cs[7];
      // if(I+Tw[3] < N) {
      //   C[I*N+J] = Cs[0];
      //   C[(I+Tw[1])*N+J] = Cs[1];
      //   C[(I+Tw[2])*N+J] = Cs[2];
      //   C[(I+Tw[3])*N+J] = Cs[3];
      // } else if(I+Tw[2] < N){
      //   C[I*N+J] = Cs[0];
      //   C[(I+Tw[1])*N+J] = Cs[1];
      //   C[(I+Tw[2])*N+J] = Cs[2];
      // } else if(I+Tw[1] < N) {
      //   C[I*N+J] = Cs[0];
      //   C[(I+Tw[1])*N+J] = Cs[1];
      // } else if(I < N) {
      //   C[I*N+J] = Cs[0];
      // }
    }
}
