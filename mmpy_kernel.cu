// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

      const unsigned int TW = 32;
      const unsigned int TW1 = TW/4, TW2 = TW/2, TW3 = 3*TW/4;
      const unsigned int edge_limit = N/TW;
      const unsigned int ty = threadIdx.y, tx = threadIdx.x;
      const unsigned int by = blockIdx.y, bx = blockIdx.x;
      const unsigned int I = by*TW + ty, J = bx*TW + tx;

     __shared__ _DOUBLE_ As[TW][TW], Bs[TW][TW];

     _DOUBLE_ Cs[4] = {0};
     As[ty][tx] = 0;
     As[ty+TW1][tx] = 0;
     As[ty+TW2][tx] = 0;
     As[ty+TW3][tx] = 0;

     Bs[ty][tx] = 0;
     Bs[ty+TW1][tx] = 0;
     Bs[ty+TW2][tx] = 0;
     Bs[ty+TW3][tx] = 0;

     if(N%TW !=0)
     {
       for(int Ai =I; (Ai < N) && (Ai < I + TW); Ai +=TW/4)
       {
         if(edge_limit*TW+tx < N)
          As[ty+Ai-I][tx] = A[(Ai)*N+(edge_limit*TW+tx)];
       }
       int b_start = edge_limit*TW+ty;
       for(int Bi = b_start;(Bi<N) && (Bi < b_start+TW) ; Bi+=TW/4)
       {
        if(J < N)
          Bs[Bi-edge_limit*TW][tx] = B[Bi*N+J];

       }
       __syncthreads();

       for(int k = 0; k < TW; k++)
       {
         Cs[0] += As[ty][k] * Bs[k][tx];
         Cs[1] += As[ty+TW1][k] * Bs[k][tx];
         Cs[2] += As[ty+TW2][k] * Bs[k][tx];
         Cs[3] += As[ty+TW3][k] * Bs[k][tx];
       }

     }
      __syncthreads();

    for(int kk = 0; kk < edge_limit; kk++)
    {

      As[ty][tx] = A[I*N+(kk*TW+tx)];
      As[ty+TW1][tx] = A[(I+TW1)*N+(kk*TW+tx)];
      As[ty+TW2][tx] = A[(I+TW2)*N+(kk*TW+tx)];
      As[ty+TW3][tx] = A[(I+TW3)*N+(kk*TW+tx)];
      Bs[ty][tx] = B[(kk*TW+ty)*N+J];
      Bs[ty+TW1][tx] = B[(kk*TW+ty+TW1)*N+J];
      Bs[ty+TW2][tx] = B[(kk*TW+ty+TW2)*N+J];
      Bs[ty+TW3][tx] = B[(kk*TW+ty+TW3)*N+J];
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < TW; k++)
      {
        Cs[0] += As[ty][k] * Bs[k][tx];
        Cs[1] += As[ty+TW1][k] * Bs[k][tx];
        Cs[2] += As[ty+TW2][k] * Bs[k][tx];
        Cs[3] += As[ty+TW3][k] * Bs[k][tx];
      }

      __syncthreads();
    }
    if((J<N))
    {
      if(I+TW3 < N) {
        C[I*N+J] = Cs[0];
        C[(I+TW1)*N+J] = Cs[1];
        C[(I+TW2)*N+J] = Cs[2];
        C[(I+TW3)*N+J] = Cs[3];
      } else if(I+TW2 < N){
        C[I*N+J] = Cs[0];
        C[(I+TW1)*N+J] = Cs[1];
        C[(I+TW2)*N+J] = Cs[2];
      } else if(I+TW1 < N) {
        C[I*N+J] = Cs[0];
        C[(I+TW1)*N+J] = Cs[1];
      } else if(I < N) {
        C[I*N+J] = Cs[0];
      }
    }
}
