// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

      const unsigned int TW = 32;
      register const unsigned int Tw[3] = {TW/4, TW/2, 3*TW/4};
      const unsigned int edge_limit = N/TW;
      const unsigned int ty = threadIdx.y, tx = threadIdx.x;
      const unsigned int by = blockIdx.y, bx = blockIdx.x;
      const unsigned int I = by*TW + ty, J = bx*TW + tx;

      register __shared__ _DOUBLE_ As[TW][TW], Bs[TW][TW];

     _DOUBLE_ Cs[4] = {0};
     As[ty][tx] = 0;
     As[ty+Tw[0]][tx] = 0;
     As[ty+Tw[1]][tx] = 0;
     As[ty+Tw[2]][tx] = 0;

     Bs[ty][tx] = 0;
     Bs[ty+Tw[0]][tx] = 0;
     Bs[ty+Tw[1]][tx] = 0;
     Bs[ty+Tw[2]][tx] = 0;

     if(N%TW !=0)
     {
       for(int Ai =I; (Ai < N) && (Ai < I + TW); Ai +=Tw[0])
       {
         if(edge_limit*TW+tx < N)
          As[ty+Ai-I][tx] = A[(Ai)*N+(edge_limit*TW+tx)];
       }
       int b_start = edge_limit*TW+ty;
       for(int Bi = b_start;(Bi<N) && (Bi < b_start+TW) ; Bi+=Tw[0])
       {
        if(J < N)
          Bs[Bi-edge_limit*TW][tx] = B[Bi*N+J];

       }
       __syncthreads();

       for(int k = 0; k < TW; k++)
       {
         Cs[0] += As[ty][k] * Bs[k][tx];
         Cs[1] += As[ty+Tw[0]][k] * Bs[k][tx];
         Cs[2] += As[ty+Tw[1]][k] * Bs[k][tx];
         Cs[3] += As[ty+Tw[2]][k] * Bs[k][tx];
       }

     }
      __syncthreads();

    for(int kk = 0; kk < edge_limit; kk++)
    {

      As[ty][tx] = A[I*N+(kk*TW+tx)];
      As[ty+Tw[0]][tx] = A[(I+Tw[0])*N+(kk*TW+tx)];
      As[ty+Tw[1]][tx] = A[(I+Tw[1])*N+(kk*TW+tx)];
      As[ty+Tw[2]][tx] = A[(I+Tw[2])*N+(kk*TW+tx)];
      Bs[ty][tx] = B[(kk*TW+ty)*N+J];
      Bs[ty+Tw[0]][tx] = B[(kk*TW+ty+Tw[0])*N+J];
      Bs[ty+Tw[1]][tx] = B[(kk*TW+ty+Tw[1])*N+J];
      Bs[ty+Tw[2]][tx] = B[(kk*TW+ty+Tw[2])*N+J];
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < TW; k++)
      {
        Cs[0] += As[ty][k] * Bs[k][tx];
        Cs[1] += As[ty+Tw[0]][k] * Bs[k][tx];
        Cs[2] += As[ty+Tw[1]][k] * Bs[k][tx];
        Cs[3] += As[ty+Tw[2]][k] * Bs[k][tx];
      }

      __syncthreads();
    }

// not helping.

    // for(int Ci = I,csub=0; (Ci<N)&&(Ci<I+TW); Ci += TW1,csub += 1) {
    //   if(J<N) C[Ci*N+J] = Cs[csub];
    // }

    // if((J<N))
    // {
      // if(I+TW3 < N) {
      //   C[I*N+J] = Cs[0];
      //   C[(I+TW1)*N+J] = Cs[1];
      //   C[(I+TW2)*N+J] = Cs[2];
      //   C[(I+TW3)*N+J] = Cs[3];
      // } else if(I+TW2 < N){
      //   C[I*N+J] = Cs[0];
      //   C[(I+TW1)*N+J] = Cs[1];
      //   C[(I+TW2)*N+J] = Cs[2];
      // } else if(I+TW1 < N) {
      //   C[I*N+J] = Cs[0];
      //   C[(I+TW1)*N+J] = Cs[1];
      // } else if(I < N) {
      //   C[I*N+J] = Cs[0];
      // }
      #pragma unroll
      for(int Ci = 0; Ci < (blockDim.x/blockDim.y) && (I+Tw[0]*Ci) < N; Ci++)
      {
        if(J<N)
        C[(I+Tw[0]*Ci)*N + J] = Cs[Ci];
      }
    // }
}
