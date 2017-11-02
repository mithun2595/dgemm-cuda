// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    // int I =  blockIdx.y*blockDim.y + threadIdx.y;
    // int J =  blockIdx.x*blockDim.x + threadIdx.x;

    // if((I < N) && (J < N)){
//        _DOUBLE_ _c = 0;
//        for (unsigned int k = 0; k < N; k++) {
//            _DOUBLE_ a = A[I * N + k];
//            _DOUBLE_ b = B[k * N + J];
//            _c += a * b;
//        }
//        C[I * N + J] = _c;
    // }
    const int TW = 16;

    const int TX = blockDim.x;
    const int TY = blockDim.y;

    int edge_limit = (int) ceilf((float)N/TW);

    // __shared__ _DOUBLE_ **As, **Bs;

    // As = (double*) malloc(TX * TY * sizeof(double));
    // Bs = (double*) malloc(TX * TY * sizeof(double));

    __shared__ _DOUBLE_ As[TW][TW], Bs[TW][TW];

     int ty = threadIdx.y, tx = threadIdx.x;
     int by = blockIdx.y, bx = blockIdx.x;
     int I = by*TW + ty, J = bx*TW + tx;
    //  int i = threadIdx.y;
    //  int j = threadIdx.x;
    //  int ii = blockIdx.y;
    //  int jj = blockIdx.x;

    //  int TI = 16, TJ = 16, TK = 32;

     _DOUBLE_ Cij = 0;
    // if((ii*TI + i < N) && (jj*TJ + j < N))
    // {
       // for(int kk = 0; kk < N; kk+= TK)
       // {
        //  for(int k =0; k < N; k++)
        //  {
        //    Cij += A[(ii*TI+i)*N + k]*B[k*N + jj*TJ + j];
        //  }
       // }

      //  C[(ii*TI+i)*N + jj*TJ + j] = Cij;
    // }

    for(int kk = 0; kk < edge_limit; kk++)
    {

      if((I<N)&&((kk*TW+tx)<N))
      {
          As[ty][tx] = A[I*N+(kk*TW+tx)];
      }
      else
      {
        As[ty][tx] = 0;
      }

      if(((kk*TW+ty)<N)&&(J<N))
      {
          Bs[ty][tx] = B[(kk*TW+ty)*N+J];
      }
      else
      {
        Bs[ty][tx] = 0;
      }
      __syncthreads();

      for(int k = 0; k < TW; k++)
      {
        Cij += As[ty][k]*Bs[k][tx];
      }
      __syncthreads();
    }
    if((I<N)&&(J<N))
    {
        C[I*N+J] = Cij;
    }




}
