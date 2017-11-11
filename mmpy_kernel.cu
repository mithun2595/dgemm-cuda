// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

      register const unsigned int TW = 32;
      register const unsigned int Tw[8] = {0,TW/8, TW/4, 3*TW/8, TW/2, 5*TW/8, 3*TW/4, 7*TW/8}; // Array used to do unrolling.
      const unsigned int edge_limit = N/TW;
      register const unsigned int ty = threadIdx.y, tx = threadIdx.x;
      register const unsigned int by = blockIdx.y, bx = blockIdx.x;
      register const unsigned int I = by*TW + ty, J = bx*TW + tx;

      __shared__ _DOUBLE_ As[TW][TW], Bs[TW][TW]; // Shared memory used to store the tiles of A and B

     register _DOUBLE_ Cs[8] = {0}; // Array used to store the computed value of C.
     register int i = 0;
     register int kk = 0;

     // This section of the code handles the initialising of the As and Bs to 0.
    #pragma unroll
    for(i = 0; i < 8; i++)
    {
        As[ty+Tw[i]][tx] = 0;
    }
     #pragma unroll
     for(i = 0; i < 8; i++)
     {
        Bs[ty+Tw[i]][tx] = 0;
     }

     // This section of the code handles the corner cases.
     if(N%TW !=0)
     {
       register int a_1 = ty - I;
       register int a_2 = edge_limit*TW+tx;
       for(int Ai =I; (Ai < N) && (Ai < I + TW); Ai +=Tw[1])
       {
         if(edge_limit*TW+tx < N)
          As[a_1+Ai][tx] = A[(Ai)*N+a_2];
       }
       register int b_start = edge_limit*TW+ty;
       register int b_1 = edge_limit*TW;
       for(int Bi = b_start;(Bi<N) && (Bi < b_start+TW) ; Bi+=Tw[1])
       {
        if(J < N)
          Bs[Bi - b_1][tx] = B[Bi*N+J];

       }
       __syncthreads();

       // This section handles the computation of Cs
       for(int k = 0; k < TW; k++)
       {
         #pragma unroll
         for(i = 0 ;i < 8; i++)
         {
            Cs[i] += As[ty+Tw[i]][k] * Bs[k][tx];
         }
       }

     }
      __syncthreads();

      // This seciton of the code is for the perfect case without corner cases.
    for( kk = 0; kk < edge_limit; kk++)
    {
      // Loading of A into As and B into Bs
      #pragma unroll
      for(i = 0; i < 8; i++)
      {
          As[ty+Tw[i]][tx] = A[(I+Tw[i])*N+(kk*TW+tx)];
      }

      register int B_1 = kk*TW+ty;
      #pragma unroll
      for(i = 0; i < 8; i++)
      {
          Bs[ty+Tw[i]][tx] = B[(B_1+Tw[i])*N+J];
      }

      __syncthreads();

      // Computing the values of C
      for(int k = 0; k < TW; k++)
      {
        #pragma unroll
        for(i = 0; i < 8; i++)
        {
            Cs[i] += As[ty+Tw[i]][k] * Bs[k][tx];
        }

      }
      __syncthreads();
    }
    //Storing the values of Cs to C while checking the indices within range
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

     }
}
