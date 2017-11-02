void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
   blockDim.x = 16;
   blockDim.y = 16;
   gridDim.x = n / blockDim.x;
   gridDim.y = n / blockDim.y;
   if(n % blockDim.x != 0)
   	gridDim.x++;
   if(n % blockDim.y != 0)
    	gridDim.y++;
}
