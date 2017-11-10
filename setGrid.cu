void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
   blockDim.x = 32;
   blockDim.y = 4;
   gridDim.x = n / blockDim.x;
   gridDim.y = n / (blockDim.y*8);
   if(n % blockDim.x != 0)
   	gridDim.x++;
   if(n % (blockDim.y*8) != 0)
    	gridDim.y++;
}
