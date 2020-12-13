#include<stdio.h>

__global__ void Array_add(int *a, int *b, int *c, int *n)
{
    unsigned short tid = threadIdx.x;
    if(tid < *n)
      c[tid] = a[tid] + b[tid];
}

int main()
{
    int n = 5, i;
    int a[n], b[n], c[n];
    int *cuda_a, *cuda_b, *cuda_c, *cuda_n;
 
    for(i=0; i<n; i++)
      a[i] = rand()%100;
    for(i=0; i<n; i++)
      b[i] = rand()%100;

    cudaMalloc((void**)&cuda_a, n*sizeof(int));
    cudaMalloc((void**)&cuda_b, n*sizeof(int));
    cudaMalloc((void**)&cuda_c, n*sizeof(int));
    cudaMalloc((void**)&cuda_n, sizeof(int));
    
    cudaMemcpy(cuda_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    Array_add <<<1, n>>>(cuda_a, cuda_b, cuda_c, cuda_n);
    cudaMemcpy(c, cuda_c, n*sizeof(int), cudaMemcpyDeviceToHost);
    for(i=0; i<n; i++)
      printf("%d + %d = %d\n", a[i], b[i], c[i]);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    cudaFree(cuda_n);
    return 0;
}