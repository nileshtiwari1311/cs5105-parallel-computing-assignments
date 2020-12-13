#include<stdio.h>

__managed__ int sum=0;

__global__ void Array_sum(int *a, int *n)
{
    int tid = threadIdx.x;
    if(tid < *n)
      atomicAdd(&sum, a[tid]);
}

int main()
{
    int n = 10, i;
    int a[n];
    int *cuda_a, *cuda_n;
 
    for(i=0; i<n; i++)
    {
        a[i] = rand()%100;
        printf("%d ", a[i]);
    }
    printf("\n");

    cudaMalloc((void**)&cuda_a, n*sizeof(int));
    cudaMalloc((void**)&cuda_n, sizeof(int));
    
    cudaMemcpy(cuda_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    Array_sum <<<1, n>>>(cuda_a, cuda_n);
    printf("Sum:%d\n", sum);
    cudaFree(cuda_a);
    cudaFree(cuda_n);
    return 0;
}