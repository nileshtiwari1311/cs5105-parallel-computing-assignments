#include<stdio.h>

__global__ void Matrix_mult(int *a, int *b, int *c, int *m, int *n, int *p)
{
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0, i;

    if(row<*m && col<*p)
    for(i=0; i<*n; i++)
      temp += a[row * (*n) + i] * b[i * (*p) + col];
    c[row * (*p) + col] = temp;
}

int main()
{
    int m = 3,n = 2,p = 2, i, j;
    int a[m*n], b[n*p], c[m*n];
    int *cuda_a, *cuda_b, *cuda_c, *cuda_m, *cuda_n, *cuda_p;
    printf("Matrix A:\n");
    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            a[i*n+j] = rand()%100;
            printf("%d ", a[i*n+j]);       
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for(i=0; i<n; i++)
    {
        for(j=0; j<p; j++)
        {
            b[i*p+j] = rand()%100;
            printf("%d ",b[i*p+j]);       
        }
        printf("\n");
    }
    printf("\n");

    cudaMalloc((void**)&cuda_a, m*n*sizeof(int));
    cudaMalloc((void**)&cuda_b, n*p*sizeof(int));
    cudaMalloc((void**)&cuda_c, m*p*sizeof(int));
    cudaMalloc((void**)&cuda_m, sizeof(int));
    cudaMalloc((void**)&cuda_n, sizeof(int));
    cudaMalloc((void**)&cuda_p, sizeof(int));

    cudaMemcpy(cuda_a, a, m*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, n*p*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_m, &m, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_p, &p, sizeof(int), cudaMemcpyHostToDevice);
 
    dim3 threadsPerBlock(m, p);
    dim3 blocksPerGrid(1, 1);
        

    Matrix_mult<<<blocksPerGrid,threadsPerBlock>>> (cuda_a, cuda_b, cuda_c, cuda_m, cuda_n, cuda_p);   

    cudaMemcpy(c, cuda_c, m*p*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result Matrix:\n");
    for(i=0; i<m; i++)
    {
        for(j=0; j<p; j++)
        {
            printf("%d ", c[i*p+j]);       
        }
        printf("\n");
    }
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    cudaFree(cuda_m);
    cudaFree(cuda_n);
    cudaFree(cuda_p);
    return 0;
}