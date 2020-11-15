#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <random>
#include <iostream>
#include <chrono>

using namespace std::chrono;

int main(int argc, char **argv)
{
    int rank,np;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int n= atoi(argv[1]);
    double a[n];
    if(rank==0)
    {
        double lbound = 0;
        double ubound = 10;
        std::uniform_real_distribution<double> urd(lbound, ubound);
        std::default_random_engine re;
        for (int i = 0; i < n; i++)
        {
             a[i]=urd(re);
             printf("%f ",a[i]);
        } 
        printf("\n");
    }

    auto start = high_resolution_clock::now();

    if(rank==0)
    {
        int i,base;
        int n1=n/np;
        if(np>1)
        {
            for(i=1;i<np-1;i++)
            {
                base=i*n1;
                MPI_Send(&n1,1,MPI_INT,i,0,MPI_COMM_WORLD);
                MPI_Send(&a[base],n1,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
            }
            base=i*n1;
            int remain=n-base;
            MPI_Send(&remain,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(&a[base],remain,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
        }
        double sum=0;
        for(i=0;i<n1;i++)
        {
            sum+=a[i];
        }
        double part_sum;
        for(i=1;i<np;i++)
        {
            MPI_Recv(&part_sum,1,MPI_DOUBLE,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            sum+=part_sum;
        }
        printf("Array Sum:%f\n",sum);
    }
    else
    {
        int n1,i;
        MPI_Recv(&n1,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        double a1[n1];
        MPI_Recv(&a1,n1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        double part_sum=0;
        for(i=0;i<n1;i++)
        {
            part_sum+=a1[i];
        }
        MPI_Send(&part_sum,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    //std::cout << duration.count() << std::endl;
}