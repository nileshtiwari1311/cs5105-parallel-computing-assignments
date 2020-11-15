#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <random>
#include <iostream>
#include <chrono>

using namespace std::chrono;

int main(int argc, char **argv)
{
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    double a[n];
    for (int j = 0; j < np; j++)
    {
        if (rank == j)
        {
            printf("Rank:%d\n",rank);
            double lbound = 0;
            double ubound = 10;
            std::uniform_real_distribution<double> urd(lbound, ubound);
            std::default_random_engine re;
            for (int i = 0; i < n; i++)
            {
                //a[i] = urd(re);
                a[i]=rank;
                printf("%f ", a[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    auto start = high_resolution_clock::now();
    int cur=0,val;
    for(int i=0;i<m;i++)
    {
        if(cur==rank)
        {
            val=rand()%np;
        }
        MPI_Bcast(&val,1,MPI_INT,cur,MPI_COMM_WORLD);
        if(val==cur)
            continue;
        if(cur==rank)
        {
            MPI_Send(&a[0],n,MPI_DOUBLE,val,1,MPI_COMM_WORLD);
        }
        if(rank==val)
        {
            MPI_Recv(&a[0],n,MPI_DOUBLE,cur,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        cur=val;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    for (int j = 0; j < np; j++)
    {
        if (rank == j)
        {
            printf("Rank:%d\n",rank);
            for (int i = 0; i < n; i++)
            {
                printf("%f ", a[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    //std::cout << duration.count() << std::endl;
}