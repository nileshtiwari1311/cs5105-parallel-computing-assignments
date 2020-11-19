#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
	int rank, np; // np is the number of processes
	int sum, offset, flag = 1, count = 1;

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int n = atoi(argv[1]); // number of nodes in the graph
    int s = atoi(argv[2]); // source
    int d = atoi(argv[3]); // destination

    // check for valid input parameters n, s, d, np
    if(n <=0 || s >= n || d >= n || n%np != 0)
    {
        if(rank == 0)
            cout << "Invalid parameters...EXITING" << endl;
        MPI_Finalize();
        exit(0);
    }

    // A contains the adjacency matrix, Ak contains the matrix after multiplication
    // temp contains the partial matrix to store elements after matrix multiplication
    int A[n][n], Ak[n][n], temp[n/np][n];
    
    // read the adjacency matrix A from a file and set Ak = A
    ifstream fin;
    fin.open("matrix.txt");
    string line = "";
    int line_count = 0;
    while(getline(fin, line))
    {
        for(int i=0; i<n; i++)
        {
            A[line_count][i] = line[2*i] - '0';
            Ak[line_count][i] = A[line_count][i];
        }
        line_count++;
    }
    fin.close();

    int n1 = (n * n) / np; // n1 is the number of elements of the partial matrix 
    int base = (n * rank) / np; // base is the base address from the Ak matrix on which multiplication has to be done by each process

    if(s == d || Ak[s][d] > 0) flag = 0;

    if(rank == 0)
    {
    	cout << "\nAk matrix (before) " << endl;
    	for(int i=0; i<n; i++)
    	{
    		for(int j=0; j<n; j++)
    		{
    			cout << Ak[i][j] << "\t";
    		}
    		cout << endl;
    	}
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    double t1 = clock();
    while(flag)
    {
        // do local computation of (n*n)/np elements of Ak matrix starting from a base address
    	for(int i=base; i<(base + n/np); i++)
        {
            for(int j=0; j<n; j++)
            {
                sum = 0;
                for(int k=0; k<n; k++)
                {
                    sum += (Ak[i][k] * A[k][j]);
                }
                temp[i-base][j] = sum;
            }
        }

        for(int i=0; i<n/np; i++)
        {
        	for(int j=0; j<n; j++)
        	{
        		Ak[i+base][j] = temp[i][j];
        	}
        }

        count++;

        if(rank == 0)
        {
            // receive updated partial Ak matrices from all other processes
            // only process 0 contains the updated complete Ak matrix
        	for(int i=1; i<np; i++)
         	{
	            MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	            MPI_Recv(&Ak[offset][0], n1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	            
         	}

         	if(Ak[s][d] > 0 || count == n) flag = 0; // compute flag to whether continue the while loop or exit

            // send the flag to all other processes
         	for(int i=1; i<np; i++)
         		MPI_Send(&flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

        }
        else
        {
            // send the updated partial Ak matrix of (n*n)/np elements starting from base address
        	MPI_Send(&base, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        	MPI_Send(&Ak[base][0], n1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            //receive the flag from process 0
        	MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    t1 = clock() - t1;

    if(rank == 0)
    {
    	cout << "\nAk matrix (after) " << endl;
    	for(int i=0; i<n; i++)
    	{
    		for(int j=0; j<n; j++)
    		{
    			cout << Ak[i][j] << "\t";
    		}
    		cout << endl;
    	}

    	if(s == d)
    	{
    		cout << "\nshortest path length from " << s << " to " << d << " = 0" << endl;
    	}
		else if(count == n)
		{
			cout << "\nno path between " << s << " to " << d << endl;
		}
		else
		{
			cout << "\nshortest path length from " << s << " to " << d << " = " << count << endl;	
		}

		cout << "time taken = " << ((double)t1) / CLOCKS_PER_SEC << endl;
    }

    MPI_Finalize();
}