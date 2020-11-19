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

    int n1 = n / np; // n1 is the number of rows per process
    int base = (n * rank) / np; // base is the base address of the element in the Ak_s from which n1 elements will be updated by any process 

    // A contains the partial adjacency matrix, Ak_s contains the sth row of matrix after multiplication
    // temp contains the partial matrix to store n1 elements after matrix multiplication
    int A[n1][n], Ak_s[n], temp[n1];
    
    // read the adjacency matrix A (only n1 rows) from a file
    ifstream fin;
    fin.open("matrix.txt");
    string line = "";
    int line_count = 0;
    int rows = 0;
    while(getline(fin, line))
    {
        if(rank > line_count/n1) 
        {
            line_count++;
            continue;
        }
        else if(rank == line_count/n1)
        {
            for(int i=0; i<n; i++)
            {
                A[rows][i] = line[2*i] - '0';
            }
            rows++;
            line_count++;
        }
        else break;
    }
    fin.close();

    if(rank == 0)
    {
    	cout << "\nA matrix " << endl;
    }

    for(int i=0; i<n; i++)
    {
        if(rank == i/n1)
        {
            for(int j=0; j<n; j++)
            {
                cout << A[i%n1][j] << "\t";
            }
            cout << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // set Ak_s from the A[][] of the process which contains the sth row of complete adjacency matrix
    if(rank == s/n1)
    {
        for(int i=0; i<n; i++)
        {
            Ak_s[i] = A[s%n1][i];
        }
    }

    // update Ak_s[] of each process
    MPI_Bcast(&Ak_s, n, MPI_INT, s/n1, MPI_COMM_WORLD);

    if(s == d || Ak_s[d] > 0) flag = 0;
    
    double t1 = clock();
    while(flag)
    {
        // send updated value of Ak_s from process ranked 0 to other processes
        MPI_Bcast(&Ak_s, n, MPI_INT, 0, MPI_COMM_WORLD);

        // do local computation of n1 elements of the Ak_s (sth row of Ak matrix)
        for(int i=0; i<n1; i++)
        {
            sum = 0;
            for(int j=0; j<n; j++)
            {
                sum += (Ak_s[j] * A[i][j]); // as A[][] is symmetric, so either multiply with ith row or ith column, result will be same
            }
            temp[i] = sum;
        }

        if(rank == 0)
        {
            for(int i=0; i<n1; i++) Ak_s[i] = temp[i];
        }

        count++;

        if(rank == 0)
        {
            // receive updated partial n1 elements of Ak_s[] from all other processes
        	for(int i=1; i<np; i++)
         	{
	            MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	            MPI_Recv(&Ak_s[offset], n1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	            
         	}

         	if(Ak_s[d] > 0 || count == n) flag = 0; // compute flag to whether continue the while loop or exit

            // send the flag to all other processes
         	for(int i=1; i<np; i++)
         		MPI_Send(&flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

        }
        else
        {
            // send the updated partial n1 elements of temp
        	MPI_Send(&base, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        	MPI_Send(&temp, n1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            //receive the flag from process 0
        	MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    t1 = clock() - t1;

    if(rank == 0)
    {
    	cout << "\n" << s << "th " << "(sth) row of Ak matrix (after multiplication) " << endl;
    	for(int i=0; i<n; i++)
    	{
    		cout << Ak_s[i] << "\t";
    	}
        cout << endl;

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