#### Name - Nilesh Tiwari
#### Roll Number - CS17B022

#### Problem
Implement an MPI based program for shortest path algorithm using matrix multiplication. Assume an undirected graph with edge weights 1. The number of processes is a perfect square. The number of nodes in the graph can be assumed to be a multiple of number of processes. Output is the shortest path. Time the runtime of your algorithm (excluding the file write/read) for various runtime parameters.

##### Sub problem - 1
The adjacency matrix can be assumed to fit on a single process.

###### Code explanation
**generateMatrix.cpp** takes the number of nodes as a command line input and generates an adjacency matrix and dumps it to a file named 'matrix.txt'.

**1.cpp** contains the MPI based program for sub problem 1. 
1. All processes read adjacency matrix into an array A[][] from the file matrix.txt and set Ak[][] which is the matrix after multiplication equal to A[][].
2. All processes do local matrix multiplication of (n * n)/np number of elements, i.e., n/np number of rows of Ak[][] with A[][] where n is the number of nodes and np is the number of processes.
3. All processes except process ranked 0 send the updated partial Ak[][] after multiplication to the process ranked 0.
4. The process ranked 0 after receiving the complete Ak[][], checks if the number of paths from source to destination becomes non-zeo or the matrix has been multiplied (n-1) times. Accordingly, it sets the flag and sends it to all other processes.
5. All processes based on flag repeat the steps 2 to 4.
6. Once the while loop breaks, the count = (number of times matrix has been multiplied + 1) is the length of the shortest path, if the count < n, else no path exists.

##### Sub problem - 2
The adjacency matrix cannot be allocated on a single process, but will occupy all the processes on which the program is launched on.

###### Code explanation
**generateMatrix.cpp** takes the number of nodes as a command line input and generates an adjacency matrix and dumps it to a file named 'matrix.txt'.

**2.cpp** contains the MPI based program for sub problem 2. 
1. All processes read (n/np) number of rows of adjacency matrix starting from ((rank * n)/np)th row into an array A[][] from the file matrix.txt.
2. Ak_s[] contains the sth row of the Ak matrix used in 1.cpp. It is set accordingly and sent to all processes.
3. All processes do local matrix multiplication to set (n/np) elements of Ak_s[].
4. All processes except process ranked 0 send the updated partial Ak_s[] after multiplication to the process ranked 0.
5. The process ranked 0 after receiving the complete Ak_s[], checks if the number of paths from source to destination becomes non-zeo or the matrix has been multiplied (n-1) times. Accordingly, it sets the flag and sends it to all other processes.
6. All processes based on flag repeat the steps 3 to 5.
7. Once the while loop breaks, the count = (number of times matrix has been multiplied + 1) is the length of the shortest path, if the count < n, else no path exists.

#### Code execution
'Makefile' contains the code for generating executables named 'genMax', 'pgm1' and 'pgm2'.

genMax for generating adjacency matrix.

pgm[1/2] to run MPI based program.

##### Compilation + generation of executables 
		make
Or use following commands

		g++ generateMatrix.cpp -o genMax
		mpic++ 1.cpp -o pgm1
		mpic++ 2.cpp -o pgm2

##### Running the program
		./genMax <n>
		mpirun -np <np> ./pgm1 <n> <s> <d>
		mpirun -np <np> ./pgm2 <n> <s> <d>