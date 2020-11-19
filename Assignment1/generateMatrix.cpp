#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv)
{
	int n = atoi(argv[1]);
	int A[n][n];
	ofstream fout;
	fout.open("matrix.txt");

	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			if(i==j)
				A[i][j] = 0;
			else if(i < j)
				A[i][j] = rand()%2;
			else A[i][j] = A[j][i];
			fout << A[i][j] << " ";
		}
		fout << endl;
	}

	cout << "matrix generated. check matrix.txt" << endl;

	fout.close();
	return 0;
}
