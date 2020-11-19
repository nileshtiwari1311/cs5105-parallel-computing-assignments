#include <bits/stdc++.h>
#include <thread>
#include <mutex>

using namespace std;

pthread_mutex_t thread_lock; 
pthread_barrier_t thread_barrier;

void iterative_stencil(int tid, int nt, vector<vector<double>> &Original_Matrix, vector<vector<double>> &Updated_Matrix, double &max_diff, double threshold, int &n_iter)
{
    int n = Original_Matrix.size(); 
    int rows_per_thread = n/nt;

    int begin = rows_per_thread*tid;
    int end = begin + rows_per_thread;
    if(tid == nt-1) end = n;

    while(max_diff >= threshold )
    {
        double temp_max_diff = 0;

        for(int i=begin; i<end; i++)
        {
            for(int j=0; j<n; j++)
            {
                int neighbor_count = 0;
                double sum = Original_Matrix[i][j];
                if(i-1 >= 0)
                {
                    sum += Original_Matrix[i-1][j];
                    neighbor_count++;
                }
                if(j-1 >= 0)
                {
                    sum += Original_Matrix[i][j-1];
                    neighbor_count++;
                }
                if(i+1 < n)
                {
                    sum += Original_Matrix[i+1][j];
                    neighbor_count++;
                }
                if(j+1 < n)
                {
                    sum += Original_Matrix[i][j+1];
                    neighbor_count++;
                }

                Updated_Matrix[i][j] = sum/(neighbor_count+1);
                temp_max_diff = max(temp_max_diff, abs(Original_Matrix[i][j] - Updated_Matrix[i][j]));
            }
        }

        pthread_barrier_wait(&thread_barrier);
        if(tid == 0)
        {
            pthread_mutex_lock(&thread_lock);
            max_diff = 0;
            n_iter++;
            pthread_mutex_unlock(&thread_lock);
        }
        pthread_barrier_wait(&thread_barrier);

        for(int i=begin; i<end; i++)
        {
            for(int j=0; j<n; j++)
            {
                Original_Matrix[i][j] = Updated_Matrix[i][j];
            }
        }

        pthread_mutex_lock(&thread_lock);
        max_diff = max(max_diff, temp_max_diff);
        pthread_mutex_unlock(&thread_lock);
        pthread_barrier_wait(&thread_barrier);
    }    
}

void example_random_number_generator(vector<vector<double>> &matrix)
{
   double lbound = 0;
   double ubound = 10;
   std::uniform_real_distribution<double> urd(lbound, ubound);
   std::default_random_engine re;
   int n = matrix.size();
   for(int i=0; i<n; i++)
   {
       for(int j=0; j<n; j++)
       {
           matrix[i][j] = urd(re);
       }
   }
}

int main(int argc, char **argv)
{
    int n = atoi(argv[1]); // number of rows
    double threshold = atof(argv[2]);

    int nt = n; // number of threads
    double sum_time = 0, average_time = 0;
    for(int i=0; i<10; i++)
    {
        pthread_barrier_init(&thread_barrier, NULL, nt);
        double max_diff = INT_MAX;

        vector<vector<double>> Original_Matrix(n,vector<double>(n,0)), Updated_Matrix(n,vector<double>(n,0));
        example_random_number_generator(Original_Matrix);
        int count = 0, n_iter = 0;
        vector<thread> threads;

        auto start = chrono::high_resolution_clock::now();

        for(int i=0;i<nt;i++)
        {
            threads.push_back( thread(iterative_stencil, i, nt, ref(Original_Matrix), ref(Updated_Matrix), ref(max_diff), threshold, ref(n_iter)) );
        }
        for(auto &th:threads)
            th.join();        

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

        cout << "\n-----------------------------" << endl;
        cout << "number of iterations =" << n_iter << endl;
        cout << "time taken = " << duration.count() <<" microseconds" << endl;
        cout << "array after " << n_iter << " iterations :" << endl;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                cout << Original_Matrix[i][j] << " ";
            }
            cout << endl;
        }

        sum_time += (double)duration.count();
    }
    average_time = sum_time/10.0;
    cout << "\n-----------------------------" << endl;
    cout << "average time taken = " << average_time << endl;
}