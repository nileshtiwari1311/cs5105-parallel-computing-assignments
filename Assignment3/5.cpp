#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <mutex>

using namespace std::chrono;

const int nthreads = 1;
int64_t n = 10;
std::mutex mtx;

double sum[nthreads] = { 0 };

void thread_sum(int tid,std::vector<double>& a)
{
    int start=(n/nthreads)*tid;
    int end=(n/nthreads)*(tid+1);
    if(tid==nthreads-1)
    {
        end+=(n%nthreads);
    }
    for(int i=start;i<end;i++)
    {
        sum[tid]+=a[i];
    }
}

int main(int argc,char** argv)
{
    int i;
    std::vector<std::thread> threads;
    std::vector<double> a;
    n=atoi(argv[1]);
    double lbound = 0;
    double ubound = 10;
    std::uniform_real_distribution<double> urd(lbound, ubound);
    std::default_random_engine re;
    for(i=0;i<n;i++)
    {
        a.push_back(urd(re));
        printf("%f ",a[i]);
    }
    printf("\n");
    auto start = high_resolution_clock::now();

    for(i=0;i<nthreads;i++)
    {
        threads.push_back(std::thread(thread_sum,i,std::ref(a)));
    }
    for(auto &th:threads)
    {
        th.join();
    }
    double total_sum=0;
    for(i=0;i<nthreads;i++)
    {
        total_sum+=sum[i];
    }
    auto stop = high_resolution_clock::now();
    std::cout << "Array sum is: " << total_sum << std::endl;
}