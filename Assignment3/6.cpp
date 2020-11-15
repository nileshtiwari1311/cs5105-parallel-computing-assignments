#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <random>

int main(int argc,char** argv)
{
    int n=atoi(argv[1]),i;
    double a[n];
    double total_sum=0;
    double lbound = 0;
    double ubound = 10;
    std::uniform_real_distribution<double> urd(lbound, ubound);
    std::default_random_engine re;
    for(i=0;i<n;i++)
    {
        a[i]=urd(re);
        printf("%f ",a[i]);
    }
    printf("\n");
    #pragma omp parallel for reduction(+: total_sum)
        for(i=0;i<n;i++)
        {
            total_sum+=a[i];
        }
    printf("Total Sum:%f\n",total_sum);
}