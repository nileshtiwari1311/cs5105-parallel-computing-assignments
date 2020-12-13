#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/generate.h>
#include<thrust/sort.h>
#include<thrust/copy.h>
#include<cstdlib>

int main()
{
    thrust::host_vector<int> H(22);
    thrust::generate(H.begin(), H.end(), rand);
    thrust::device_vector<int> D = H;
    thrust::sort(D.begin(), D.end());
    thrust::copy(D.begin(), D.end(), H.begin());
    for(int i=0; i<H.size(); i++)
      printf("%d ",H[i]);
    return 0;
}