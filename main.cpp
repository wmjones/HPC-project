#include <omp.h>
#include <stdio.h>
#include <thread>

int main(){
    unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
    printf("%d\n", concurentThreadsSupported);
}
