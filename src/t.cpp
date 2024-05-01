#include <iostream>
#include <omp.h>
int main() {
#pragma omp parallel
    { std::cout << "hello world\n" << std::endl; }

    return 0;
}
