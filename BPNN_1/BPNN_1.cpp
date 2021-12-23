#include "NumCpp.hpp"
#include "network.h"

#include <cstdlib>
#include <iostream>


int main()
{
    auto a = nc::random::randInt<int>({ 780, 30 }, 0, 5);
    auto b = nc::zeros<double>({ 100,1 });
    Network n(784, 30, 10);
    auto data = nc::random::normal<double>({ 784,1 });
    std::cout << data;
    n.LoadData(data,0);
    n.front_proceed();
    n.printZout();
}