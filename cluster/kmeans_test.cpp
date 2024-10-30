/*
* KMeans.h
* Copyright 2024 (C) by Jon Ander Gómez
*/

#include <kmeans.h>
#include <random>
#include <iostream>


int main(int argc, char * argv[])
{
    KMeans kmeans(10, 3);

    // Seed with a real random value, if available
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> urd_1(1.0, 2.0);
    std::uniform_real_distribution<> urd_2(2.0, 8.0);
    std::uniform_real_distribution<> urd_3(9.0, 5.0);

    std::vector<double> x;
    x.push_back(urd_1(gen));
    x.push_back(urd_2(gen));
    x.push_back(urd_3(gen));
    kmeans.add(x);

    for (int i = 1; i < 100000; i++)
    {
        x[0] = urd_3(gen); x[1] = urd_2(gen); x[2] = urd_1(gen); kmeans.add(x);
        x[0] = urd_1(gen); x[1] = urd_3(gen); x[2] = urd_2(gen); kmeans.add(x);
        x[0] = urd_1(gen); x[1] = urd_2(gen); x[2] = urd_3(gen); kmeans.add(x);
    }

    std::string filename = "k10.txt";
    kmeans.save(filename);

    KMeans k2(filename);
    filename = "k11.txt";
    k2.save(filename);

    return EXIT_SUCCESS;
}
