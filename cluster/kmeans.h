/*
* KMeans.h
* Copyright 2024 (C) by Jon Ander Gómez
*/

#include <vector>
#include <string>
#include <fstream>


class KMeans 
{
private:
    int         num_clusters;
    int         dim;

    std::vector<std::vector<double>>    codebook;
    std::vector<size_t>                 counters;
    size_t                              global_counter;

    void reset()
    {
        codebook.clear();
        counters.clear();
        global_counter = 0;
    }

    double squared_distance(std::vector<double> & a, std::vector<double> & b)
    {
        double sd = 0.0;

        for (size_t i = 0; i < a.size(); i++)
        {
            double d = a[i] - b[i];
            sd += d * d;
        }
        return sd;
    }

    int closest_cluster(std::vector<double> & a)
    {
        int k = 0;
        double min_sd = squared_distance(a, codebook[0]);

        for (size_t c = 1; c < codebook.size(); c++)
        {
            double sd = squared_distance(a, codebook[c]);
            if (sd < min_sd)
            {
                min_sd = sd;
                k = c;
            }
        }

        return k;
    }

    void update_cluster(int k, std::vector<double> & a)
    {
        double alpha = 1.0 / counters[k];

        for (size_t i = 0; i < a.size(); i++)
            codebook[k][i] = (1.0 - alpha) * codebook[k][i] + alpha * a[i];

        counters[k]++;
    }

public:

    KMeans(int num_clusters, int dim)
        : num_clusters(num_clusters), dim(dim)
    {
        global_counter = 0;
    }
    KMeans(const std::string & filename)
    {
        load(filename);
    }

    ~KMeans()
    {
        reset();
    }

    int get_num_clusters() { return num_clusters; }

    int classify(std::vector<double> & sample)
    {
        return closest_cluster(sample);
    }

    void add(std::vector<double> & sample)
    {
        if (global_counter < num_clusters)
        {
            counters.push_back(1);
            codebook.push_back(sample);
        }
        else
        {
            int k = closest_cluster(sample);
            update_cluster(k, sample);
        }
        ++global_counter;
    }

    void load(const std::string & filename)
    {
        reset();

        std::ifstream f(filename, std::ios_base::in);

        f >> num_clusters >> dim;

        for (int i = 0; i < num_clusters; i++)
        {
            size_t s;
            f >> s;
            counters.push_back(s);

            std::vector<double> m(dim);
            for (int j = 0; j < dim; j++)
            {
                f >> m[j];
            }
            codebook.push_back(m);
        }
        f.close();
    }

    void save(const std::string & filename)
    {
        std::ofstream f(filename, std::ios_base::out);

        f << num_clusters << " " << dim << std::endl;

        for (int i = 0; i < num_clusters; i++)
        {
            f << counters[i];
            for (int j = 0; j < dim; j++)
            {
                f << " " << codebook[i][j];
            }
            f << std::endl;
        }

        f.close();
    }
};
