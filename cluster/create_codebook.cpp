#include <iostream>
#include "kmeans.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

int main() {
    std::string imagePath;

    KMeans kmeans(8, 3);

    while (std::cin >> imagePath) {
        // Load image in BGR format
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Could not load image at " << imagePath << std::endl;
            continue;
        }

        std::cout << "Loaded image: " << imagePath << " with size " 
                  << image.rows << "x" << image.cols << std::endl;

        // Traverse each pixel in the image
        for (int row = 0; row < image.rows; ++row) {
            for (int col = 0; col < image.cols; ++col) {
                // Access the BGR pixel values
                cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
                double blue  = pixel[0] / 255.0;
                double green = pixel[1] / 255.0;
                double red   = pixel[2] / 255.0;
                std::vector<double> x = {red, green, blue};
                kmeans.add(x);
            }
        }
    }

    // Save the codebook to a file
    std::string filename = "codebook.txt";
    kmeans.save(filename);

    return EXIT_SUCCESS;
}
