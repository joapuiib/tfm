#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

int main() {
    std::string imagePath;

    // Create K-means codebook

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
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Optional: Print pixel values (can be commented out for performance)
                std::cout << "Pixel at (" << row << "," << col << "): "
                          << "R=" << red << ", G=" << green << ", B=" << blue << std::endl;

                // Add pixel to the codebook
            }
        }
    }

    // Save the codebook to a file
    return 0;
}
