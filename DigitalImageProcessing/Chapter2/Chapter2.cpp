#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

#define IN_WINDOW "入"
#define OUT_WINDOW "出"

int main(int argc, char* argv[]) {
	cv::Mat in_image = cv::imread("testpic.jpg", cv::IMREAD_COLOR);
	cv::Mat out_image();

	cv::namedWindow(IN_WINDOW);
	cv::namedWindow(OUT_WINDOW);

	if (in_image.empty()) {
		cout << "error!";
	} else {
		cv::imshow(IN_WINDOW, in_image);
	}

	cv::waitKey(0);
	return 0;
}
