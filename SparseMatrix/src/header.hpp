/*
 * header.hpp
 *
 *  Created on: Dec 1, 2018
 *      Author: sunkg
 */

#ifndef SRC_HEADER_HPP_
#define SRC_HEADER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <Core/Assert.hpp>
#include <boost/lexical_cast.hpp>
#include <math.h>
#include <stdio.h>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/backend/opencl.hpp>
#include <viennacl/compressed_matrix.hpp>


using namespace cv;

viennacl::compressed_matrix<float> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor);
viennacl::compressed_matrix<float> Hmatrix(cv::Mat & Dest, const cv::Mat& kernel);
viennacl::compressed_matrix<float> Mmatrix(cv::Mat &Dest, float deltaX, float deltaY);
void motionMat(std::vector<Mat>& motionVec, size_t image_count, size_t rfactor, bool clockwise);
viennacl::compressed_matrix<float> ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel);
void Gaussiankernel(cv::Mat& dst);
void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex, std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor);

#define sign_float(a,b) (a>b)?1.0f:(a<b)?-1.0f:0.0f
#define max_float(a,b) (a>b)?a:b
#define ROUND_UINT(d) ( (unsigned int) ((d) + ((d) > 0 ? 0.5 : -0.5)) )


#endif /* SRC_HEADER_HPP_ */
