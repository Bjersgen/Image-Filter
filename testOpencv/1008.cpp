#include <opencv.hpp>  
#include<iostream>  
#include"time.h"  
#include<vector>  
#include<map> 
#include<math.h>

using namespace std;
using namespace cv;

void saltAndPepper(cv::Mat image, int n);
Mat addGaussianNoise(Mat& srcImag);
double generateGaussianNoise(double mu, double sigma);
static cv::Mat my_bilateral_filter(const cv::Mat& src, int size, double sigmaColor, double sigmaSpace);
double get_color_factor(int fij, int fkl, double sigmaColor);
double get_space_factor(int i, int j, int k, int l, double sigmaSpace);
void my_blur(cv::Mat& src, cv::Mat& dst);
void my_medianBlur(cv::Mat& src, cv::Mat& dst);

int main()
{

	Mat srcImage = imread("C://Users//60105//Desktop//opencv-3.4.9-vc14_vc15//hahaha.jpeg");
	Mat srcImage1 = srcImage;
	Mat rawImage;
	Mat src_gray, raw_gray;
	Mat blur1(srcImage.size(), srcImage.type()), blur2(srcImage.size(), srcImage.type()), blur3(srcImage.size(), srcImage.type()), blur4(srcImage.size(), srcImage.type()), blur5, blur6;
	imshow("原图", srcImage);
	saltAndPepper(srcImage, 10000);
	cvtColor(srcImage, src_gray, COLOR_BGR2GRAY);
	rawImage = addGaussianNoise(srcImage1);
	cvtColor(rawImage, raw_gray, COLOR_BGR2GRAY);
	imshow("高斯", rawImage);
	imshow("椒盐图", srcImage);	


	//medianBlur(rawImage,blur1, 5);
	//medianBlur(srcImage,blur2, 5);
	my_medianBlur(rawImage, blur1);
	my_medianBlur(srcImage, blur2);
	my_blur(rawImage, blur3);
	my_blur(srcImage, blur4);
	blur5 =  my_bilateral_filter(raw_gray, 7, 20.0, 20.0);
	blur6 =  my_bilateral_filter(src_gray, 7, 20.0, 20.0);
	imshow("中值+高斯", blur1);
	imshow("中值+椒盐", blur2);
	imshow("平均+高斯", blur3);
	imshow("平均+椒盐", blur4); 
	imshow("双边+高斯", blur5);
	imshow("双边+椒盐", blur6);

	int rows = srcImage.rows;
	int cols = srcImage.cols;
	waitKey(0);

	return 0;
}

void my_medianBlur(cv::Mat& src, cv::Mat& dst) {

	if (!src.data) return;
	//at访问像素点
	for (int i = 1; i < src.rows; ++i) {
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {//边缘不进行处理
				for (int h = 0; h < 3; h++) {

					int l = 0;
					int window[10] = {0};
					for (int p = i - 1; p <= i + 1; p++) {
						for (int k = j - 1; k <= j + 1; j++) {

							window[l++] = src.at<Vec3b>(p, k)[h];

						}
					}
					for (int m = 0; m < 5; ++m)
					{
						int min = m;
						for (int n = m + 1; n < 9; ++n) {
							if (window[n] < window[min])
								min = n;
						}
						//Put found minimum element in its place  
						int temp = window[m];
						window[m] = window[min];
						window[min] = temp;
					}
					dst.at<Vec3b>(i, j)[h] = window[4];
				}
			}
			else {//边缘赋值
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
	}

}



//均值滤波，默认K = 3
void my_blur(cv::Mat& src, cv::Mat& dst) {


	if (!src.data) return;
	//at访问像素点
	for (int i = 1; i < src.rows; ++i)
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {//边缘不进行处理
				dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
					src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
					src.at<Vec3b>(i + 1, j)[0]) / 9;
				dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
					src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
					src.at<Vec3b>(i + 1, j)[1]) / 9;
				dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
					src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
					src.at<Vec3b>(i + 1, j)[2]) / 9;
			}
			else {//边缘赋值
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}


}
//椒盐噪声
void saltAndPepper(cv::Mat image, int n)
{
	for (int k = 0; k < n / 2; k++)
	{
		// 随机确定图像中添加椒盐噪声的位置
		int i, j;
		i = std::rand() % image.cols;       // 取余数运算，保证在图像的列数内 
		j = std::rand() % image.rows;       // 取余数运算，保证在图像的行数内 
		int write_black = std::rand() % 2;  // 判定为白色噪声还是黑色噪声的变量
		// 添加白色噪声
		if (write_black == 0)
		{
			image.at<cv::Vec3b>(j, i)[0] = 255; //cv::Vec3b为opencv定义的一个3个值的向量类型  
			image.at<cv::Vec3b>(j, i)[1] = 255; //[]指定通道，B:0，G:1，R:2  
			image.at<cv::Vec3b>(j, i)[2] = 255;

		}
		// 添加黑色噪声
		else
		{
			image.at<cv::Vec3b>(j, i)[0] = 0; //cv::Vec3b为opencv定义的一个3个值的向量类型  
			image.at<cv::Vec3b>(j, i)[1] = 0; //[]指定通道，B:0，G:1，R:2  
			image.at<cv::Vec3b>(j, i)[2] = 0;
		}
	}

}

double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits <double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
	return z0 * sigma + mu;
}

//为图像添加高斯噪声
Mat addGaussianNoise(Mat& srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols * channels;
	//判断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//添加高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}

//计算空间当前点空间权重Wd(其实就是高斯核，高斯核与距离挂钩)
double get_space_factor(int i, int j, int k, int l, double sigmaSpace)
{
	double sp1 = std::pow(i * 1.0 - k, 2);
	double sp2 = std::pow(j * 1.0 - l, 2);
	double denom = 2.0 * sigmaSpace * sigmaSpace;

	return std::exp(-(sp1 + sp2) / denom);
}


//计算像素层面权重
double get_color_factor(int fij, int fkl, double sigmaColor)
{
	return std::exp(-std::pow(fij * 1.0 - fkl, 2) / (2.0 * std::pow(sigmaColor, 2)));
}
//双边滤波
static cv::Mat my_bilateral_filter(const cv::Mat& src, int size, double sigmaColor, double sigmaSpace)
{
	
    CV_Assert(src.type() == CV_8UC1);
	CV_Assert(size > 0 && size % 2 == 1);

	int ps = size / 2;
	cv::Mat matPadded;
	cv::copyMakeBorder(src, matPadded, ps, ps, ps, ps, cv::BORDER_REFLECT101);

	cv::Mat result(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			double numerator = 0;						//分子的和
			double denominator = 0;						//分母的和（权重总和）
			double fij = matPadded.ptr<uchar>(i + ps)[j + ps] * 1.0;

			for (int k = i - ps; k <= i + ps; ++k)
			{
				for (int l = j - ps; l <= j + ps; ++l)
				{
					double fkl = matPadded.ptr<uchar>(k + ps)[l + ps] * 1.0;
					double w_ijkl = get_color_factor(fij, fkl, sigmaColor) * get_space_factor(i, j, k, l, sigmaSpace);
					numerator += fkl * w_ijkl;
					denominator += w_ijkl;
				}
			}

			result.ptr<uchar>(i)[j] = cv::saturate_cast<uchar>(numerator / denominator);
		}
	}

	return result;
}
