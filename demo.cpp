#include "featuredetection.h"
#include "featurematch.h"
#include <opencv2/opencv.hpp>
#include <iostream> 

using namespace cv;
using namespace std;

#define USINGMATCH 1

Mat img1;
Mat img2;

Mat image1, image2;
vector<KeyPoint> keyPoint1, keyPoint2;
Mat result;
Mat result_kp;
int th;
int dis;

struct UserData
{
	FeatureDetection m_fd;
	FeatureMatch m_fm;
};

void on_trackbar(int, void *usrdata)
{
	double d = dis / 100.0;
	(*(UserData*)usrdata).m_fd.RunFeatureDetector(image1, image2, result, keyPoint1, keyPoint2, th);
	Mat res = result.clone();
	for (int i = 0; i <keyPoint1.size(); i++)
	{
		circle(res, keyPoint1[i].pt, 5, Scalar(255, 0, 0), 2);
	}
	for (int i = 0; i <keyPoint2.size(); i++)
	{
		circle(res, Point(keyPoint2[i].pt.x + img1.cols, keyPoint2[i].pt.y), 5, Scalar(255, 0, 0), 2);
	}
	if (USINGMATCH)
	{
		if ((keyPoint1.size() > 0 && keyPoint2.size() > 0))
		{
			vector<DMatch> GoodMatchePoints;
			std::vector<cv::Point2f> pts1;
			std::vector<cv::Point2f> pts2;
			(*(UserData*)usrdata).m_fm.RunMatch(image1, image2, keyPoint1, keyPoint2, GoodMatchePoints, d);
			for (int i = 0; i < GoodMatchePoints.size(); i++)
			{
				pts1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
				pts2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
				line(res, keyPoint1[GoodMatchePoints[i].trainIdx].pt, Point(keyPoint2[GoodMatchePoints[i].queryIdx].pt.x + img1.cols, keyPoint2[GoodMatchePoints[i].queryIdx].pt.y), Scalar(0, 255, 0), 2);
			}
		}
	}
	result_kp = res.clone();
	imshow("result", res);

}

int main()
{
	img1 = imread("../data/aloeL.jpg", 1);
	img2 = imread("../data/aloeR.jpg", 1);

	if (img1.empty() || img2.empty())
	{
		cerr << "input image err!";
		return -1;
	}

	cvtColor(img1, image1, CV_RGB2GRAY);
	cvtColor(img2, image2, CV_RGB2GRAY);

	Mat resulttmp(img1.rows, img1.cols * 2, CV_8UC3);
	resulttmp.copyTo(result);
	img1.copyTo(result.colRange(0, img1.cols));
	img2.copyTo(result.colRange(img1.cols, img1.cols * 2));

	FeatureDetection featdetct;
	string featmod = "fast";//选择需要的特征算子
	th = 50;
	featdetct.SetFeatureDetector(featmod);

	FeatureMatch featmatch;
	string matmod = "flann";//选择需要的匹配方法
	featmatch.SetMatchMode(featmod, matmod);

	UserData mydata;
	mydata.m_fd = featdetct;
	mydata.m_fm = featmatch;

	namedWindow("result", WINDOW_NORMAL);
	cv::createTrackbar("th : ", "result", &th, 100, on_trackbar, &mydata);
	on_trackbar(th, &mydata);
	dis = 50;//0-100
	cv::createTrackbar("dis : ", "result", &dis, 100, on_trackbar, &mydata);
	on_trackbar(dis, &mydata);

	cv::waitKey(0);
	//当参数合适时，按任意键保存
	imwrite("result.jpg", result_kp);//保存结果
	return 0;
}