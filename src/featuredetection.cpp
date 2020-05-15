#include "featuredetection.h"
#include <string>

#include <chrono>
using namespace cv;
using namespace std;

FeatureDetection::FeatureDetection()
{
}
FeatureDetection::~FeatureDetection()
{
}

void FeatureDetection::SetFeatureDetector(const string &detector)
{
	m_detector = detector;
}
void FeatureDetection::RunFeatureDetector(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int th)
{
	auto start = chrono::system_clock::now();
	if (m_detector == "sift")
		Sift(img1, img2, result, kp1, kp2, th);
	else if (m_detector == "surf")
		Surf(img1, img2, result, kp1, kp2, th);
	else if (m_detector == "orb")
		Orb(img1, img2, result, kp1, kp2, th);
	else if (m_detector == "fast")
		Fast(img1, img2, result, kp1, kp2, th);
	else if (m_detector == "harris")
		Harris(img1, img2, result, kp1, kp2, th);
	else
		return;
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "DetectSpent: " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " seconds." << endl;
}

void FeatureDetection::Sift(cv::Mat& img1, cv::Mat& img2, Mat& result, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, int th)
{
	SiftFeatureDetector siftDetector(th, 3, 0.04, 10.0, 1.6);// ����������ֵ��������������ȣ�ֵԽ���Խ�٣�Խ��׼ ,������ֵҲ����ȡ���������

	siftDetector.detect(img1, kp1);
	siftDetector.detect(img2, kp2);
}

void FeatureDetection::Surf(cv::Mat& img1, cv::Mat& img2, Mat& result, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, int th)
{
	SurfFeatureDetector surfDetector(th, 4, 2);  // ����������ֵ��������������ȣ�ֵԽ���Խ�٣�Խ��׼ 

	surfDetector.detect(img1, kp1);
	surfDetector.detect(img2, kp2);
}

void FeatureDetection::Orb(cv::Mat& img1, cv::Mat& img2, Mat& result, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, int th)
{
	OrbFeatureDetector OrbDetector(th, 1.2, 8, 32, 0, 2, 1, 31);  //thΪ����������  ��Ϊ��fastscore

	OrbDetector.detect(img1, kp1);
	OrbDetector.detect(img2, kp2);
}

void FeatureDetection::Fast(cv::Mat& img1, cv::Mat& img2, Mat& result, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, int th)
{
	FastFeatureDetector Detector(th);  //��ֵ 

	Detector.detect(img1, kp1);
	Detector.detect(img2, kp2);
}

void FeatureDetection::Harris(cv::Mat& img1, cv::Mat& img2, Mat& result, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, int th)
{
	GoodFeaturesToTrackDetector Detector(th,0.01,1.0,3,true);  //����������

	Detector.detect(img1, kp1);
	Detector.detect(img2, kp2);
}