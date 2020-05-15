#ifndef FEATUREDETECTION_H_
#define FEATUREDETECTION_H_

#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/calib3d/calib3d.hpp"

class FeatureDetection
{
public:
	FeatureDetection();
	~FeatureDetection();

	void SetFeatureDetector(const std::string &detector);
	void RunFeatureDetector(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int th);

private:
	void Sift(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int H_th);
	void Surf(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int H_th);
	void Orb(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int F_num);
	void Fast(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int th);
	void Harris(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, int th);

private:
	std::string m_detector;

};
#endif