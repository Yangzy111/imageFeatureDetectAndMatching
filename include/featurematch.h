#ifndef FEARTUREMATCH_H_
#define FEARTUREMATCH_H_

#include "opencv2/nonfree/nonfree.hpp" 
 
class FeatureMatch
{
public:
	FeatureMatch();
	~FeatureMatch();
	void SetMatchMode(const std::string& detmode, const std::string& matchmod);
	void RunMatch(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& goodpoints, double dis);
private:
	void SiftDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2);
	void SurfDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2);
	void OrbDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2);
	void FastDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2);
	void HarrisDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2);
	
	void FlannMatch(cv::Mat& imdec1, cv::Mat& imdec2, std::vector<cv::DMatch>& goodpoints, double dis);
	void BFMatch(cv::Mat& imdec1, cv::Mat& imdec2, std::vector<cv::DMatch>& goodpoints, double dis);
private:
	std::string m_detector;
	std::string m_matcher;
};

#endif