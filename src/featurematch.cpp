#include "featurematch.h" 
#include <iostream> 
#include <chrono>
using namespace std;
FeatureMatch::FeatureMatch(){}
FeatureMatch::~FeatureMatch(){}

void FeatureMatch::SetMatchMode(const std::string& detmode,const std::string& matchmod)
{
	m_detector = detmode;
	m_matcher = matchmod;
}

void FeatureMatch::RunMatch(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& goodpoints, double dis)
{
	auto start = chrono::system_clock::now();
	if (!(kp1.size() > 0 && kp2.size() > 0))
		return;
	cv::Mat imdec1, imdec2;
	if (m_detector == "sift")
		SiftDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
	else if (m_detector == "surf")
		SurfDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
	else if (m_detector == "orb")
		OrbDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
	else if (m_detector == "fast")
		FastDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
	else if (m_detector == "harris")
		HarrisDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
	else
		return;
	
	if (m_detector == "orb" && m_matcher == "flann")//特殊处理，orb由于是brief描述，只能用汉明距 knn
	{
		cv::flann::Index flannIndex(imdec1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
		cv::Mat macthIndex(imdec2.rows, 2, CV_32SC1), matchDistance(imdec2.rows, 2, CV_32FC1);
		flannIndex.knnSearch(imdec2, macthIndex, matchDistance, 2, cv::flann::SearchParams());

		// Lowe's algorithm,获取优秀匹配点
		for (int i = 0; i < matchDistance.rows; i++)
		{
			if (matchDistance.at<float>(i, 0) < dis * matchDistance.at<float>(i, 1))
			{
				cv::DMatch matches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
				goodpoints.push_back(matches);
			}
		}
	}
	else if (m_matcher == "flann")
		FlannMatch(imdec1, imdec2, goodpoints, dis);
	else
		BFMatch(imdec1, imdec2, goodpoints, dis);
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "MatchSpent: " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " seconds." << endl;
}

void FeatureMatch::SiftDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2)
{
	cv::SiftDescriptorExtractor SiftDescriptor;
	SiftDescriptor.compute(i1, kp1, imdec1);
	SiftDescriptor.compute(i2, kp2, imdec2);
}
void FeatureMatch::SurfDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2)
{
	cv::SurfDescriptorExtractor SurfDescriptor;
	SurfDescriptor.compute(i1, kp1, imdec1);
	SurfDescriptor.compute(i2, kp2, imdec2);
}
void FeatureMatch::OrbDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2)
{
	cv::OrbDescriptorExtractor OrbDescriptor;
	OrbDescriptor.compute(i1, kp1, imdec1);
	OrbDescriptor.compute(i2, kp2, imdec2);
}
void FeatureMatch::FastDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2)
{
	SiftDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
}
void FeatureMatch::HarrisDescriptor(cv::Mat& i1, cv::Mat& i2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& imdec1, cv::Mat& imdec2)
{
	SiftDescriptor(i1, i2, kp1, kp2, imdec1, imdec2);
}

void FeatureMatch::FlannMatch(cv::Mat& imdec1, cv::Mat& imdec2, std::vector<cv::DMatch>& goodpoints, double dis)
{
	std::cout << "using FlannMatch" << std::endl;
	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch> > matches;


	std::vector<cv::Mat> train_desc(1, imdec1);
	matcher.add(train_desc);
	matcher.train();

	matcher.knnMatch(imdec2, matches, 2);

	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < dis * matches[i][1].distance)
		{
			goodpoints.push_back(matches[i][0]);
		}
	}
}
void FeatureMatch::BFMatch(cv::Mat& imdec1, cv::Mat& imdec2, std::vector<cv::DMatch>& goodpoints, double dis)
{
	std::cout << "using BFMatch" << std::endl;
	std::vector<cv::DMatch> matches;
	cv::BFMatcher bfMatcher(cv::NORM_HAMMING);
	bfMatcher.match(imdec1, imdec2, matches);

	double min_dist = 1000, max_dist = 0;
	// 找出所有匹配之间的最大值和最小值
	for (int i = 0; i < imdec1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	// 当描述子之间的匹配不大于2倍的最小距离时，即认为该匹配是一个错误的匹配。
	// 但有时描述子之间的最小距离非常小，可以设置一个经验值作为下限
	for (int i = 0; i < imdec1.rows; i++)
	{
		if (matches[i].distance <= std::max(2 * min_dist, dis))
			goodpoints.push_back(matches[i]);
	}
}