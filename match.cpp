// line_match.cpp : 定义控制台应用程序的入口点。
//
#include <iostream>
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_FEATURES2D

#include <map>
#include <vector>
#include <list>
#include <inttypes.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include "opencv2/line_descriptor/descriptor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Geometry>

#define MATCHES_DIST_THRESHOLD 25

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

static const char* keys = { "{@image_path1 | | Image path 1 }" "{@image_path2 | | Image path 2 }" };

static void help()
{
    std::cout << "\nThis example shows the functionalities of lines extraction "
              << "and descriptors computation furnished by BinaryDescriptor class\n"
              << "Please, run this sample using a command in the form\n"
              << "./example_line_descriptor_compute_descriptors <path_to_input_image 1>"
              << "<path_to_input_image 2>"
              << std::endl;
}

void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
                            std::vector<KeyPoint> keypoints_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
    double focal_length = 521;			//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.R和t是从1到2
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;

}

int main(int argc, char** argv)
{

    String image_path1 = "../1.png";
    String image_path2 = "../2.png";

    if (image_path1.empty() || image_path2.empty())
    {
        help();
        return -1;
    }

    /* 加载图片 */
    cv::Mat imageMat1 = imread(image_path1, 1);
    cv::Mat imageMat2 = imread(image_path2, 1);

    if (imageMat1.data == NULL || imageMat2.data == NULL)
    {
        std::cout << "Error, images could not be loaded. Please, check their path" << std::endl;
    }

    resize(imageMat1,imageMat1,Size(780,680));
    resize(imageMat2,imageMat2,Size(780,680));
    /* create binary masks */
    cv::Mat mask1 = Mat::ones(imageMat1.size(), CV_8UC1);
    cv::Mat mask2 = Mat::ones(imageMat2.size(), CV_8UC1);

    /* BinaryDescriptor指针默认参数 */
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    //
    std::vector<KeyPoint> kps1,kps2;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    detector->detect(imageMat1,kps1);

    std::vector<KeyPoint> point1,point2;

    Mat outImage1,outImage2,outIm1,outIm2;
   // drawKeypoints(imageMat1,kps1,imageMat1);

    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    std::vector<DMatch> matches;

    std::vector<DMatch> good_matches;

    /* 画出匹配对 */
    cv::Mat outImg;
    cv::Mat scaled1, scaled2;
    std::vector<char> mask(matches.size(), 1);

    /* LSD 检测 */
    Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();

    /* 检测线段 */
    std::vector<KeyLine> klsd1, klsd2;
    Mat lsd_descr1, lsd_descr2;

    // lsd->detect(image, klsd, scale, numOcatives, mask)   numOcatives = 2
    lsd->detect(imageMat1, klsd1, 2, 2, mask1);
    lsd->detect(imageMat2, klsd2, 2, 2, mask2);

    /* 计算尺度第一塔的描述 */
    bd->compute(imageMat1, klsd1, lsd_descr1);
    bd->compute(imageMat2, klsd2, lsd_descr2);

    /* 尺度第一塔进行特征与描述提取 */
    std::vector<KeyLine> octave0_1, octave0_2;
    Mat leftDEscr, rightDescr;
    point1.clear();
    point2.clear();
    for (int i = 0; i < (int)klsd1.size(); i++)
    {
        if (klsd1[i].octave == 1)
        {
            octave0_1.push_back(klsd1[i]);
            KeyPoint keypoint(klsd1[i].pt,5);
            point1.push_back(keypoint);
            leftDEscr.push_back(lsd_descr1.row(i));
        }
    }

    for (int j = 0; j < (int)klsd2.size(); j++)
    {
        if (klsd2[j].octave == 1)
        {
            octave0_2.push_back(klsd2[j]);
            KeyPoint keypoint(klsd2[j].pt,5);
            point2.push_back(keypoint);
            rightDescr.push_back(lsd_descr2.row(j));
        }
    }

    drawKeypoints(imageMat1,point1,outImage1);
    drawKeypoints(imageMat2,point2,outImage2);

    drawKeylines(outImage1,octave0_1,outIm1);
    drawKeylines(outImage2,octave0_2,outIm2);
    imshow("test",outIm1);
    /* 匹配点对 */
    std::vector<DMatch> lsd_matches;
    bdm->match(leftDEscr, rightDescr, lsd_matches);

    /* 选择高精度匹配点对 */
    good_matches.clear();
    for (int i = 0; i < (int)lsd_matches.size(); i++)
    {
        if (lsd_matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(lsd_matches[i]);
    }

    //提供初始位姿
    Mat R,t;
    pose_estimation_2d2d(point1,point2,good_matches,R,t);

    //光流跟踪优化


//
//    /* 画出匹配点对 */
//    cv::Mat lsd_outImg;
////    resize(imageMat1, imageMat1, Size(imageMat1.cols/2, imageMat1.rows/2));
////    resize(imageMat2, imageMat2, Size(imageMat2.cols/2, imageMat2.rows/2));
//    std::vector<char> lsd_mask(matches.size(), 1);
////    drawLineMatches(imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg,
////                    Scalar::all(-1), Scalar::all(-1), lsd_mask, DrawLinesMatchesFlags::DEFAULT);
//    drawMatches(outIm1,point1,outIm2,point2,good_matches,lsd_outImg);
//    imshow("LSD matches", lsd_outImg);
//    std::cout << "LSDescriptorMatcher is : " << good_matches.size() << std::endl;
//    imwrite("..\\line_match\\image\\matches.jpg", outImg);
//    imwrite("..\\line_match\\image\\lsd_matches.jpg", lsd_outImg);

    waitKey(0);
    return 0;
}

#else

int main()
{
	std::cerr << "OpenCV was built without features2d module" << std::endl;
	return 0;
}

#endif // HAVE_OPENCV_FEATURES2D
