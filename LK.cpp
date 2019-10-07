//
// Created by luoyongheng on 19-10-6.
//
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>

//#define harris

using namespace std;
using namespace cv;

int ROW = 480;
int COL = 640;
int MAX_CNT = 150;
int MIN_DIST = 30;
int iniThFAST = 20;
int minThFAST = 7;


bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


int main ( int argc, char** argv ) {
    //相机参数
    float cx = 318.6;
    float cy = 255.3;
    float fx = 517.3;
    float fy = 516.5;

    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;

    //-- 读取图像
    if (argc != 2) {
        cout << "usage: useLK path_to_dataset" << endl;
        return 1;
    }
    //srand ( ( unsigned int ) time ( 0 ) );
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin(associate_file);

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat forw_img, cur_img;
    vector<KeyPoint> keypoints;;
    vector<Point2f> forw_pts, cur_pts;
    vector<cv::Point2f> n_pts;
    vector<int> track_cnt;
    vector<int> ids;
    Mat imgPreColor;

    //保存轨迹
    //ofstream out("../../data/data/trajectory.txt", ofstream::out);

    // 我们以第一个图像为参考，对后续10张图像和参考图像做直接法
    for (int index = 0; index < 20; index++) {
        cout << "*********** loop " << index << " ************" << endl;
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        cout << time_rgb << " " << rgb_file << " " << time_depth << " " << depth_file << endl;
        Mat imgColor = cv::imread(path_to_dataset + "/" + rgb_file,1);

        Mat img;
        cvtColor(imgColor,img,COLOR_BGR2GRAY);

        //直方图均衡化
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);

        if (forw_img.empty()) {
            cur_img = forw_img = img;
            imgPreColor = imgColor;
        } else {
            forw_img = img;
        }

        forw_pts.clear();

        if (cur_pts.size() > 0) {
            //TicToc t_o;
            vector<uchar> status;
            vector<float> err;
            cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

            for (int i = 0; i < int(forw_pts.size()); i++)
                if (status[i] && !inBorder(forw_pts[i]))
                    status[i] = 0;
            //reduceVector(prev_pts, status);
            reduceVector(cur_pts, status);
            reduceVector(forw_pts, status);
            reduceVector(ids, status);
            //reduceVector(cur_un_pts, status);
            reduceVector(track_cnt, status);
            //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        }

        for (auto &n : track_cnt)
            n++;
        //对提取出来的特征点求带Ransanc的基础矩阵，从而进一步筛选掉不匹配的点;
        if (forw_pts.size() >= 8) {
            vector<uchar> status;
            cv::findFundamentalMat(cur_pts, forw_pts, cv::FM_RANSAC, 1.0, 0.99, status);
            int size_a = cur_pts.size();
            //reduceVector(prev_pts, status);
            reduceVector(cur_pts, status);
            reduceVector(forw_pts, status);
            //reduceVector(cur_un_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
        }

        cv::Mat img_window(forw_img.rows*2, forw_img.cols, CV_8UC3 );//8位unsigned 3通道
        imgPreColor.copyTo(img_window(cv::Rect(0,0,cur_img.cols,cur_img.rows)));
        imgColor.copyTo(img_window(cv::Rect(0,forw_img.rows,forw_img.cols, forw_img.rows)));

        for(int i=0;i<cur_pts.size();i++){
//            if (rand() > RAND_MAX/5 )
//                continue;
            float b = 255*float ( rand() ) /RAND_MAX;//用随机颜色框选特征点
            float g = 255*float ( rand() ) /RAND_MAX;
            float r = 255*float ( rand() ) /RAND_MAX;
            cv::circle(img_window,cur_pts[i],4,cv::Scalar( b,g,r ),2);
            cv::circle(img_window,Point2f(forw_pts[i].x ,forw_pts[i].y + forw_img.rows),4,cv::Scalar( b,g,r), 2);
            cv::line(img_window,cur_pts[i],Point2f(forw_pts[i].x ,forw_pts[i].y + forw_img.rows),cv::Scalar( b,g,r), 1);
        }
        cv::imshow ("result", img_window);
        cv::waitKey ( 0 );

        //为下一次特征提取做准备
#ifdef harris
        Mat mask;
        //设置mask用于非极大值抑制
        {
            mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

            vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

            for (unsigned int i = 0; i < forw_pts.size(); i++)
                cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

            sort(cnt_pts_id.begin(), cnt_pts_id.end(),
                 [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
                     return a.first > b.first;
                 });

            forw_pts.clear();
            ids.clear();
            track_cnt.clear();

            for (auto &it : cnt_pts_id) {
                if (mask.at<uchar>(it.second.first) == 255) {
                    forw_pts.push_back(it.second.first);
                    ids.push_back(it.second.second);
                    track_cnt.push_back(it.first);
                    cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
                }
            }
        }

        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0) {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            //优先从之前跟踪到的特征点中提取角点,虽然这些特征点被非极大值抑制掉了，认为没有成功跟踪，但是可以优先作为新的特征点被提取
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        } else
            n_pts.clear();
#else
            vector<KeyPoint> kp;
            const float W=30;
            //fast特征点提取

            const int nDesiredFeatures = MAX_CNT - forw_pts.size();
            if(nDesiredFeatures>0)
            {
                const int nCols = int(COL/W);//21
                const int nRows = int(ROW/W);//16
                const int nCells = nCols*nRows;

                const int nFeatureEachCell = ceil(float(nDesiredFeatures)/nCells);
                vector<vector<vector<KeyPoint> > > cellKeyPoints(nRows, vector<vector<KeyPoint> >(nCols));

                //计算每个窗口的大小
                const int wCell = COL/nCols;//30
                const int hCell = ROW/nRows;//30

                vector<vector<int>> nTotal(nRows,vector<int>(nCols));
                vector<vector<bool>> bNoMore(nRows,vector<bool>(nCols,false));
                vector<vector<int>> nToRetain(nRows,vector<int>(nCols));
                int nNoMore = 0;
                int nToDistribute = 0;


                for(int i=0; i<nRows; i++)
                {
                    //每个窗口纵向的范围
                    const int iniY =i*hCell;
                    int maxY = iniY+hCell;
                    //出了图片的有效区域
//                if(iniY>=maxBorderY-3)
//                    continue;
                    //超出了边界的话就使用图像的边界作为边界
//                if(maxY>maxBorderY)
//                    maxY = maxBorderY;

                    for(int j=0; j<nCols; j++)
                    {
                        //计算每列的位置
                        //每个窗口横向的范围
                        const int iniX =j*wCell;
                        int maxX = iniX+wCell;
                        //出了横向范围
//                    if(iniX>=maxBorderX-6)
//                        continue;
                        //超了横向范围，就使用图像的边界作为窗口的边界
//                    if(maxX>maxBorderX)
//                        maxX = maxBorderX;
                        //计算FAST关键点
                        //vector<cv::KeyPoint> vKeysCell;
                        //对每一个窗口都计算FAST角点
                        cellKeyPoints[i][j].reserve(nFeatureEachCell*5);
                        FAST(forw_img.rowRange(iniY,maxY).colRange(iniX,maxX),
                             cellKeyPoints[i][j],iniThFAST,true);

                        if(cellKeyPoints[i][j].size()<=3)
                        {
                            cellKeyPoints[i][j].clear();
                            FAST(forw_img.rowRange(iniY,maxY).colRange(iniX,maxX),cellKeyPoints[i][j],7,true);
                        }

                        //HarrisResponses(cellImage,cellKeyPoints[i][j], 7, HARRIS_K);
                        const int nKeys = cellKeyPoints[i][j].size();
                        nTotal[i][j] = nKeys;

                        if(nKeys>nFeatureEachCell)
                        {
                            nToRetain[i][j] = nFeatureEachCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nKeys;
                            nToDistribute += nFeatureEachCell-nKeys;
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }

                while(nToDistribute>0 && nNoMore<nCells)
                {
                    int nNewFeaturesCell = nFeatureEachCell + ceil((float)nToDistribute/(nCells-nNoMore));
                    nToDistribute = 0;

                    for(int i=0; i< nRows; i++)
                    {
                        for(int j=0; j<nCols; j++)
                        {
                            if(!bNoMore[i][j])
                            {
                                if(nTotal[i][j]>nNewFeaturesCell)
                                {
                                    nToRetain[i][j] = nNewFeaturesCell;
                                    bNoMore[i][j] = false;
                                }
                                else
                                {
                                    nToRetain[i][j] = nTotal[i][j];
                                    nToDistribute += nNewFeaturesCell-nTotal[i][j];
                                    bNoMore[i][j] = true;
                                    nNoMore++;
                                }
                            }
                        }
                    }
                }

                keypoints.reserve(nDesiredFeatures*2);

                for(int i=0; i<nRows; i++)
                {
                    for(int j=0; j<nCols; j++)
                    {
                        vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                        KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                        if((int)keysCell.size()>nToRetain[i][j])
                            keysCell.resize(nToRetain[i][j]);

                        for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                        {
                            keysCell[k].pt.x+=j*wCell;
                            keysCell[k].pt.y+=i*hCell;
                            keysCell[k].size = 15;
                            keypoints.push_back(keysCell[k]);
                        }
                    }
                }

                if((int)keypoints.size()>nDesiredFeatures)
                {
                    KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
                    keypoints.resize(nDesiredFeatures);
                }
            } else
                n_pts.clear();
#endif

#ifdef harris
        for (auto &p : n_pts) {
            forw_pts.push_back(p);
            ids.push_back(-1);
            track_cnt.push_back(1);
        }
#else
        for (auto &p : keypoints) {
            forw_pts.push_back(p.pt);
            ids.push_back(-1);
            track_cnt.push_back(1);
        }
#endif
        //waitKey(0);
        //prev_img = cur_img;
        //prev_pts = cur_pts;
        //prev_un_pts = cur_un_pts;
        cur_img = forw_img;
        cur_pts = forw_pts;
        imgPreColor = imgColor.clone();
        keypoints.clear();
        //undistortedPoints();
        //prev_time = cur_time;
    }
    return 0;
}