#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <boost/format.hpp>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
//#include<opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  /*
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  cout<<argv[0]<<endl;
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  */
 string trajectory_file="/home/wei/dataset/poses/07.txt";
 ifstream fin(trajectory_file);
 double R00,R01,R02,R03,R10,R11,R12,R13,R20,R21,R22,R23;
    fin>> R00>>R01>>R02>>R03>>R10>>R11>>R12>>R13>>R20>>R21>>R22>>R23;
  string dataset_path_=argv[1];
  Mat traj(621,621,CV_8UC3);
  while(!fin.eof()){
  fin>> R00>>R01>>R02>>R03>>R10>>R11>>R12>>R13>>R20>>R21>>R22>>R23;

 //int x=-t.at<double>(0,0)+300;
 //int y= t.at<double>(2,0)+500;
int x0=R03+300;
int y0=-R23+250;



 cv::circle(traj,Point(x0,y0),1,Scalar(255,255,255),-1);
  }
   imshow("sequence 07",traj);
   cv::waitKey(0);
   unsigned long current_image_index_=1;
  boost::format fmt("%s/image_%d/%06d.png");
  cv::Mat img_1, img_2;
  cv::Mat img_last;
  img_last=cv::imread((fmt % dataset_path_ % 0 % 0).str(),cv::IMREAD_GRAYSCALE);
  //cv::resize(img_last, img_last, cv::Size(), 0.5, 0.5,cv::INTER_NEAREST);

 //Sophus::SE3d T_cur;
Eigen::Matrix3d R = Eigen::AngleAxisd(4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
 Eigen::Vector3d t(1, 1, 1);           // 沿X轴平移1
  Sophus::SE3d T_cur;//(R, t);           // 从R,t构造SE(3)
 Sophus::SE3d Last_ref_from_cur;
 



               //chushihua
  std::vector<KeyPoint> keypoints_last;
  Mat descriptors_last;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");


 detector->detect(img_last, keypoints_last);
 descriptor->compute(img_last, keypoints_last, descriptors_last);

    
	for (;; current_image_index_++)
	{
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

    // read images
    img_1 =
        cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    img_2 =
        cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    if (img_1.data == nullptr || img_2.data == nullptr) {
       // LOG(WARNING) << "cannot find images at index " << current_image_index_;
       cout<<"Finished!"<<endl;
        return 0;
    }
    //cv::resize(img_1, img_1, cv::Size(), 0.5, 0.5,
           //  cv::INTER_NEAREST);
   // cv::resize(img_2, img_2, cv::Size(), 0.5, 0.5,
            //   cv::INTER_NEAREST);
    
  cout<<"img_1.rows"<<img_1.rows<<endl;

 

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;
  
   // detector->detect(img_last, keypoints_last);

 // descriptor->compute(img_last, keypoints_last, descriptors_last);

  //Mat outimg1;
  //drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
 // imshow("ORB features", outimg1);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  vector<DMatch> matches_last;
  t1 = chrono::steady_clock::now();

  matcher->match(descriptors_1, descriptors_2, matches);
  
  //-- qianhoupipei
  
  matcher->match(descriptors_1, descriptors_last, matches_last);
  
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;





  
t1 = chrono::steady_clock::now();
  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  //printf("-- Max dist : %f \n", max_dist);
  //printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches1;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches1.push_back(matches[i]);
    }
  }
  matches=good_matches1;
  

  /////////////////////////////////////////////
   min_max = minmax_element(matches_last.begin(), matches_last.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
   min_dist = min_max.first->distance;
   max_dist = min_max.second->distance;

  //printf("-- Max dist : %f \n", max_dist);
  //printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches2;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches_last[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches2.push_back(matches_last[i]);
    }
  }
  matches_last=good_matches2;


//Mat img_lastMCur;

  //drawMatches(img_1, keypoints_1, img_last, keypoints_last, good_matches2, img_lastMCur);
  //imshow("img_lastMCur",img_lastMCur);
  vector<Point3f> kp_3d;
  vector<Point2f> kp_2d;
 // cout<<"matches_last:"<<matches_last.size()<<"matches"<<matches.size()<<endl;
  for (auto match_i:matches){
    for(auto match_j:matches_last){
      if(match_i.queryIdx==match_j.queryIdx){
        float diff=keypoints_1[match_i.queryIdx].pt.x-keypoints_2[match_i.trainIdx].pt.x;
        
        if(diff>0.5 && diff<150)
        {
          float baseline=0.537;
          float  focal=721.5377;
        float b_by_d = baseline / diff;
        float Z = focal * b_by_d;
        float X = (keypoints_1[match_i.queryIdx].pt.x-609.5593) * b_by_d;
        float Y = (keypoints_1[match_i.queryIdx].pt.y-172.854) * b_by_d;
        kp_3d.push_back(Point3f(X,Y,Z));
        //cout<<"x"<<X<<"Y"<<Y<<"Z"<<Z<<endl;
        kp_2d.push_back(Point2f(keypoints_last[match_j.trainIdx].pt.x,keypoints_last[match_j.trainIdx].pt.y));
      // kp_2d.push_back(Point2f(keypoints_2[match_i.trainIdx].pt.x,keypoints_2[match_i.trainIdx].pt.y));
        }
      }

    }
  }




  cv::Mat K =(cv::Mat_<double>(3, 3) << 721.5377, 0,609.5593 , 0, 721.5377, 172.854, 0, 0, 1);
  Mat rvec,tvec,R_rvec;
 vector<int> inliers;
 solvePnPRansac(kp_3d,kp_2d,K,Mat(),rvec, tvec, false, 500, 2.0f, 0.999, inliers, cv::SOLVEPNP_ITERATIVE);
 double inliers_ratio=1.0*inliers.size()/kp_3d.size();
 cout<<"inliers.size()"<<inliers.size()<<endl;
cv::Rodrigues	(rvec,R_rvec);
 //!提出 outlier 并使用 pnp 求解位姿
 vector<Point3f> pts3d;
 vector<Point2f> pts2d;
 for(int i=0;i<inliers.size();i++)
 {
 pts3d.push_back(kp_3d[inliers[i]]);
 pts2d.push_back(kp_2d[inliers[i]]);
 }
 Sophus::SE3d T_ref_from_cur;
 cout<<"pts3d.size()"<<pts3d.size()<<"inliers_ratio"<<inliers_ratio<<endl;
  if(pts3d.size()>=5 && inliers_ratio>0.30)
 {
  // cout<<"yes"<<endl;
 solvePnP(kp_3d, kp_2d, K, Mat(), rvec, tvec);
 //!将两帧相对位姿累计，计算相对于出发点的位姿
 Eigen::Matrix3d Rm;
Rm<<R_rvec.at<double>(0,0), R_rvec.at<double>(0,1), R_rvec.at<double>(0,2),
                                           R_rvec.at<double>(1,0), R_rvec.at<double>(1,1), R_rvec.at<double>(1,2),
                                           R_rvec.at<double>(2,0), R_rvec.at<double>(2,1), R_rvec.at<double>(2,2);
 T_ref_from_cur = Sophus::SE3d(Sophus::SO3d(Rm),Eigen::Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0)));
 if((tvec.at<double>(0,0)*tvec.at<double>(0,0)+ tvec.at<double>(1,0)*tvec.at<double>(1,0)+ tvec.at<double>(2,0)*tvec.at<double>(2,0))>2.0)
 {
 T_ref_from_cur=Last_ref_from_cur;
 //cout<<"Yes"<<endl;
 }
 }
 else
 {
 //T_ref_from_cur=Last_ref_from_cur;
 
 }

 Last_ref_from_cur = T_ref_from_cur;
 Sophus::SE3d T_cur_from_ref = T_ref_from_cur.inverse();
 T_cur = T_cur_from_ref * T_cur;
  Mat t = (Mat_<double>(3,1)<<T_cur.translation()[0],
 T_cur.translation()[1],
 T_cur.translation()[2]);
 Mat r = 
(Mat_<double>(3,3)<<T_cur.rotationMatrix()(0,0),T_cur.rotationMatrix()(0,1),T_cur.rotationMatrix()(0,2),
 T_cur.rotationMatrix()(1,0),T_cur.rotationMatrix()(1,1),T_cur.rotationMatrix()(1,2),
 T_cur.rotationMatrix()(2,0),T_cur.rotationMatrix()(2,1),T_cur.rotationMatrix()(2,2));
 t = r.inv()*t;




 fin>> R00>>R01>>R02>>R03>>R10>>R11>>R12>>R13>>R20>>R21>>R22>>R23;

 int x=-t.at<double>(0,0)+300;
 int y= t.at<double>(2,0)+500;
int x0=R03+300;
int y0=-R23+500;



 cv::circle(traj,Point(x,y),1,Scalar(255,255,255),-1);

 cv::circle(traj,Point(x0,y0),1,Scalar(0,0,255),-1);

  cout<<"x="<<tvec.at<double>(0,0)<<"y="<< R_rvec.at<double>(0,0)
<<endl;

  cout<<"kp_3d size:"<<kp_3d.size()<<"kp_2d size :"<<kp_2d.size()<<endl;
t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "po3d-2d choosing  =:" << time_used.count() << " seconds. " << endl;

  cout<<"Img size:rows"<<img_1.rows<<"cols:"<<img_1.cols<<endl;
  

  
  

  //-- 第五步:绘制匹配结果
  //Mat img_match;
 Mat img_goodmatch;
 
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_goodmatch);
  imshow("img_last", traj);
  cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.5, 0.5,cv::INTER_NEAREST);
  imshow("good matches", img_goodmatch);
  waitKey(1);
  img_last=img_1;
  keypoints_last=keypoints_1;
  descriptors_last=descriptors_1;
  }

  return 0;
}
