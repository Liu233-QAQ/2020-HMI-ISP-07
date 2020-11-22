//

//
#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"
#include <boost/format.hpp>

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam {

Viewer::Viewer() {
    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    std::string path="./00.txt";
    fin=std::ifstream(path);
}

void Viewer::Close() {
    viewer_running_ = false;
    viewer_thread_.join();
}

void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    current_frame_ = current_frame;
}

void Viewer::UpdateMap() {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    assert(map_ != nullptr);
    active_keyframes_ = map_->GetActiveKeyFrames();
    active_landmarks_ = map_->GetActiveMapPoints();
    keyframes_=map_->GetAllKeyFrames();
    landmarks_=map_->GetAllMapPoints();
    map_updated_ = true;
}

void Viewer::ThreadLoop() {
    pangolin::CreateWindowAndBind("Stereo Visual Odometry based on SURF", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};

    while (!pangolin::ShouldQuit() && viewer_running_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        if (current_frame_) {
            DrawFrame(current_frame_, blue);
            //FollowCurrentFrame(vis_camera);

            cv::Mat img = PlotFrameImage();
            cv::imshow("image", img);
            cv::waitKey(1);
        }

        if (map_) {
          DrawMapPoints();
        }

        pangolin::FinishFrame();
        usleep(5000);
    }

    LOG(INFO) << "Stop viewer";
}

cv::Mat Viewer::PlotFrameImage() {
    cv::Mat img_out_left,img_out_right;
    cv::cvtColor(current_frame_->left_img_, img_out_left, CV_GRAY2BGR);
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.lock()) {
            auto feat = current_frame_->features_left_[i];
            cv::circle(img_out_left, feat->position_.pt, 6, cv::Scalar(93, 66, 255),
                       2);
        }
    }
    cv::cvtColor(current_frame_->right_img_, img_out_right, CV_GRAY2BGR);
   /*
    for (size_t i = 0; i < current_frame_->features_right_.size(); ++i) {
        if (current_frame_->features_right_[i]->map_point_.lock()) {
            auto feat = current_frame_->features_right_[i];
            cv::circle(img_out_right, feat->position_.pt, 6, cv::Scalar(93, 66, 255),
                       2);
        }
    }
    */
    
    cv::Mat img_out;
   
    cv::vconcat(img_out_left,img_out_right,img_out);
     boost::format number_id("image_%d:%06d");
     boost::format keyframe_id("KeyFrame_Id:%d");
     boost::format is_keyframe("Is_KeyFrame:%s");
     boost::format Pose("Pose{x:%.1f,y:%.1f,z:%.1f");
     static unsigned long  KF_ID=0;
     if(current_frame_->is_keyframe_==true){
         KF_ID=current_frame_->keyframe_id_;
         cv::putText(img_out, (is_keyframe % "true").str(), cv::Point(10,260), cv::FONT_HERSHEY_TRIPLEX, 0.8, 
                                                                              cv::Scalar(6,128, 67),2);
     }
     else{
         cv::putText(img_out, (is_keyframe % "false").str(), cv::Point(10,260), cv::FONT_HERSHEY_TRIPLEX, 0.8, 
                                                                              cv::Scalar(6,128, 67),2);
     }
     


    cv::putText(img_out, (number_id % 0 % (2*(current_frame_->id_))).str(), cv::Point(10,210), cv::FONT_HERSHEY_TRIPLEX, 0.8, 
                                                                              cv::Scalar(6,128, 67),2);
    cv::putText(img_out, (keyframe_id % KF_ID).str(), cv::Point(10,235), cv::FONT_HERSHEY_TRIPLEX, 0.8, 
                                                                              cv::Scalar(6,128, 67),2);
    cv::putText(img_out, (Pose%current_frame_->Pose().inverse().translation()[0]%current_frame_->Pose().inverse().translation()[1]%current_frame_->Pose().inverse().translation()[2]).str(), cv::Point(10,285), cv::FONT_HERSHEY_TRIPLEX, 0.8, 
                                                                              cv::Scalar(6,128, 67),2);
   

    return img_out;
}

void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    SE3 Twc = current_frame_->Pose().inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
}

void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
    SE3 Twc = frame->Pose().inverse();
    const float sz = 1.0;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;
    //float R00,R01,R02,R03,R10,R11,R12,R13,R20,R21,R22,R23;
    //fin>> R00>>R01>>R02>>R03>>R10>>R11>>R12>>R13>>R20>>R21>>R22>>R23;
    //Eigen::Isometry3f T1=Eigen::Isometry3f::Identity();
   // Eigen::Matrix3f R;
   // R(0,0)=R00,R(0,1)=R01,R(0,2)=R02;
   // R(1,0)=R10,R(1,1)=R11,R(1,2)=R12;
   // R(2,0)=R20,R(2,1)=R21,R(2,2)=R22;
   // Eigen::Vector3f t(R03,R13,R23);
   /*
    T1(0,0) = R00, T1(0,1) = R01, T1(0,2) = R02, T1(0,3) = R03;
T1(1,0) = R10, T1(1,1) = R11, T1(1,2) = R12, T1(1,3) = R13;
T1(2,0) = R20, T1(2,1) = R21, T1(2,2) =R22, T1(2,3) = R23;
T1(3,0) =    0, T1(3,1) =    0, T1(3,2) =  0, T1(3,3) =   1;
*/
//SE3 Twc(R,t);
//Twc=Twc.inverse();
    glPushMatrix();

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    m(1,3)=0.0f;
    glMultMatrixf((GLfloat*)m.data());

    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}
 void DrawTrajectory(Frame::Ptr current,Frame::Ptr last,const float* color){
     // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(2);
    auto p1=current->Pose().inverse();
    auto p2=last->Pose().inverse();
     glBegin(GL_LINES);
      glVertex3f(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3f(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    

 }

void Viewer::DrawMapPoints() {
   //std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    const float red[3] = {3.0/256, 101.0/256, 100.0/256};
    const float color2[3] = {34.0/256, 8.0/256, 7.0/256};
   // Frame::Ptr  lf;
    for (auto& kf : keyframes_) {
       // if(lf!=nullptr)
      // DrawTrajectory(lf,kf.second,color2);
        DrawFrame(kf.second, color2);
      //  lf=kf.second;
    }
     /*
    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto& landmark : landmarks_) {
        auto pos = landmark.second->Pos();
        glColor3f(red[0], red[1], red[2]);
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
    */
    
}

}  // namespace myslam
