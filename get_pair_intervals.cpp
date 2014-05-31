#define _MAIN

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>       /* atan2 */

#define PI 3.14159265

using namespace std;
using namespace cv;

bool sortRects (Rect i,Rect j) { return (i.x<j.x); }
bool sortContours (vector<Point> i, vector<Point> j) { return (boundingRect(i).x<boundingRect(j).x); }

int main( int argc, char** argv )
{


    if ((argc < 3) || (argc%2 != 1))
    {

      cout << "Usage: ./get_pair_intervals <file1> <gt_file1> <file2> <gt_file2> ... <fileN> <gt_fileN>" << endl;
      return -1;

    }

    vector<float> height_ratios;
    vector<float> centroid_angles;
    vector<float> norm_distances;
    vector<int> grey_distances;
    vector<float> lab_distances;
    
    for (int f=1; f<argc; f=f+2) {


      Mat gt,src,lab,grey;
      src = imread(argv[f]);
      cvtColor(src, lab, CV_RGB2Lab);
      cvtColor(src, grey, CV_RGB2GRAY);
      gt = imread(argv[f+1]);
      //imshow( "src", src );
      //imshow( "gt", gt );
      cvtColor(gt, gt, CV_RGB2GRAY);
      threshold(gt, gt, 1, 255, CV_THRESH_BINARY_INV);
     
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
       
      findContours( gt, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

      vector<Rect> char_boxes;
      vector< vector<Point> > char_contours;
      vector<int> grey_means;
      vector< vector<int> > lab_means;
      vector<int> tmp;
      lab_means.push_back(tmp);
      lab_means.push_back(tmp);
      lab_means.push_back(tmp);

      for( int i = 0; i< contours.size(); i++ )
      {
        Rect bbox = boundingRect(contours[i]);

        if((hierarchy[i][3] == 0) && ((bbox.width>6) || (bbox.height>6)))
        {
          char_boxes.push_back(bbox);
          char_contours.push_back(contours[i]);
          Mat mask = Mat::zeros( gt.size(), CV_8UC1);
          drawContours( mask, contours, i, Scalar(255), CV_FILLED, 8, hierarchy, 0, Point() );
          drawContours( mask, contours, i, Scalar(0), 1, 8, hierarchy, 0, Point() );
          for( int j = 0; j< contours.size(); j++ )
          {
            if (hierarchy[j][3] == i)
              drawContours( mask, contours, j, Scalar(0), CV_FILLED, 8, hierarchy, 0, Point() );
          }
          Scalar mean,std;
          meanStdDev(grey,mean,std,mask);
          //cout << "grey mean " << mean[0] << endl;
          grey_means.push_back((int)mean[0]);
          Scalar mean_v,std_v;
          meanStdDev(lab,mean_v,std_v,mask);
          lab_means[0].push_back((int)mean_v[0]);
          lab_means[1].push_back((int)mean_v[1]);
          lab_means[2].push_back((int)mean_v[2]);
          //cout << "lab mean " << mean_v[0] << "," << mean_v[1] << "," << mean_v[2] << endl;
          //imshow( "gt contours", mask );
          //waitKey(0);
        }
      }


      if (char_boxes.size() < 2)
        continue;
      
      sort(char_boxes.begin(),char_boxes.end(), sortRects);
      sort(char_contours.begin(),char_contours.end(), sortContours);


      for( int i = 0; i< char_boxes.size()-1; i++ )
      {
        height_ratios.push_back((float)min(char_boxes[i].height,char_boxes[i+1].height) /
                                       max(char_boxes[i].height,char_boxes[i+1].height));
        Point center_i(char_boxes[i].x+char_boxes[i].width/2, char_boxes[i].y+char_boxes[i].height/2);
        Point center_j(char_boxes[i+1].x+char_boxes[i+1].width/2, char_boxes[i+1].y+char_boxes[i+1].height/2);
        centroid_angles.push_back(atan2(center_j.y-center_i.y, center_j.x-center_i.x));
        int avg_width = (char_boxes[i].width + char_boxes[i+1].width) / 2;
        norm_distances.push_back((float)(char_boxes[i+1].x-(char_boxes[i].x+char_boxes[i].width))/avg_width);

        grey_distances.push_back(abs(grey_means[i]-grey_means[i+1]));
        lab_distances.push_back(sqrt(pow(lab_means[1][i]-lab_means[1][i+1],2)+
                                     pow(lab_means[2][i]-lab_means[2][i+1],2)));

        //cout << height_ratios[height_ratios.size()-1] << " " << centroid_angles[height_ratios.size()-1] << " " << norm_distances[height_ratios.size()-1] << endl;
        cout << "." << flush;

        //Set this to true if you want to visualize each possible pair
        //if (grey_distances[grey_distances.size()-1] > 110)
        if (false)
        {
          Mat drawing = Mat::zeros( gt.size(), CV_8UC3);
          drawContours(drawing, char_contours, i, Scalar(255,255,255), CV_FILLED);
          drawContours(drawing, char_contours, i+1, Scalar(255,255,255), CV_FILLED);
          rectangle(drawing, char_boxes[i].tl(), char_boxes[i].br(), Scalar(255,0,0));
          rectangle(drawing, char_boxes[i+1].tl(), char_boxes[i+1].br(), Scalar(255,0,0));
          line(drawing, center_i, center_j, Scalar(0,0,255));

          resize(drawing,drawing,Size(800,600));
          imshow( "src", src );
          imshow( "gt", gt );
          imshow( "pair features", drawing );
          imwrite( "pair_features.jpg", drawing );
          waitKey(0);
        }
      }

    } // end for all GT files

    double minValue, maxValue;
    minMaxIdx(Mat(height_ratios), &minValue, &maxValue);
    cout << endl;
    cout << " height ratios        min=" << minValue << " max=" << maxValue << endl;
    minMaxIdx(Mat(centroid_angles), &minValue, &maxValue);
    cout << " centroid angles      min=" << minValue << " max=" << maxValue << endl;
    minMaxIdx(Mat(norm_distances), &minValue, &maxValue);
    cout << " normalized distances min=" << minValue << " max=" << maxValue << endl;
    minMaxIdx(Mat(grey_distances), &minValue, &maxValue);
    cout << " grey_distance        min=" << minValue << " max=" << maxValue << endl;
    minMaxIdx(Mat(lab_distances), &minValue, &maxValue);
    cout << " (L)ab_distance       min=" << minValue << " max=" << maxValue << endl;
}
