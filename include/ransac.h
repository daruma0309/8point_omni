#ifndef RANSAC_H_
#define RANSAC_H_

#include "myUtil.h"

#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <random>

class Ransac {
public:
  Ransac() {}
  ~Ransac(){}

  // RANSACによるアウトライン除去
  void ransac(vector<Point3f> points1, vector<Point3f> points2, int point_num, double d, Matrix3d &E_best);
};

#endif
