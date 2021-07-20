#include "myUtil.h"
#include "ransac.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )	{
  if (argc < 3) {
    cout << "[usage] argv[1]=point_num, argv[2]=point_num_used" << endl;
    return 0;
  }
  int point_num = atoi(argv[1]);  // 上位point_num個の特徴点を運動推定に使用
  int point_num_uesd = atoi(argv[2]);  // 上位point_num_uesd個の特徴点を3次元点群表示に使用

  // 画像読み込み
  string filename1 = "image/R0010027.JPG";
  string filename2 = "image/R0010028.JPG";
  Mat img_1_big = imread(filename1, 0);
  Mat img_2_big = imread(filename2, 0);

  // リサイズ
  Mat img_1, img_2;
  resize(img_1_big, img_1, Size(0,0), 0.15, 0.15);
  resize(img_2_big, img_2, Size(0,0), 0.15, 0.15);

  // 画像サイズ取得，画像中心を保存
  int rows = img_1.rows;
  int cols = img_1.cols;
  cout << rows << "," << cols << endl;
  int center_row = rows/2;
  int center_col = cols/2;
  double f0 = cols;
  //cout << center_row << "," << center_col << endl;
  Point2f center(center_col, center_row);

  // 特徴点抽出, 特徴量計算
  Ptr<Feature2D> fdetector;
  fdetector = ORB::create();
  //fdetector = AKAZE::create();
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  fdetector->detectAndCompute(img_1, Mat(), keypoints1, descriptors1);
  fdetector->detectAndCompute(img_2, Mat(), keypoints2, descriptors2);

  // 対応点のマッチング
  Ptr<BFMatcher> bf;
  bf = BFMatcher::create(NORM_HAMMING, true);
  vector<DMatch> matches;
  bf->match(descriptors1, descriptors2, matches);

  cout << "matches.size()= " << matches.size() << endl;

  // マッチングが強い(距離が短い)ものから昇順にソート
  cout << "[DEBUG] point_num = " << point_num << endl;
  sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) {return a.distance < b.distance; });
  matches.resize(point_num);

  // カメラ面の座標(u, v)から角度(theta, phi)に変換
  vector<Point2f> points1, points2;
  Point2f mul(2.0*MY_PI/cols, MY_PI/rows);
  for (int i=0; i<matches.size(); i++) {
    int idx_1 = matches[i].queryIdx;
    int idx_2 = matches[i].trainIdx;
    Point2f point1_buf;
    Point2f point2_buf;
    Util::point2f_multi(keypoints1[idx_1].pt - center, mul, point1_buf);
    Util::point2f_multi(keypoints2[idx_2].pt - center, mul, point2_buf);
    points1.push_back(point1_buf);
    points2.push_back(point2_buf);
  }

  // 視線に変換
  vector<Point3f> points1_ray, points2_ray;
  for (int i=0; i<matches.size(); i++) {
    Point3f point1_ray_buf(cos(points1[i].y)*sin(points1[i].x), sin(points1[i].y), cos(points1[i].y)*cos(points1[i].x));
    Point3f point2_ray_buf(cos(points2[i].y)*sin(points2[i].x), sin(points2[i].y), cos(points2[i].y)*cos(points2[i].x));
    points1_ray.push_back(point1_ray_buf);
    points2_ray.push_back(point2_ray_buf);
  }


  // 特徴量マッチング結果の表示
  Mat resImg;
  drawMatches(img_1, keypoints1, img_2, keypoints2, matches, resImg);
  cv::imshow("image", resImg);
  cv::waitKey();


  Matrix3d E; 
  Ransac r;
  r.ransac(points1_ray, points2_ray, 10, MY_PI/18.0, E);

/*
  // 8点法
  // 行列Aを設定 Ax = 0
  Eigen::MatrixXd A(static_cast<int>(matches.size()), 9);
  for (int i=0; i<matches.size(); i++) {
    A(i, 0) = points1_ray[i].x*points2_ray[i].x;
    A(i, 1) = points1_ray[i].x*points2_ray[i].y;
    A(i, 2) = points1_ray[i].x*points2_ray[i].z;
    A(i, 3) = points1_ray[i].y*points2_ray[i].x;
    A(i, 4) = points1_ray[i].y*points2_ray[i].y;
    A(i, 5) = points1_ray[i].y*points2_ray[i].z;
    A(i, 6) = points1_ray[i].z*points2_ray[i].x;
    A(i, 7) = points1_ray[i].z*points2_ray[i].y;
    A(i, 8) = points1_ray[i].z*points2_ray[i].z;
  }
  JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
  VectorXd min_vector = svd.matrixV().col(8);
  cout << "Vmin = \n" << min_vector << endl;

  // 基本行列
  Matrix3d E; 
  E << min_vector(0), min_vector(1), min_vector(2), min_vector(3), min_vector(4), min_vector(5), min_vector(6), min_vector(7), min_vector(8);
*/

  
  // 並進パラメータt
  EigenSolver<Matrix3d> es_EEt(E*E.transpose());
  //cout << "eigen values = \n"<< es_EEt.eigenvalues() << endl;
  //cout << "eigen vectors = \n"<< es_EEt.eigenvectors() << endl;
  Vector3cd values = es_EEt.eigenvalues();
  int min_index = Util::find_minEigenValue3(values);
  Vector3cd  t = es_EEt.eigenvectors().col(min_index);
  Vector3d t_double;
  t_double << t[0].real() ,t[1].real(), t[2].real();
  cout << "t = " << t_double << endl;

  // 回転行列1
  Matrix3d t_cross;
  t_cross = Util::calc_CrossMatrix(t_double);
  Matrix3d t_cross_E;
  t_cross_E = -1*t_cross*E;
  //特異値分解
  JacobiSVD<Matrix3d> svdR(t_cross_E, ComputeFullU | ComputeFullV);
  Matrix3d R, Lambda, UVt;
  UVt = svdR.matrixU() * svdR.matrixV().transpose();
  Lambda << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, UVt.determinant();
  R = svdR.matrixU() * Lambda * svdR.matrixV().transpose();
  cout << "R = " << R << endl;

  // 符号チェック
  int cnt=0;
  for (int i=0; i<matches.size(); i++) {
    Vector3d v1, v2;
    v1(0) = points1_ray[i].x; v1(1) = points1_ray[i].y; v1(2) = points1_ray[i].z;
    v2(0) = points2_ray[i].x; v2(1) = points2_ray[i].y; v2(2) = points2_ray[i].z;
    double theta1 = acos(v1.dot(t_double));
    double theta2 = acos((R*v2).dot(-1*t_double));
    if (theta1+theta2 < MY_PI) cnt++;
  }
  if (cnt < matches.size()/2) t_double *= -1;
  cout << "updated_t = " << t_double << endl;

  cout << "End." << endl;
  return 0;
}
