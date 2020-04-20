

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

string to_str(int i){
	ostringstream is;
	is<<i;
	return is.str(); 
}
void rgb_adjust(Mat& input)
{
	for (int i = 0; i <input.rows; i++)//基于黄色三通道阈值处理
	{
		for (int j = 0; j < input.cols; j++)
		{
        		input.at<Vec3b>(i, j)[0] -= 1;//第一通道
				input.at<Vec3b>(i, j)[1] -= 5;
				input.at<Vec3b>(i, j)[2] -= 2;
		}
	}
}
int Flip(int num){
	
	//读入图像，并判断图像是否读入正确
	string path_read = "C:\\Users\\Administrator\\Desktop\\" + to_str(num) +".png";
	string path_out = "C:\\Users\\Administrator\\Desktop\\" + to_str(num) +"_.png";
	cv::Mat srcImage = imread(path_read), outImage;
	if (!srcImage.data){
		cout<<"read image error!"<<endl;
		return -1;
	}

	flip(srcImage, outImage, 1);
	imshow("srcImage", outImage);
	rgb_adjust(outImage);
	imshow("adjust_srcImage", outImage);
	imwrite(path_out,outImage);
	waitKey(0);
	return 0;
}

int main(){
	for(int i=1; i<=12; i++){
		Flip(i);
	}
	return 0;
}