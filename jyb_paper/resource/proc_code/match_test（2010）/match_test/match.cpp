#include<iostream>
#include<sstream>
#include<string>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\core\core.hpp>

using namespace std;
using namespace cv;

Mat      src_img, src_img1,roi, part2;
Mat		 locateImage;								     //��λģ��
Mat      locateUp;
Mat      temp;
Mat      rect;//���ڶ��ζ�λģ��
Point	 matchLocation;
Point	 matchLocation0;                               //���ڴ洢����ڴ�ͼ�ľ��Զ�λ��Ϣ
Point    tempLocation;                                   //���ڴ洢΢���������Сͼ�����λ����Ϣ
Point    midLocation;
int		 partw = 150;
int		 parth = 190;
int      judge_count = 0;

struct Node
{
	int x;
	int y;
	Node():x(0),y(0){};
};
template<typename T>
string ValToString(T& val)
{
	ostringstream os;
	os << val;
	return os.str();
}
void Locate(Mat &input, Mat &input2, Point &location)
{
	if (!input.data)
	{
		cout << ("û�е�֡ͼ������������ڲ�ͼ������ָ���Ƿ�Ϊ�գ�");
		return;
	}
	Mat resultimage, srcimage1;
	srcimage1 = input.clone();
	int m_tw = input2.cols;
	int m_th = input2.rows;
	int resultimage_cols = srcimage1.cols - m_tw + 1;
	int resultimage_rows = srcimage1.rows - m_th + 1;
	resultimage.create(resultimage_rows, resultimage_cols, CV_32FC1);//��ͼ�����ռ䣬��ͨ��  CV_32FC1   CV_8UC4

	//����ƥ��ͱ�׼�� �÷���1
	matchTemplate(srcimage1, input2, resultimage, 4);//***********************ƥ�䣬ģ������������ͼ�񣬱���Ϊ��ͨ��32-bitλ������ͼ��
	normalize(resultimage, resultimage, 0, 1, NORM_MINMAX, -1, Mat());//��һ��Ϊ��ͨ��
	//��һ����Χ��0��1��                  ����
	//ͨ������minMaxLoc��λ��ƥ���λ��
	double minValue, maxValue;
	Point	minLocation, maxLocation;

	minMaxLoc(resultimage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());//************�ҳ�ƥ�����
	//���ڷ���SQDIFF(0)��SQDIFF_NORMED(1),ԽС����ֵ���Ÿ��ߵ�ƥ��Ч����������ķ�������ֵԽ��ƥ��Ч��ԽС
	location = maxLocation;
}
//bool continuity(Mat& input)
//{
//	bool t_f = true;//����ֵ
//
//	int c_long = 0;//ͳ���У��������ߣ�
//	int r_count = 0;//ͳ�Ƶ��а׵���
//	int beg = 0, end = 0;//�����������Ҽ���λ��
//	//cout<<"rows:"<<input.rows<<"   cols:"<<input.cols<<endl;
//	for (int i = input.rows - 70; i < input.rows; i++)
//	{
//		r_count = 0;
//		int r_long = 0;
//		uchar* data = input.ptr<uchar>(i);
//		for (int j = 1; j < input.cols; j++)
//		{
//			if (data[j] > 0) r_count++;
//			if ((data[j]>data[j - 1]) && r_count == 1) beg = j;//����������ʼλ�� 
//			if (data[j]<data[j - 1]) end = j;//��������λ��
//			r_long = end - beg;//�����������
//		}
//		if (r_long < 30) c_long++;//���õ��߿����ֵΪ30
//		//cout<<"r="<<i<<"  r_long:"<<r_long<<endl;
//		if (r_long >= 30 && c_long<30) c_long = 0;//�жϵ�������������30�ж��ǵ���
//		if (c_long>30) break;
//	}
//	if (c_long >30) t_f = false;
//	else t_f = true;
//	//cout << "c_long:" << c_long << endl;
//	return t_f;
//}
void contourProc(Mat &input, Mat &output)
{
	/******ǰ�����������ȱ���������*********/
	int m_ih = input.rows;
	int loc_x = 0;
	int loc_y = 0;
	//Mat input_dilate;
	//Mat element=getStructuringElement(MORPH_RECT,Size(5,5));
	//dilate(input,input_dilate,element);
	//imshow("input_dilate",input_dilate);
	output = Mat::zeros(input.size(), CV_8UC1);


	vector<Mat> contours1;
	Mat hierarchy1;
	findContours(input, contours1, hierarchy1, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	vector<Moments> mu(contours1.size());
	vector<Point2f> mc(contours1.size());

	int count = 0;

	/*********��ʼ����������ɸѡ�����Ĳ�����ȴ���***************/
	for (int i = 0; i< contours1.size(); i++)
	{
		if (contourArea(contours1[i])>200)//ȥ��С�������
		{
			Scalar color = Scalar(255);
			//������������
			mu[i] = moments(contours1[i], false);
			mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

			if ((mc[i].x>30 && mc[i].y>=160 && mc[i].x<120)||mc[i].y<160)//ȥ���������ұ߽Ǵ�����
			{
				//cout << "mc_x:" << mc[i].x << " mc_y" << mc[i].y << endl;
				drawContours(output, contours1, i, color, 1, 8, hierarchy1, 0, Point());
			}
		}
	}
}
//int right_node(int x, int y, Mat& img, vector<vector<int>>& marked, int max_x, int max_y)
//{
//	vector<vector<int>> vv = { { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, -1 }, { 1, -1 } };
//	for (int i = 0; i < 8;i++)
//	{
//		if (marked[x + vv[i][0]][y + vv[i][1]]==1) continue;
//		else marked[x + vv[i][0]][y + vv[i][1]] = 1;
//
//		if (x + vv[i][0]<=5 || x + vv[i][0]>=img.cols-5 || y + vv[i][1]<=5 || y + vv[i][1]>=img.rows-5) continue;
//
//		uchar* data0 = img.ptr<uchar>(y + vv[i][1]);
//		int data = data0[x + vv[i][0]];
//
//		//cout << "x=" << x + vv[i][0] << "y=" << y + vv[i][1] <<"  data="<<data<< endl;
//		if ((int)data != 0)
//		{
//			if (x + vv[i][0]>=max_x)
//			{
//				max_x = x + vv[i][0];
//				max_y = y + vv[i][1];
//				//cout << "max_x=" << x + vv[i][0] << " max_y=" << y + vv[i][1] << endl;
//			}
//			//cout << ">0" << endl;
//			max_y=right_node(x + vv[i][0], y + vv[i][1], img, marked,max_x,max_y);
//			break;
//		}
//	}
//	//cout << "max_y=" << max_y << endl;
//	return max_y;
//}
//int min_line(Mat & input_contour)
//{
//	int light_count = 0, light_count0 = 0;
//	int beg = 0, end = 0, beg0 = 0, end0 = 0;
//	int count = 0, count0 = 0;
//	int min_line = 100;
//	int min_flag = 0;
//	for (int i = input_contour.rows - 100; i<input_contour.rows; i++)//��⽻���λ��
//	{
//		uchar *data = input_contour.ptr<uchar>(i);
//		for (int j = 0; j<input_contour.cols; j++)//ͳ�Ƶ����������
//		{
//			if ((int)data[j] != 0) light_count0++;
//			if ((data[j - 1]<data[j]) && (light_count0 == 1)) beg0 = j;
//			if ((int)data[j]<(int)data[j - 1]) end0 = j;
//		}
//		if ((end0 - beg0) <= min_line) //ͳ����С���λ�ã�����㣩
//		{
//			min_line = (end0 - beg0);
//			min_flag = i;
//		}
//	}
//	return min_flag;
//}
bool turning_node(Mat & input,Mat& color_img)
{
	Node up_node, middle_node, low_node;
	for (int r = 70; r < input.rows; r++)
	{
		int light_count = 0;
		uchar *data = input.ptr<uchar>(r);
		for (int c = 0; c < input.cols; c++)
		{
			if (data[c] == 255) light_count++;
			if (light_count>1) break;
			if (up_node.x == 0 && data[c] == 255 && c < 60)
			{
				up_node.x = r;
				up_node.y = c;
				middle_node.x = r;
				middle_node.y = c;
				break;
			}
			if (up_node.x != 0 && low_node.x==0 && data[c] == 255 && c>middle_node.y && c - middle_node.y < 10)
			{
				middle_node.x = r;
				middle_node.y = c;
				break;
			}
			if (up_node.x != 0 && middle_node.x != 0 && low_node.x==0 && r - middle_node.x>30) return false;
			if (middle_node.y - c>10 && low_node.x == 0 && data[c] == 255)
			{
				low_node.x = r;
				low_node.y = c;
				break;
			}
			if (up_node.x != 0 && data[c] == 255 && middle_node.x != 0 && low_node.x != 0 && c < low_node.y && low_node.y - c < 10)
			{
				low_node.x = r;
				low_node.y = c;
				break;
			}

		}
	}
	rectangle(color_img, Point(up_node.y, up_node.x), Point(up_node.y + 10,up_node.x + 10), Scalar(0, 0, 255), 1, 8);
	rectangle(color_img, Point(middle_node.y, middle_node.x), Point(middle_node.y + 10,middle_node.x + 10), Scalar(0, 0, 255), 1, 8);
	rectangle(color_img, Point(low_node.y, low_node.x), Point(low_node.y + 10,low_node.x + 10), Scalar(0, 0, 255), 1, 8);

	if (middle_node.x - up_node.x < 10||low_node.x-up_node.x<40) return false;
	if (low_node.x != 0) return true;
	else return false;
}
void RGB_threshold(Mat& input)
{
	for (int i = 0; i <input.rows; i++)//���ڻ�ɫ��ͨ����ֵ����
	{
		for (int j = 0; j < input.cols; j++)
		{
			if (input.at<Vec3b>(i, j)[2]>input.at<Vec3b>(i, j)[1])
			{              //     ����ͨ��                 �ڶ�ͨ��
				input.at<Vec3b>(i, j)[0] = 255;//��һͨ��
				input.at<Vec3b>(i, j)[1] = 255;//
				input.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				input.at<Vec3b>(i, j)[0] = 0;
				input.at<Vec3b>(i, j)[1] = 0;
				input.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
}
bool judge(Mat& input, Mat& out_img)
{
	judge_count++;
	out_img = input.clone();
	RGB_threshold(input);
	Mat input_gray, input_contour;//m11,m1;
	cvtColor(input, input_gray, CV_RGB2GRAY);
	cv::imshow("input_gray", input_gray);
	contourProc(input_gray, input_contour);
	cv::imshow("first_pro", input_contour);

	int min_line = 100;
	int min_flag = 0;
	bool T_F;
	T_F = turning_node(input_contour,out_img);
	if (T_F == true) cout << "�ϸ�" << endl;
	else
	{
		if (judge_count > 1) return T_F;
		cout << "���ϸ�" << endl;
		input = src_img1(Rect(tempLocation.x + matchLocation.x + 160, tempLocation.y + matchLocation.y, partw, parth + 30));
		out_img = input.clone();
		T_F=judge(input, out_img);
	}
	//bool t_f;
	//t_f = continuity(input_contour);//˫���������ж�
	//t_f = true;
	//if (t_f == false)
	//{
	//	cout << "��ת�Ӳ��ϸ�" << endl;
	//	T_F = false;
	//}
	//else
	//{
	//	uchar *data = input_contour.ptr<uchar>(100);
	//	for (int j = 0; j < input_contour.cols; j++)
	//	{
	//		vector<vector<int>> marked(200,vector<int>(200,0));//����ѷ��ʵ�ı�Ǿ���
	//		if ((int)data[j] != 0)
	//		{
	//			min_flag = right_node(j, 100, input_contour, marked, 0, 0);//���㽻���λ��
	//			break;
	//		}
	//	}
	//	//���������λ�ã�����ͼ��ĸߣ�**************************************************************************************
	//	Mat contour_temp = input_contour;
	//	if (min_flag<(contour_temp.rows - 35))//�µ�,�����λ�ã�min_flag)�����±�Ե����30
	//	{
	//		int parthh = parth - (contour_temp.rows-20- min_flag);
	//		if (parthh < 50) parthh =parthh+10;
	//		input = src_img1(Rect(tempLocation.x + matchLocation.x + 160, tempLocation.y + matchLocation.y, partw, parthh));
	//		out_img = input.clone();
	//		cv::imshow("jianshao_color", input);
	//		cout << "�µ�" << endl;
	//		RGB_threshold(input);
	//		cvtColor(input, input_gray, CV_RGB2GRAY);
	//		cv::imshow("jianshao", input_gray);
	//		contourProc(input_gray, input_contour);
	//	}
	//	if (min_flag>(contour_temp.rows - 20))//�ϵ�
	//	{
	//		input = src_img1(Rect(tempLocation.x + matchLocation.x + 160, tempLocation.y + matchLocation.y, partw, parth + 20));
	//		out_img = input.clone();
	//		cv::imshow("zengjia_color", input);
	//		cout << "�ϵ�" << endl;
	//		RGB_threshold(input);
	//		cvtColor(input, input_gray, CV_RGB2GRAY);
	//		cv::imshow("zengjia", input_gray);
	//		contourProc(input_gray, input_contour);
	//	}
	//	//***********************************************************************************************************************************************************
	//	//����±�Ե���
	//	int light_count = 0, beg = 0, end = 0, count = 0;
	//
	//	uchar *data1 = input_contour.ptr<uchar>(input.rows - 3);
	//	for (int j = 1; j<input_contour.cols; j++)
	//	{
	//		if ((int)data1[j] != 0) light_count++;//������Ե����
	//		if ((data1[j - 1]<data1[j]) && (light_count == 1)) beg = j;//������ʼλ��
	//		if ((int)data1[j]<(int)data1[j - 1]) end = j;//��ֹλ��
	//	}
	//	count = end - beg;//�±�Ե�������
	//	if (count>30)
	//	{
	//		cout << "ת�Ӻϸ�" << endl;//�ж�
	//		T_F = true;
	//	}
	//	else
	//	{
	//		cout << "ת�Ӳ��ϸ�" << endl;
	//		T_F = false;
	//	}
	//}
	//cv::imshow("pro", input_contour);
	return T_F;
}

void IsOk(string path, ostringstream& n,ostringstream& m)  /////ת��ת��24��
{
	Mat out_img;
	bool T_F = true;
	int cycleCount = 1;
	int locw, loch;
	
	//locateImage = imread("t1.jpg", 1);
	//locateUp = imread("t3.jpg", 1);

	locateImage = imread("96100_big_model.jpg", 1);
	locateUp = imread("96100_model.jpg", 1);

	locw = locateImage.cols;//740
	loch = locateImage.rows;//300
	int roiw = locateUp.cols;//450
	int roih = locateUp.rows;//200

	while (true)//һȦ
	{
		if (cycleCount>3481) break;
		string image_path = "";
		image_path = path + ValToString(cycleCount) + ".jpg";
		Mat src_img = imread(image_path,1);
		if (src_img.empty()) { cout << image_path << "���ݴ���!" << endl;  break; }
		else cout << "����" << image_path << "��������!" << endl;
		imshow("Pic", src_img);
		if (cycleCount != 0)//ƥ��
		{
			Locate(src_img, locateImage, matchLocation);
			rect = src_img.clone();
			src_img1 = src_img.clone();
			rectangle(rect, Point(matchLocation.x, matchLocation.y), Point(matchLocation.x + locw, matchLocation.y + loch), Scalar(0, 0, 255), 1, 8);
			temp = src_img(Rect(matchLocation.x, matchLocation.y, locw, loch));
			Locate(temp, locateUp, tempLocation);
			rectangle(rect, Point(tempLocation.x + matchLocation.x, tempLocation.y + matchLocation.y), Point(tempLocation.x + matchLocation.x + roiw, tempLocation.y + matchLocation.y + roih), Scalar(0, 0, 255), 1, 8);
			cv::imshow("rect", rect);
			cv::imshow("src_img", src_img);
			roi = temp(Rect(tempLocation.x, tempLocation.y, roiw, roih));
			part2 = src_img(Rect(tempLocation.x + matchLocation.x + 160, tempLocation.y + matchLocation.y, partw, parth));
		}
		else
		{
			temp = src_img(Rect(matchLocation.x, matchLocation.y, locw, loch));
			rect = src_img.clone();
			src_img1 = src_img.clone();
			rectangle(rect, Point(matchLocation.x, matchLocation.y), Point(matchLocation.x + locw, matchLocation.y + loch), Scalar(0, 0, 255), 1, 8);
			rectangle(rect, Point(tempLocation.x + matchLocation.x, tempLocation.y + matchLocation.y), Point(tempLocation.x + matchLocation.x + roiw, tempLocation.y + matchLocation.y + roih), Scalar(0, 0, 255), 1, 8);
			cv::imshow("rect", rect);
			cv::imshow("src_img", src_img);
			part2 = src_img(Rect(tempLocation.x + matchLocation.x + 160, tempLocation.y + matchLocation.y, partw, parth));
		}
		judge_count = 0;
		T_F = judge(part2, out_img);//���ࣨ�ϸ�
		if (T_F == true) imwrite("C:\\Users\\Jia Youbin\\Desktop\\rotor\\gb\\good\\" + n.str() + "-" + m.str() +"-"+ ValToString(cycleCount)+".jpg", out_img);
		else		  imwrite("C:\\Users\\Jia Youbin\\Desktop\\rotor\\gb\\bad\\"+n.str() + "-" + m.str() + "-" + ValToString(cycleCount) + ".jpg", out_img);
		cycleCount++;
		cv::imshow("out_img", out_img);
		waitKey(0);
	}
}

int main()
{
	for (int n = 1; n<=1; n++)
	{
		ostringstream os;
		os<<n;
		for (int m = 1; m <= 1;m++)
		{
			ostringstream oss;
			oss << m;
			string path = "E:\jia_you_bing\\"+os.str()+"/"+oss.str()+"/";
			IsOk(path, os,oss);
		}
	}
	getchar();
}



