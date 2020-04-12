#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
#include <opencv.hpp>
#include <calib3d/calib3d_c.h>
#include <imgproc/types_c.h>
#include <io.h>
#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include<sys/stat.h>
#include<sys/types.h>


using namespace cv;
using namespace std;


/*
获取某个目录下制定文件类型的文件
将数组FileNames中原有元素清空
将所有符合要求的文件放置到FileNames数组中
*/
bool IsFileExits(string& dir, vector<string>subdir) {
    struct _finddata_t FileInfo;//查询到的文件信息
    intptr_t lfile = 0;//查询到的long类型文件句柄
    string sfile_r;//搜索文件时用到的正则表达式
    lfile = _findfirst(dir.c_str(), &FileInfo);

    if (
        (lfile != -1L//查找存在符合要求的文件
            && FileInfo.attrib != _A_SUBDIR)//而且如果不是目录
        ||
        lfile == -1L//或者没找到
        ) {
        string cmd("mkdir ");
        string tmp(dir.begin(), dir.end());

        system(cmd.append(tmp).c_str());
        for (vector<string>::iterator st = subdir.begin(); st != subdir.end(); ++st) {
            string tmpcmd(cmd);

            system(tmpcmd.append(*st).c_str());
        }
    }

    return true;
}

/*获取文件函数
根据给定的文件目录和文件类型，该目录下分别放置在两个不同子目录下的左右相机图片的路径
需要保证两个子目录的名字分别为left和right
需要保证左右相机图片数量相等
如果正确获得左右相机图片的路径，则返回true，否则返回false
*/
bool GetFile(string& Path, string& Type, vector<string>& FileNames) {
    struct _finddata_t FileInfo;//查询到的文件信息
    intptr_t lfile = 0;//查询到的long类型文件句柄
    string sfile_r;//搜索文件时用到的正则表达式

    vector<vector<string>> images;
    images.resize(2);
    vector<string> subdir{ "\\left","\\right" };
    
    for (int i = 0; i < 2; ++i) {        
        
        lfile = _findfirst(sfile_r.assign(Path).append(subdir.at(i)).append("\\*").append(Type).c_str(), &FileInfo);        
                
        if (lfile != -1L) {//查找存在符合要求的文件
            do {
                if (FileInfo.attrib != _A_SUBDIR) {//如果不是文件夹
                    images.at(i).push_back(sfile_r.assign(Path).append(subdir.at(i)).append("\\").append(FileInfo.name));
                    
                }
            } while (_findnext(lfile, &FileInfo) == 0);
        }
        else {//未查到符合要求的文件, 提示后退出
            _findclose(lfile);
            cout << "No such type files!" << endl;
            return false;
        }
    }
    
    if (images.at(0).size() != images.at(1).size()) {
        return false;
    }
    else if (images.at(0).size() == 0 || images.at(1).size() == 0) {
        return false;
    }

    int num = images.at(0).size();
    for (int i = 0; i < num; ++i) {
        FileNames.push_back(images.at(0).at(i));
        FileNames.push_back(images.at(1).at(i));
    }
    return true;
}

/*
打印文件名函数
*/
void PrintImagesNames(vector<string> ImageNames) {
    for (vector<string>::iterator st = ImageNames.begin(); st != ImageNames.end(); ++st) {
        cout << *st << endl;
    }
    return;
}

/*
双目相机立体标定函数
*/
static void
StereoCalib(
    const vector<string>& imagelist, //输入图片路径名称，左右相间
    Size boardSize, //角点矩阵尺寸
    float squareSize, //棋盘方格边长
    bool displayCorners = false, //是否显示角点并保存
    string CornersFound = "corners_found",
    bool useCalibrated = true, //是否使用库函数标定相机
    bool Rectify = true,//是否校正图像并保存
    string Rectified = "rectified"
)
{
    //左右图像是否数量匹配检验
    if (imagelist.size() % 2 != 0)
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    //如果图像太小则放大图像长宽比例，最大放大倍数为maxScale
    const int maxScale = 4;
    
    vector<vector<Point2f> > imagePoints[2];//存放图像角点的像素坐标
    vector<vector<Point3f> > objectPoints;//存放物点的三维坐标
    Size imageSize;//用于检查图像的大小是否一致的参数

    int i, k;//循环迭代使用参数
    int goodimagesnum;//有效的图像数量
    int nimages = (int)imagelist.size() / 2;//左右图像对的数量
    
    imagePoints[0].resize(nimages);//存放左相机图像角点的像素坐标
    imagePoints[1].resize(nimages);//存放右相机图像角点的像素坐标
    vector<string> goodImageList;//存放有效图像的路径

    
    vector<string> subdir{ "\\left","\\right" };//保存图片用的子目录
    //读取确实有效的图像
    for (i = goodimagesnum = 0; i < nimages; i++)//左右图像对循环
    {
        for (k = 0; k < 2; k++)//左右图像循环
        {
            //获取文件名
            const string& filename = imagelist[i * 2 + k];
            //按照灰度图读取图像
            Mat img = imread(filename, IMREAD_GRAYSCALE);
            //如果有一幅图像为空，结束该对图像的读取
            if (img.empty()) {
                break;
            }   
            
            if (imageSize == Size()) {//读取前一幅不为空的图像时，记录图像的大小
                imageSize = img.size();
            }   
            else if (img.size() != imageSize)//如果出现图像大小和前幅图像对大小不一致，则结束该对图像的读取
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }

            bool found = false;//表示是否找到角点的参数
            vector<Point2f>& corners = imagePoints[k][goodimagesnum];//临时存放找到的角点

            //如果图像太小则放大图像长宽比例，最大放大倍数为maxScale
            for (int scale = 1; scale <= maxScale; scale++)
            {
                Mat timg;
                if (scale == 1) {//放大倍数为1倍时数据为原图像
                    timg = img;
                }   
                else {//按照比例放大
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                }
                    
                found = findChessboardCorners(
                    timg, //输入图像数据
                    boardSize, //角点矩阵的尺寸
                    corners,//输出的角点坐标
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE
                );
                if (found)//如果已经找到角点，则按照已经放大的比例数除找到的角点的坐标，使角点坐标恢复到原图像时的坐标
                {
                    /*
                    亚像素精细化
                    因为角点并不一定在整数像素上
                    */
                    cornerSubPix(timg, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 20, 0.01));
                    if (scale > 1)
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
            }
            
            //如果需要显示找到的角点,则显示并保存
            if (displayCorners)
            {

                Mat cimg, cimg1;//保存图片和显示图片
                
                cvtColor(img, cimg, COLOR_GRAY2BGR);//转化为灰度图
                drawChessboardCorners(cimg, boardSize, corners, found);//绘制角点
                
                string tmpdir("");
                stringstream tmpss;
                tmpss.clear();
                tmpss << CornersFound << subdir.at(k) << "\\cornersfind_" << k << "_" << i << ".jpg";
                tmpss >> tmpdir;
                
                cout << tmpdir << endl;
                
                imwrite(tmpdir.c_str(), cimg);
                tmpdir.clear();

                double sf = 640. / MAX(img.rows, img.cols);//固定显示图片的长宽最高不能超过640毫米
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
                
                imshow("corners", cimg1);
                waitKey(1000);
                destroyWindow("corners");
            }
            
            if (!found) {//如果没有找到到就退出
                break;
            }
        }
        
        if (k == 2)
        {
            goodImageList.push_back(imagelist[i * 2]);
            goodImageList.push_back(imagelist[i * 2 + 1]);
            goodimagesnum++;

        }
    }
    //输出确实有效的图像的个数
    cout << goodimagesnum << " pairs have been successfully detected.\n";
    nimages = goodimagesnum;//更新图像对数量
    if (nimages < 2)//如果有效对图像数量不到2对，则无法完成标定，退出
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }


    //更新存放图像角点和三维物点坐标的大小
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for (i = 0; i < nimages; i++)
    {//设置物点的坐标
        for (goodimagesnum = 0; goodimagesnum < boardSize.height; goodimagesnum++)
            for (k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(k * squareSize, goodimagesnum * squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];//相机内参矩阵和畸变向量
    //初步得到相机的内参矩阵
    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    Mat R, T, E, F;//相机间变换的旋转矩阵、平移向量、本质矩阵、基本矩阵
    
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, R, T, E, F,
        CALIB_FIX_ASPECT_RATIO +
        CALIB_USE_INTRINSIC_GUESS +
        CALIB_SAME_FOCAL_LENGTH +
        CALIB_ZERO_TANGENT_DIST +
        CALIB_RATIONAL_MODEL,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    cout << "done with RMS error=" << rms << endl;
    cout << cameraMatrix[0] << endl;
    cout << distCoeffs[0] << endl;
    cout << cameraMatrix[1] << endl;
    cout << distCoeffs[1] << endl;
    cout << R << endl;
    cout << T << endl;
    cout << E << endl;
    cout << F << endl;
    


    
    /*
    检查双目相机标定的质量
    由于基础矩阵包含了所有输出的信息，我们可以通过核线约束来检查标定的质量m2^t*F*m1=0
    */
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];//核线
    for (i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();//左相机第i幅图包含的角点的数量
        Mat imgpt[2];
        //计算核线
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);//读取第i对图像
            
            //校正格点坐标

            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            //计算核线
            computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);

        }
        //计算一对图像所有角点
        for (goodimagesnum = 0; goodimagesnum < npt; goodimagesnum++)
        {
            double errij = //按照核线约束，errij应当为0
                fabs(
                imagePoints[0][i][goodimagesnum].x * lines[1][goodimagesnum][0]
                + imagePoints[0][i][goodimagesnum].y * lines[1][goodimagesnum][1]
                + lines[1][goodimagesnum][2])
                //为什么角点坐标和极线分别用作用图像的，因为左极线的方程是由右至左变换得到的
                + fabs(
                    imagePoints[1][i][goodimagesnum].x * lines[0][goodimagesnum][0]
                    + imagePoints[1][i][goodimagesnum].y * lines[0][goodimagesnum][1]
                    + lines[0][goodimagesnum][2]
                );
            err += errij;//统计所有图像所有点对在核线约束上的误差
        }
        npoints += npt;//统计所有图像的点对 数量
    }
    //输出一对点对的平均核线误差
    cout << "average epipolar err = " << err / npoints << endl;

    //保存内参矩阵和畸变向量
    ofstream fout("intrinsics.txt");
    //FileStorage fs("intrinsics.txt", FileStorage::WRITE);    
    if (fout.is_open())
    {
        fout << "left camera matrix:\n" << cameraMatrix[0] << "\n";
        fout << "left camera distort coefficient:\n" << distCoeffs[0] << "\n";
        fout << "left camera matrix:\n" << cameraMatrix[1] << "\n";
        fout << "right camera distort coefficient:\n" << distCoeffs[1] << "\n";
        fout.close();
    }
    else {
        cout << "Error: can not save the intrinsic parameters\n";
    }


    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    
    
    stereoRectify(
        cameraMatrix[0], //左相机内参
        distCoeffs[0],//左相机畸变向量
        cameraMatrix[1], //右相机内参
        distCoeffs[1],//右相机畸变向量
        imageSize, //图像大小
        R,//从左相机坐标系变换到右相机坐标系的旋转矩阵
        T,//从左相机到右相机的平移向量
        R1, //为使双相机核线平行，左相机坐标系需要作的旋转变换
        R2, //为使双相机核线平行，右相机坐标系需要作的旋转变换
        P1, //左相机图像二维点的齐次坐标到三维物体点坐标的投影变换
        P2, //右相机图像二维点的齐次坐标到三维物体点坐标的投影变换
        Q,//左相机的重投影矩阵
        CALIB_ZERO_DISPARITY, 
        1, 
        imageSize, 
        &validRoi[0], 
        &validRoi[1]
    );
    
    ofstream fs("extrinsics.txt");
    if (fs.is_open())
    {
        fs << "R" << endl;
        fs << R << endl;
        fs << "T" << endl;
        fs << T << endl;
        fs << "R1" << endl;
        fs << R1 << endl;
        fs << "R2" << endl;
        fs << R2 << endl;
        fs << "P1" << endl;
        fs << P1 << endl;
        fs << "P2" << endl;
        fs << P2 << endl;
        fs << "Q" << endl;
        fs << Q << endl;
        fs.close();
    }
    else {
        cout << "Error: can not save the extrinsic parameters\n";
    }
        

    //判断是左右放置的双相机系统还是上下放置的双相机系统
    //true表示上下放置
    //false表示左右放置
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    //如果不需要校正，则结束
    if (!Rectify) {
        return;
    }

    Mat rmap[2][2];
    //使用基于BOUGUET'S METHOD的库函数
    if (useCalibrated)
    {
        //那么已经标定完成
    }
    //使用HARTLEY'S METHOD
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv() * H1 * cameraMatrix[0];
        R2 = cameraMatrix[1].inv() * H2 * cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
        /*cout << P1 << endl;
        cout << P2 << endl;
        exit(0);*/
    }

    //校正查找映射表可以将原始图像和校正后的图像上的点一一对应起来
    //一一映射，大小为imageSize
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;//容纳显示图像对的矩阵
    double sf;
    int w, h;
    if (!isVerticalStereo)//如果左右双相机系统，则左右显示
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else//如果上下双相机系统，则上下显示
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    //逐一显示校正后的图像
    for (i = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            Mat img = imread(goodImageList[i * 2 + k], 0);//读入的图像数据
            Mat rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);//查表映射
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);//RGB显示

            //设定画布位置和大小
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if (useCalibrated)//用方框框处图像位置
            {
                Rect vroi(
                    cvRound(validRoi[k].x * sf), 
                    cvRound(validRoi[k].y * sf),
                    cvRound(validRoi[k].width * sf),
                    cvRound(validRoi[k].height * sf)
                );

                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
            }
        }

        //画线，每隔16像素画绿线
        if (!isVerticalStereo) {
            for (goodimagesnum = 0; goodimagesnum < canvas.rows; goodimagesnum += 16) {
                line(canvas, Point(0, goodimagesnum), Point(canvas.cols, goodimagesnum), Scalar(0, 255, 0), 1, 8);
            }   
        }   
        else {
            for (goodimagesnum = 0; goodimagesnum < canvas.cols; goodimagesnum += 16) {
                line(canvas, Point(goodimagesnum, 0), Point(goodimagesnum, canvas.rows), Scalar(0, 255, 0), 1, 8);
            }   
        }
            
        imshow("rectified", canvas);
        waitKey(1000);
        destroyWindow("rectified");
    }
}



int main(int argc, char** argv)
{
    
    string imagelistfn("images");//存放图片的目录
    string cornersfound("corners_found");//存放找到角点的图片的目录
    string rectified("rectified");//存放已经校正的图片的目录

    string ftype(".jpg");//获取文件的类型
    vector<string> imagelist;//存放读取的图片的路径
    bool ok = GetFile(imagelistfn, ftype, imagelist);
    //PrintImagesNames(imagelist);
    if (!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        exit(0);
    }

    
    Size boardSize = Size(9, 6);//标定板上每行、列的角点数
    float squareSize = 30;//实际测量得到的标定板上每个棋盘格的物理尺寸，单位mm
    bool displayCorners = true;
    bool useCalibrated = true;
    bool showRectified = true;
 
    StereoCalib(imagelist, boardSize, squareSize, true, cornersfound, true, showRectified, rectified);
    
    return 0;
}