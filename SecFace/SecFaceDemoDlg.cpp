
// ArcFaceDemoDlg.cpp : 实现文件
//
#include "stdafx.h"
#include <direct.h>
#include "resource.h"
#include "ArcFaceDemo.h"
#include "ArcFaceDemoDlg.h"
#include "afxdialogex.h"
#include <mysql.h>
#include <afx.h>
#include <afxwin.h>
#include <io.h>
#include <vector>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
//#include "immintrin.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <zstd.h>

#include <map>
#include <mutex>
#include <strmif.h>
#include <initguid.h>
#include <string>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <memory>
#include <limits>
#include <algorithm>
#include <numeric>
#include "seal/seal.h"
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <conio.h>

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "libmysql.lib")

using namespace std;
using namespace Gdiplus;
using namespace seal;

MYSQL* mysql = new MYSQL;

size_t number_n; // 有效数目个数
const double scale = pow(2.0, 40);// 放大因子

//Ciphertext* cloud[1000];//全局变量模拟云端

string MATCH_FACE_IMAGE, REGIST_FACE_IMAGE;

#define HOST "localhost"

#define USERNAME "root"

#define PASSWORD "Lzx2000919"

#define DATABASE "fdb"

#define PORT 3306

#define VIDEO_FRAME_DEFAULT_WIDTH 640
#define VIDEO_FRAME_DEFAULT_HEIGHT 480

#define FACE_FEATURE_SIZE 1032

#define THUMBNAIL_WIDTH  55
#define THUMBNAIL_HEIGHT  55
#define Threshold 0.23

#define VI_MAX_CAMERAS 20
DEFINE_GUID(CLSID_SystemDeviceEnum, 0x62be5d10, 0x60eb, 0x11d0, 0xbd, 0x3b, 0x00, 0xa0, 0xc9, 0x11, 0xce, 0x86);
DEFINE_GUID(CLSID_VideoInputDeviceCategory, 0x860bb310, 0x5d01, 0x11d0, 0xbd, 0x3b, 0x00, 0xa0, 0xc9, 0x11, 0xce, 0x86);
DEFINE_GUID(IID_ICreateDevEnum, 0x29840822, 0x5b84, 0x11d0, 0xbd, 0x3b, 0x00, 0xa0, 0xc9, 0x11, 0xce, 0x86);

#define SafeFree(p) { if ((p)) free(p); (p) = NULL; }
#define SafeArrayDelete(p) { if ((p)) delete [] (p); (p) = NULL; } 
#define SafeDelete(p) { if ((p)) delete (p); (p) = NULL; } 

mutex g_mutex;
vector<string> g_cameraName;
static int g_cameraNum = 0;
static int g_rgbCameraId = -1;
static int g_irCameraId = -1;
static float g_rgbLiveThreshold = 0.0;
static float g_irLiveThreshold = 0.0;





template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric< dlib::fc_no_bias<128, dlib::avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
	dlib::input_rgb_image_sized<150>
	>>>>>>>>>>>>;

unsigned long _stdcall RunLoadThumbnailThread(LPVOID lpParam);
unsigned long _stdcall RunFaceFeatureOperation(LPVOID lpParam);
unsigned long _stdcall RunFaceDetectOperation(LPVOID lpParam);
unsigned long _stdcall ClearFaceFeatureOperation(LPVOID lpParam);
Bitmap* IplImage2Bitmap(const IplImage* pIplImg);
IplImage* Bitmap2IplImage(Bitmap* pBitmap);
CBitmap* IplImage2CBitmap(const IplImage *img);
BOOL SetTextFont(CFont* font, int fontHeight, int fontWidth, string fontStyle);
int listDevices(vector<string>& list);			//获取摄像头
//读取配置文件
void ReadSetting(char* appID, char* sdkKey, char* activeKey, char* tag,
	char* rgbLiveThreshold, char* irLiveThreshold, char* rgbCameraId, char* irCameraId);

// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CArcFaceDemoDlg 对话框

CArcFaceDemoDlg::CArcFaceDemoDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CArcFaceDemoDlg::IDD, pParent),
	m_strEditThreshold(_T("")),
	m_curStaticImageFRSucceed(FALSE)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDI_ICON_ARCSOFT);
}

CArcFaceDemoDlg::~CArcFaceDemoDlg()
{

}

void CArcFaceDemoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIST_IMAGE, m_ImageListCtrl);
	DDX_Control(pDX, IDC_EDIT_LOG, m_editLog);
	DDX_Text(pDX, IDC_EDIT_THRESHOLD, m_strEditThreshold);
}

BEGIN_MESSAGE_MAP(CArcFaceDemoDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_REGISTER, &CArcFaceDemoDlg::OnBnClickedBtnRegister)
	ON_BN_CLICKED(IDC_BTN_RECOGNITION, &CArcFaceDemoDlg::OnBnClickedBtnRecognition)
	ON_BN_CLICKED(IDC_BTN_COMPARE, &CArcFaceDemoDlg::OnBnClickedBtnCompare)
	ON_BN_CLICKED(IDC_BTN_CLEAR, &CArcFaceDemoDlg::OnBnClickedBtnClear)
	ON_BN_CLICKED(IDC_BTN_ENCRYPT_ENV_INI, &CArcFaceDemoDlg::OnBnClickedBtnEncryptEnvIni)
	ON_BN_CLICKED(IDC_BTN_ENCRYPT, &CArcFaceDemoDlg::OnBnClickedBtnEncryptor)
	ON_BN_CLICKED(IDC_BTN_DECRYPT, &CArcFaceDemoDlg::OnBnClickedBtnDecryptor)
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_BTN_CAMERA, &CArcFaceDemoDlg::OnBnClickedBtnCamera)
	ON_EN_CHANGE(IDC_EDIT_THRESHOLD, &CArcFaceDemoDlg::OnEnChangeEditThreshold)
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_BTN_REGIST_ENCRYPT, &CArcFaceDemoDlg::OnBnClickedBtnRegistEncrypt)
END_MESSAGE_MAP()


// CArcFaceDemoDlg 消息处理程序
BOOL CArcFaceDemoDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO:  在此添加额外的初始化代码
	m_IconImageList.Create(THUMBNAIL_WIDTH,
		THUMBNAIL_HEIGHT,
		ILC_COLOR32,
		0,
		1);

	m_ImageListCtrl.SetImageList(&m_IconImageList, LVSIL_NORMAL);

	char tag[MAX_PATH] = "";
	char appID[MAX_PATH] = "";
	char  sdkKey[MAX_PATH] = "";
	char  activeKey[MAX_PATH] = "";
	char rgbLiveThreshold[MAX_PATH] = "";
	char irLiveThreshold[MAX_PATH] = "";
	char rgbCameraId[MAX_PATH] = "";
	char irCameraId[MAX_PATH] = "";

	ReadSetting(appID, sdkKey, activeKey, tag, rgbLiveThreshold, irLiveThreshold, rgbCameraId, irCameraId);

	g_rgbCameraId = atoi(rgbCameraId);
	g_irCameraId = atoi(irCameraId);
	g_rgbLiveThreshold = atof(rgbLiveThreshold);
	g_irLiveThreshold = atof(irLiveThreshold);

	CString resStr = "";

	MRESULT faceRes = m_imageFaceEngine.ActiveSDK(appID, sdkKey, activeKey);
	//resStr.Format("激活结果: %d\n", faceRes);
	//EditOut(resStr, TRUE);

	//获取激活文件信息
	ASF_ActiveFileInfo activeFileInfo = { 0 };
	m_imageFaceEngine.GetActiveFileInfo(activeFileInfo);

	if (faceRes == MOK)
	{
		resStr = "";
		faceRes = m_imageFaceEngine.InitEngine(ASF_DETECT_MODE_IMAGE);//Image
		//resStr.Format("IMAGE模式下初始化结果: %d", faceRes);
		//EditOut(resStr, TRUE);

		resStr = "";
		faceRes = m_videoFaceEngine.InitEngine(ASF_DETECT_MODE_VIDEO);//Video
		//resStr.Format("VIDEO模式下初始化结果: %d", faceRes);
		//EditOut(resStr, TRUE);
	}

	//设置输入框位数
	((CEdit*)GetDlgItem(IDC_EDIT_THRESHOLD))->SetLimitText(4);
	m_strEditThreshold.Format("%.2f", Threshold);
	UpdateData(FALSE);

	GetDlgItem(IDC_STATIC_VIEW)->GetWindowRect(&m_windowViewRect);

	//人脸库按钮置灰
	GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_CAMERA)->EnableWindow(FALSE);

	//编辑阈值置灰
	GetDlgItem(IDC_EDIT_THRESHOLD)->EnableWindow(FALSE);

	//加密解密模块置灰
	GetDlgItem(IDC_BTN_ENCRYPT)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_DECRYPT)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_REGIST_ENCRYPT)->EnableWindow(FALSE);

	m_curStaticImageFeature.featureSize = FACE_FEATURE_SIZE;
	m_curStaticImageFeature.feature = (MByte *)malloc(m_curStaticImageFeature.featureSize * sizeof(MByte));


	m_Font = new CFont;

	SetTextFont(m_Font, 20, 20, "微软雅黑");

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CArcFaceDemoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。


void CArcFaceDemoDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		if (m_videoOpened)
		{
			lock_guard<mutex> lock(g_mutex);
			//文字显示框
			CRect rect(m_curFaceInfo.faceRect.left - 10, m_curFaceInfo.faceRect.top - 50,
				m_curFaceInfo.faceRect.right, m_curFaceInfo.faceRect.bottom);
			IplDrawToHDC(TRUE, m_curVideoImage, rect, IDC_STATIC_VIEW);
		}
		else
		{
			if (m_curStaticImage)
			{
				CRect rect((int)m_curStringShowPosition.X + 10, (int)m_curStringShowPosition.Y + 10, 40, 40);
				IplDrawToHDC(FALSE, m_curStaticImage, rect, IDC_STATIC_VIEW);
			}
		}

		CDialogEx::OnPaint();
	}


}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CArcFaceDemoDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

BOOL IsImageGDIPLUSValid(CString filePath)
{
	Bitmap image(filePath.AllocSysString());

	if (image.GetFlags() == ImageFlagsNone)
		return FALSE;
	else
		return TRUE;
}

//加载缩略图片
void CArcFaceDemoDlg::LoadThumbnailImages()
{
	m_bLoadIconThreadRunning = TRUE;

	GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_CAMERA)->EnableWindow(FALSE);

	m_hLoadIconThread = CreateThread(
		NULL,
		0,
		RunLoadThumbnailThread,
		this,
		0,
		&m_dwLoadIconThreadID);

	if (m_hLoadIconThread == NULL)
	{
		::CloseHandle(m_hLoadIconThread);
	}
}

void CArcFaceDemoDlg::ShowPic()//形参可以为所给的图片对象的指针，这里为了通用性省去了形参
{
	const string path = REGIST_FACE_IMAGE;
	float cx, cy, dx, dy, k, t;//跟控件的宽和高以及图片宽和高有关的参数
	CRect rect;//用于获取图片控件的宽和高
	CImage q;//为cimage图片类创建一个对象
	q.Load(path.c_str());//构造函数的形参是所加载图片的路径 
	cx = q.GetWidth();
	cy = q.GetHeight();//获取图片的宽 高
	k = cy / cx;//获得图片的宽高比
	//用目的头文件规范不同库属同名函数可防止张冠李戴
	CWnd* pWnd = GetDlgItem(IDC_STATIC_REGIST_PIC);
	pWnd->GetClientRect(&rect);//获取Picture Control控件的客户区
	dx = rect.Width();
	dy = rect.Height();//获得控件的宽高比
	t = dy / dx;//获得控件的宽高比
	if (k >= t)
	{
		rect.right = floor(rect.bottom / k);
		rect.left = (dx - rect.right) / 2;
		rect.right = floor(rect.bottom / k) + (dx - rect.right) / 2;
	}
	else
	{
		rect.bottom = floor(k * rect.right);
		rect.top = (dy - rect.bottom) / 2;
		rect.bottom = floor(k * rect.right) + (dy - rect.bottom) / 2;
	}
	//相关的计算为了让图片在绘图区居中按比例显示，原理很好懂，如果图片很宽但是不高，就上下留有空白区；
	//如果图片很高而不宽就左右留有空白区，并且保持两边空白区一样大
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture control的DC
	int ModeOld = SetStretchBltMode(pDc->m_hDC, STRETCH_HALFTONE);//设置指定设备环境中的位图拉伸模式
	//GetDlgItem(Pic)->ShowWindow(FALSE);
	//GetDlgItem(Pic)->ShowWindow(TRUE);
	q.StretchBlt(pDc->m_hDC, rect, SRCCOPY);//显示函数
	SetStretchBltMode(pDc->m_hDC, ModeOld);
	ReleaseDC(pDc);//释放指针空间
}

void CArcFaceDemoDlg::OnBnClickedBtnRegister()
{
	// TODO:  在此添加控件通知处理程序代码
	/*
	GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(FALSE);
	m_folderPath = SelectFolder();
	if (m_folderPath == "")
	{
		GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(TRUE);
		return;
	}*/
	CFileDialog fileDlg(TRUE, _T("bmp"), NULL, 0, _T("Picture Files|*.jpg;*.jpeg;*.png;*.bmp;||"), NULL);
	fileDlg.DoModal();
	CString strFilePath;
	strFilePath = fileDlg.GetPathName();

	if (strFilePath == _T(""))
		return;

	REGIST_FACE_IMAGE = strFilePath;
	//LoadThumbnailImages();
	ShowPic();
}

unsigned long _stdcall RunLoadThumbnailThread(LPVOID lpParam)
{
	CArcFaceDemoDlg* dialog = (CArcFaceDemoDlg*)(lpParam);

	if (dialog == nullptr)
	{
		dialog->m_bLoadIconThreadRunning = FALSE;
		return 1;
	}

	if (dialog->m_folderPath == "")
	{
		dialog->m_bLoadIconThreadRunning = FALSE;
		return 1;
	}

	int iExistFeatureSize = (int)dialog->m_featuresVec.size();

	CString resStr;
	resStr.Format("开始注册人脸库");
	dialog->EditOut(resStr, TRUE);

	CFileFind finder;

	CString m_strCurrentDirectory(dialog->m_folderPath);
	CString strWildCard(m_strCurrentDirectory);
	vector<CString> m_vFileName;
	strWildCard += "\\*.*";

	BOOL bWorking = finder.FindFile(strWildCard);

	while (bWorking)
	{
		bWorking = finder.FindNextFile();

		if (finder.IsDots() || finder.IsDirectory())
		{
			continue;
		}

		CString filePath = finder.GetFileName();

		if (IsImageGDIPLUSValid(m_strCurrentDirectory + _T("\\") + filePath))//是否是图片
		{
			m_vFileName.push_back(filePath);
		}
	}

	resStr.Format("已选择图片张数: %d", m_vFileName.size());
	dialog->EditOut(resStr, TRUE);

	dialog->GetDlgItem(IDC_BTN_CLEAR)->EnableWindow(FALSE);
	dialog->GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(FALSE);
	dialog->GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(FALSE);

	if (dialog->GetDlgItem(IDC_BTN_RECOGNITION)->IsWindowEnabled())
	{
		dialog->GetDlgItem(IDC_BTN_RECOGNITION)->EnableWindow(FALSE);
	}

	vector<CString>::const_iterator iter;

	int actualIndex = iExistFeatureSize;

	for (iter = m_vFileName.begin();
		iter != m_vFileName.end();
		iter++)
	{
		if (!dialog->m_bLoadIconThreadRunning)
		{
			dialog->m_bLoadIconThreadRunning = FALSE;
			return 1;
		}

		CString imagePath;
		imagePath.Empty();
		imagePath.Format("%s\\%s", m_strCurrentDirectory, *iter);

		USES_CONVERSION;
		IplImage* originImage = cvLoadImage(T2A(imagePath.GetBuffer(0)));
		imagePath.ReleaseBuffer();

		if (!originImage)
		{
			cvReleaseImage(&originImage);
			continue;
		}

		//FD 
		ASF_SingleFaceInfo faceInfo = { 0 };
		MRESULT detectRes = dialog->m_imageFaceEngine.PreDetectFace(originImage, faceInfo, true);
		if (MOK != detectRes)
		{
			cvReleaseImage(&originImage);
			continue;
		}

		//FR
		ASF_FaceFeature faceFeature = { 0 };
		faceFeature.featureSize = FACE_FEATURE_SIZE;
		faceFeature.feature = (MByte *)malloc(faceFeature.featureSize * sizeof(MByte));
		detectRes = dialog->m_imageFaceEngine.PreExtractFeature(originImage, faceFeature, faceInfo);

		if (MOK != detectRes)
		{
			free(faceFeature.feature);
			cvReleaseImage(&originImage);
			continue;
		}

		Bitmap* image = IplImage2Bitmap(originImage);
		dialog->m_featuresVec.push_back(faceFeature);

		//计算缩略图显示位置
		int sourceWidth = image->GetWidth();
		int sourceHeight = image->GetHeight();

		int destX = 0;
		int destY = 0;

		float nPercent = 0;
		float nPercentW = ((float)THUMBNAIL_WIDTH / (float)sourceWidth);;
		float nPercentH = ((float)THUMBNAIL_HEIGHT / (float)sourceHeight);

		if (nPercentH < nPercentW)
		{
			nPercent = nPercentH;
			destX = (int)((THUMBNAIL_WIDTH - (sourceWidth * nPercent)) / 2);
		}
		else
		{
			nPercent = nPercentW;
			destY = (int)((THUMBNAIL_HEIGHT - (sourceHeight * nPercent)) / 2);
		}

		int destWidth = (int)(sourceWidth * nPercent);
		int destHeight = (int)(sourceHeight * nPercent);

		dialog->m_ImageListCtrl.InsertItem(actualIndex, to_string(actualIndex + 1).c_str(), actualIndex);

		actualIndex++;

		Bitmap* bmPhoto = new Bitmap(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT, PixelFormat24bppRGB);

		bmPhoto->SetResolution(image->GetHorizontalResolution(), image->GetVerticalResolution());

		Graphics *grPhoto = Graphics::FromImage(bmPhoto);
		Gdiplus::Color colorW(255, 255, 255, 255);
		grPhoto->Clear(colorW);
		grPhoto->SetInterpolationMode(InterpolationModeHighQualityBicubic);
		grPhoto->DrawImage(image, Gdiplus::Rect(destX, destY, destWidth, destHeight));

		HBITMAP hbmReturn = NULL;
		bmPhoto->GetHBITMAP(colorW, &hbmReturn);

		CBitmap Bmp1;
		Bmp1.Attach(hbmReturn);

		dialog->m_IconImageList.Add(&Bmp1, RGB(0, 0, 0));

		delete grPhoto;
		delete bmPhoto;
		Bmp1.Detach();
		DeleteObject(hbmReturn);

		dialog->m_ImageListCtrl.RedrawItems(actualIndex, actualIndex);

		//重绘
		if (actualIndex % 10 == 0)
		{
			dialog->m_ImageListCtrl.SetRedraw(TRUE);
			dialog->m_ImageListCtrl.Invalidate();
			dialog->m_ImageListCtrl.EnsureVisible(actualIndex - 1, FALSE);
		}

		cvReleaseImage(&originImage);
		delete image;
	}

	resStr.Format("成功注册图片张数: %d", actualIndex - iExistFeatureSize);
	dialog->EditOut(resStr, TRUE);

	dialog->m_ImageListCtrl.SetRedraw(TRUE);
	dialog->m_ImageListCtrl.Invalidate();
	dialog->m_ImageListCtrl.EnsureVisible(actualIndex - 1, FALSE);

	if (dialog->m_featuresVec.empty())
	{

	}

	//注册人脸库后按钮重置
	dialog->GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(TRUE);

	if (!dialog->m_videoOpened)
	{
		dialog->GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(TRUE);
		dialog->GetDlgItem(IDC_BTN_RECOGNITION)->EnableWindow(TRUE);
	}
	else
	{
		dialog->GetDlgItem(IDC_BTN_RECOGNITION)->EnableWindow(FALSE);
	}

	dialog->GetDlgItem(IDC_BTN_CAMERA)->EnableWindow(TRUE);
	dialog->GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(TRUE);
	dialog->GetDlgItem(IDC_BTN_CLEAR)->EnableWindow(TRUE);

	dialog->m_bLoadIconThreadRunning = FALSE;

	return 0;
}

//选择文件夹
CString CArcFaceDemoDlg::SelectFolder()
{
	TCHAR           szFolderPath[MAX_PATH] = { 0 };
	CString         strFolderPath = TEXT("");

	BROWSEINFO      sInfo;
	::ZeroMemory(&sInfo, sizeof(BROWSEINFO));
	sInfo.pidlRoot = 0;
	sInfo.lpszTitle = _T("请选择一个文件夹：");
	sInfo.ulFlags = BIF_DONTGOBELOWDOMAIN | BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX;
	sInfo.lpfn = NULL;

	// 显示文件夹选择对话框  
	LPITEMIDLIST lpidlBrowse = ::SHBrowseForFolder(&sInfo);
	if (lpidlBrowse != NULL)
	{
		// 取得文件夹名  
		if (::SHGetPathFromIDList(lpidlBrowse, szFolderPath))
		{
			strFolderPath = szFolderPath;
		}
	}
	if (lpidlBrowse != NULL)
	{
		::CoTaskMemFree(lpidlBrowse);
	}

	return strFolderPath;
}

void CArcFaceDemoDlg::OnBnClickedBtnRecognition()
{
	// TODO:  在此添加控件通知处理程序代码

	CFileDialog fileDlg(TRUE, _T("bmp"), NULL, 0, _T("Picture Files|*.jpg;*.jpeg;*.png;*.bmp;||"), NULL);
	fileDlg.DoModal();
	CString strFilePath;
	strFilePath = fileDlg.GetPathName();

	if (strFilePath == _T(""))
		return;

	MATCH_FACE_IMAGE = strFilePath;

	USES_CONVERSION;
	IplImage* image = cvLoadImage(T2A(strFilePath.GetBuffer(0)));
	strFilePath.ReleaseBuffer();
	if (!image)
	{
		cvReleaseImage(&image);
		return;
	}

	if (m_curStaticImage)
	{
		cvReleaseImage(&m_curStaticImage);
		m_curStaticImage = NULL;
	}

	m_curStaticImage = cvCloneImage(image);
	cvReleaseImage(&image);

	StaticImageFaceOp(m_curStaticImage);

	GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(TRUE);
}

MRESULT CArcFaceDemoDlg::StaticImageFaceOp(IplImage* image)
{
	Gdiplus::Rect showRect;
	CalculateShowPositon(image, showRect);
	m_curImageShowRect = showRect;
	//FD
	ASF_SingleFaceInfo faceInfo = { 0 };
	MRESULT detectRes = m_imageFaceEngine.PreDetectFace(image, faceInfo, true);

	//初始化
	m_curStaticShowAgeGenderString = "";
	m_curStaticShowCmpString = "";

	m_curFaceShowRect = Rect(0, 0, 0, 0);

	SendMessage(WM_PAINT);

	if (MOK == detectRes)
	{
		//show rect
		int n_top = showRect.Height*faceInfo.faceRect.top / image->height;
		int n_bottom = showRect.Height*faceInfo.faceRect.bottom / image->height;
		int n_left = showRect.Width*faceInfo.faceRect.left / image->width;
		int n_right = showRect.Width*faceInfo.faceRect.right / image->width;

		m_curFaceShowRect.X = n_left + showRect.X;
		m_curFaceShowRect.Y = n_top + showRect.Y;
		m_curFaceShowRect.Width = n_right - n_left;
		m_curFaceShowRect.Height = n_bottom - n_top;

		//显示文字在图片左上角
		m_curStringShowPosition.X = (REAL)(showRect.X);
		m_curStringShowPosition.Y = (REAL)(showRect.Y);

		//age gender
		ASF_MultiFaceInfo multiFaceInfo = { 0 };
		multiFaceInfo.faceOrient = (MInt32*)malloc(sizeof(MInt32));
		multiFaceInfo.faceRect = (MRECT*)malloc(sizeof(MRECT));

		multiFaceInfo.faceNum = 1;
		multiFaceInfo.faceOrient[0] = faceInfo.faceOrient;
		multiFaceInfo.faceRect[0] = faceInfo.faceRect;

		ASF_AgeInfo ageInfo = { 0 };
		ASF_GenderInfo genderInfo = { 0 };
		ASF_Face3DAngle angleInfo = { 0 };
		ASF_LivenessInfo liveNessInfo = { 0 };

		//age 、gender 、3d angle 信息
		detectRes = m_imageFaceEngine.FaceASFProcess(multiFaceInfo, image,
			ageInfo, genderInfo, angleInfo, liveNessInfo);

		if (MOK == detectRes)
		{
			CString showStr;
			showStr.Format("年龄:%d,性别:%s,活体:%s", ageInfo.ageArray[0], genderInfo.genderArray[0] == 0 ? "男" : "女",
				liveNessInfo.isLive[0] == 1 ? "是" : "否");
			m_curStaticShowAgeGenderString = showStr;
		}
		else
		{
			m_curStaticShowAgeGenderString = "";
		}

		SendMessage(WM_PAINT);

		free(multiFaceInfo.faceRect);
		free(multiFaceInfo.faceOrient);

		//FR
		detectRes = m_imageFaceEngine.PreExtractFeature(image, m_curStaticImageFeature, faceInfo);

		if (MOK == detectRes)
		{
			m_curStaticImageFRSucceed = TRUE;
		}
		else//提取特征不成功
		{
			m_curStaticImageFRSucceed = FALSE;
			CString resStr;
			resStr.Format("特征提取失败");
			EditOut(resStr, TRUE);
			return -1;
		}
		return MOK;
	}
	else
	{
		m_curStaticImageFRSucceed = FALSE;

		CString resStr;
		resStr.Format("未检测到人脸");
		EditOut(resStr, TRUE);
		return -1;
	}
}

void CArcFaceDemoDlg::EditOut(CString str, bool add_endl)
{
	if (add_endl)
		str += "\r\n";
	int iLen = m_editLog.GetWindowTextLength();
	m_editLog.SetSel(iLen, iLen, TRUE);
	m_editLog.ReplaceSel(str, FALSE);
}

IplImage* Bitmap2IplImage(Bitmap* pBitmap)
{
	if (!pBitmap)
		return NULL;

	int w = pBitmap->GetWidth();
	int h = pBitmap->GetHeight();

	BitmapData bmpData;
	Gdiplus::Rect rect(0, 0, w, h);
	pBitmap->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bmpData);
	BYTE* temp = (bmpData.Stride > 0) ? ((BYTE*)bmpData.Scan0) : ((BYTE*)bmpData.Scan0 + bmpData.Stride*(h - 1));

	IplImage* pIplImg = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
	if (!pIplImg)
	{
		pBitmap->UnlockBits(&bmpData);
		return NULL;
	}

	memcpy(pIplImg->imageData, temp, abs(bmpData.Stride)*bmpData.Height);
	pBitmap->UnlockBits(&bmpData);

	//判断Top-Down or Bottom-Up
	if (bmpData.Stride < 0)
		cvFlip(pIplImg, pIplImg);

	return pIplImg;
}

// pBitmap 同样需要外部释放
Bitmap* IplImage2Bitmap(const IplImage* pIplImg)
{
	if (!pIplImg)
		return NULL;

	Bitmap *pBitmap = new Bitmap(pIplImg->width, pIplImg->height, PixelFormat24bppRGB);
	if (!pBitmap)
		return NULL;

	BitmapData bmpData;
	Gdiplus::Rect rect(0, 0, pIplImg->width, pIplImg->height);
	pBitmap->LockBits(&rect, ImageLockModeWrite, PixelFormat24bppRGB, &bmpData);
	//BYTE *pByte = (BYTE*)bmpData.Scan0;

	if (pIplImg->widthStep == bmpData.Stride) //likely
		memcpy(bmpData.Scan0, pIplImg->imageDataOrigin, pIplImg->imageSize);

	pBitmap->UnlockBits(&bmpData);
	return pBitmap;
}


void CArcFaceDemoDlg::OnBnClickedBtnClear()
{
	// TODO: 在此添加控件通知处理程序代码

	if (m_videoOpened)
	{
		AfxMessageBox(_T("请先关闭摄像头！"));
		return;
	}

	// 注册人脸库按钮置灰
	GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_CAMERA)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(FALSE);
	GetDlgItem(IDC_BTN_CLEAR)->EnableWindow(FALSE);

	//清理原有的图片以及特征
	ClearRegisterImages();

	GetDlgItem(IDC_BTN_REGISTER)->EnableWindow(TRUE);
	GetDlgItem(IDC_BTN_CAMERA)->EnableWindow(TRUE);

}

BOOL CArcFaceDemoDlg::TerminateLoadThread()
{
	while (m_bLoadIconThreadRunning)
	{
		MSG message;
		while (::PeekMessage(&message, NULL, 0, 0, PM_REMOVE))
		{
			::TranslateMessage(&message);
			::DispatchMessage(&message);
		}
	}

	::CloseHandle(m_hLoadIconThread);

	return TRUE;
}

BOOL CArcFaceDemoDlg::ClearRegisterImages()
{
	if (m_bLoadIconThreadRunning)
	{
		TerminateLoadThread();
	}
	else
	{
		m_bClearFeatureThreadRunning = TRUE;

		m_hClearFeatureThread = CreateThread(
			NULL,
			0,
			ClearFaceFeatureOperation,//
			this,
			0,
			&m_dwClearFeatureThreadID);

		if (m_dwClearFeatureThreadID == NULL)
		{
			::CloseHandle(m_hClearFeatureThread);
		}
	}
	return 0;
}

BOOL CArcFaceDemoDlg::CalculateShowPositon(IplImage*curSelectImage, Gdiplus::Rect& showRect)
{
	//计算实际显示宽高
	int actualWidth = 0;
	int actualHeight = 0;

	int imageWidth = curSelectImage->width;
	int imageHeight = curSelectImage->height;

	int windowWidth = m_windowViewRect.Width();
	int windowHeight = m_windowViewRect.Height();

	int paddingLeft = 0;
	int paddingTop = 0;

	//以宽为基准的高
	actualHeight = windowWidth*imageHeight / imageWidth;
	if (actualHeight > windowHeight)
	{
		//以高为基准的宽
		actualWidth = windowHeight*imageWidth / imageHeight;
		actualHeight = windowHeight;
	}
	else
	{
		actualWidth = windowWidth;
	}

	paddingLeft = (windowWidth - actualWidth) / 2;
	paddingTop = (windowHeight - actualHeight) / 2;

	showRect.X = paddingLeft;
	showRect.Y = paddingTop;
	showRect.Width = actualWidth;
	showRect.Height = actualHeight;

	return 0;
}

void CArcFaceDemoDlg::OnDestroy()
{
	CDialogEx::OnDestroy();
}

void CArcFaceDemoDlg::OnBnClickedBtnCamera()
{
	// TODO: 在此添加控件通知处理程序代码

	CString btnLabel;

	GetDlgItem(IDC_BTN_CAMERA)->GetWindowText(btnLabel);

	//获取摄像头数量以及名称
	g_cameraNum = listDevices(g_cameraName);

	//防止太频繁点击按钮
	Sleep(3000);

	if (btnLabel == "启用摄像头")
	{

		GetDlgItem(IDC_EDIT_THRESHOLD)->EnableWindow(TRUE);
		GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(FALSE);
		GetDlgItem(IDC_BTN_RECOGNITION)->EnableWindow(FALSE);
		GetDlgItem(IDC_BTN_CAMERA)->SetWindowText("关闭摄像头");

		//FD 线程
		m_hFDThread = CreateThread(
			NULL,
			0,
			RunFaceDetectOperation,
			this,
			0,
			&m_dwFDThreadID);

		if (m_hFDThread == NULL)
		{
			::CloseHandle(m_hFDThread);
		}

		m_bFDThreadRunning = TRUE;

		//FR 线程
		m_hFRThread = CreateThread(
			NULL,
			0,
			RunFaceFeatureOperation,
			this,
			0,
			&m_dwFRThreadID);

		if (m_hFRThread == NULL)
		{
			::CloseHandle(m_hFRThread);
		}
	}
	else
	{
		GetDlgItem(IDC_BTN_COMPARE)->EnableWindow(TRUE);
		GetDlgItem(IDC_EDIT_THRESHOLD)->EnableWindow(FALSE);
		GetDlgItem(IDC_BTN_RECOGNITION)->EnableWindow(TRUE);

		//将之前存储的信息清除
		m_curFaceInfo = { 0 };
		m_curVideoShowString = "";
		{
			lock_guard<mutex> lock(g_mutex);
			if (m_curVideoImage)
			{
				cvReleaseImage(&m_curVideoImage);
				m_curVideoImage = NULL;
			}

			if (m_curIrVideoImage)
			{
				cvReleaseImage(&m_curIrVideoImage);
				m_curIrVideoImage = NULL;
			}
		}

		m_dataValid = false;
		m_videoOpened = false;

		Sleep(600);

		ClearShowWindow();

		if (m_hFDThread == NULL)
		{
			BOOL res = ::CloseHandle(m_hFDThread);
			if (!res)
			{
				GetLastError();
			}
		}

		m_bFDThreadRunning = FALSE;

		if (m_hFRThread == NULL)
		{
			::CloseHandle(m_hFRThread);
		}

		GetDlgItem(IDC_BTN_CAMERA)->SetWindowText("启用摄像头");
	}
}

unsigned long _stdcall RunFaceDetectOperation(LPVOID lpParam)
{
	CArcFaceDemoDlg* dialog = (CArcFaceDemoDlg*)(lpParam);

	if (dialog == nullptr)
	{
		return 1;
	}

	cv::Mat irFrame;
	cv::VideoCapture irCapture;

	cv::Mat rgbFrame;
	cv::VideoCapture rgbCapture;
	if (g_cameraNum == 2)
	{
		if (!irCapture.isOpened())
		{
			if (rgbCapture.open(g_rgbCameraId) && irCapture.open(g_irCameraId))
				dialog->m_videoOpened = true;
		}

		if (!(rgbCapture.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_DEFAULT_WIDTH) &&
			rgbCapture.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_DEFAULT_HEIGHT)))
		{
			AfxMessageBox(_T("RGB摄像头初始化失败！"));
			return 1;
		}

		if (!(irCapture.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_DEFAULT_WIDTH) &&
			irCapture.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_DEFAULT_HEIGHT)))
		{
			AfxMessageBox(_T("IR摄像头初始化失败！"));
			return 1;
		}
	}
	else if (g_cameraNum == 1)
	{
		if (!rgbCapture.isOpened())
		{
			bool res = rgbCapture.open(0);
			if (res)
				dialog->m_videoOpened = true;
		}

		if (!(rgbCapture.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_DEFAULT_WIDTH) &&
			rgbCapture.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_DEFAULT_HEIGHT)))
		{
			AfxMessageBox(_T("RGB摄像头初始化失败！"));
			return 1;
		}
	}
	else
	{
		AfxMessageBox(_T("摄像头数量不支持！"));
		return 1;
	}

	while (dialog->m_videoOpened)
	{
		if (g_cameraNum == 2)
		{
			irCapture >> irFrame;

			rgbCapture >> rgbFrame;

			ASF_SingleFaceInfo faceInfo = { 0 };

			IplImage rgbImage(rgbFrame);
			IplImage irImage(irFrame);

			MRESULT detectRes = dialog->m_videoFaceEngine.PreDetectFace(&rgbImage, faceInfo, true);
			if (MOK == detectRes)
			{
				cvRectangle(&rgbImage, cvPoint(faceInfo.faceRect.left, faceInfo.faceRect.top),
					cvPoint(faceInfo.faceRect.right, faceInfo.faceRect.bottom), cvScalar(0, 0, 255), 2);

				cvRectangle(&irImage, cvPoint(faceInfo.faceRect.left, faceInfo.faceRect.top),
					cvPoint(faceInfo.faceRect.right, faceInfo.faceRect.bottom), cvScalar(0, 0, 255), 2);

				dialog->m_curFaceInfo = faceInfo;
				dialog->m_dataValid = true;
			}
			else
			{
				//没有人脸不要显示信息
				dialog->m_curVideoShowString = "";
				dialog->m_curIRVideoShowString = "";
				dialog->m_dataValid = false;
			}

			ASF_SingleFaceInfo irFaceInfo = { 0 };
			MRESULT irRes = dialog->m_videoFaceEngine.PreDetectFace(&irImage, irFaceInfo, false);
			if (irRes == MOK)
			{
				if (abs(faceInfo.faceRect.left - irFaceInfo.faceRect.left) < 20 &&
					abs(faceInfo.faceRect.top - irFaceInfo.faceRect.top) < 20 &&
					abs(faceInfo.faceRect.right - irFaceInfo.faceRect.right) < 20 &&
					abs(faceInfo.faceRect.bottom - irFaceInfo.faceRect.bottom) < 20)
				{
					dialog->m_irDataValid = true;
				}
				else
				{
					dialog->m_irDataValid = false;
				}
			}
			else
			{
				dialog->m_irDataValid = false;
			}

			//重新拷贝
			{
				lock_guard<mutex> lock(g_mutex);
				cvReleaseImage(&dialog->m_curVideoImage);
				dialog->m_curVideoImage = cvCloneImage(&rgbImage);

				cvReleaseImage(&dialog->m_curIrVideoImage);
				dialog->m_curIrVideoImage = cvCloneImage(&irImage);
			}
		}
		else if (g_cameraNum == 1)
		{
			rgbCapture >> rgbFrame;

			ASF_SingleFaceInfo faceInfo = { 0 };

			IplImage rgbImage(rgbFrame);

			MRESULT detectRes = dialog->m_videoFaceEngine.PreDetectFace(&rgbImage, faceInfo, true);
			if (MOK == detectRes)
			{
				cvRectangle(&rgbImage, cvPoint(faceInfo.faceRect.left, faceInfo.faceRect.top),
					cvPoint(faceInfo.faceRect.right, faceInfo.faceRect.bottom), cvScalar(0, 0, 255), 2);

				dialog->m_curFaceInfo = faceInfo;
				dialog->m_dataValid = true;
			}
			else
			{
				//没有人脸不要显示信息
				dialog->m_curVideoShowString = "";
				dialog->m_dataValid = false;
			}


			//重新拷贝
			{
				lock_guard<mutex> lock(g_mutex);
				cvReleaseImage(&dialog->m_curVideoImage);
				dialog->m_curVideoImage = cvCloneImage(&rgbImage);
			}
		}
		else
		{
			AfxMessageBox(_T("摄像头数量不支持！"));
		}
		
		dialog->SendMessage(WM_PAINT);
	}

	rgbCapture.release();
	irCapture.release();

	return 0;
}

unsigned long _stdcall RunFaceFeatureOperation(LPVOID lpParam)
{
	CArcFaceDemoDlg* dialog = (CArcFaceDemoDlg*)(lpParam);

	if (dialog == nullptr)
	{
		return 1;
	}

	//设置活体检测阈值，sdk内部默认RGB:0.5 IR:0.7,可选择是否调用该接口
	dialog->m_imageFaceEngine.SetLivenessThreshold(g_rgbLiveThreshold, g_irLiveThreshold);

	//初始化特征
	ASF_FaceFeature faceFeature = { 0 };
	faceFeature.featureSize = FACE_FEATURE_SIZE;
	faceFeature.feature = (MByte *)malloc(faceFeature.featureSize * sizeof(MByte));

	ASF_MultiFaceInfo multiFaceInfo = { 0 };
	multiFaceInfo.faceOrient = (MInt32*)malloc(sizeof(MInt32));
	multiFaceInfo.faceRect = (MRECT*)malloc(sizeof(MRECT));

	while (dialog->m_bFDThreadRunning)
	{
		if (dialog->m_bLoadIconThreadRunning ||
			dialog->m_bClearFeatureThreadRunning)
		{
			//加载和清除注册库的过程中 不要显示信息
			dialog->m_curVideoShowString = "";
			continue;
		}

		if (!dialog->m_dataValid)
		{
			continue;
		}

		//先拷贝一份，防止读写冲突
		IplImage* tempImage = NULL;
		{
			lock_guard<mutex> lock(g_mutex);
			if (dialog->m_curVideoImage)
			{
				tempImage = cvCloneImage(dialog->m_curVideoImage);
			}
		}

		//发送一份到活体
		multiFaceInfo.faceNum = 1;
		multiFaceInfo.faceOrient[0] = dialog->m_curFaceInfo.faceOrient;
		multiFaceInfo.faceRect[0] = dialog->m_curFaceInfo.faceRect;

		ASF_AgeInfo ageInfo = { 0 };
		ASF_GenderInfo genderInfo = { 0 };
		ASF_Face3DAngle angleInfo = { 0 };
		ASF_LivenessInfo liveNessInfo = { 0 };

		//IR活体检测
		bool isIRAlive = false;
		if (g_cameraNum == 2)
		{
			IplImage* tempIRImage = NULL;
			lock_guard<mutex> lock(g_mutex);
			{
				if (dialog->m_curIrVideoImage)
				{
					tempIRImage = cvCloneImage(dialog->m_curIrVideoImage);
				}
			}
			
			if (dialog->m_irDataValid)
			{
				ASF_LivenessInfo irLiveNessInfo = { 0 };
				MRESULT irRes = dialog->m_imageFaceEngine.FaceASFProcess_IR(multiFaceInfo, tempIRImage, irLiveNessInfo);
				if (irRes == 0 && irLiveNessInfo.num > 0)
				{
					if (irLiveNessInfo.isLive[0] == 1)
					{
						dialog->m_curIRVideoShowString = "IR活体";
						isIRAlive = true;
					}
					else if (irLiveNessInfo.isLive[0] == 0)
					{
						dialog->m_curIRVideoShowString = "IR假体";
					}
					else
					{
						//-1：不确定；-2:传入人脸数>1； -3: 人脸过小；-4: 角度过大；-5: 人脸超出边界 
						dialog->m_curIRVideoShowString = "unknown";
					}
				}
				else
				{
					dialog->m_curIRVideoShowString = "";
				}
			}
			else
			{
				dialog->m_curIRVideoShowString = "";
			}

			cvReleaseImage(&tempIRImage);
		}
		else if (g_cameraNum == 1)
		{
			isIRAlive = true;
		}
		else
		{
			break;
		}

		//RGB属性检测
		MRESULT detectRes = dialog->m_imageFaceEngine.FaceASFProcess(multiFaceInfo, tempImage,
			ageInfo, genderInfo, angleInfo, liveNessInfo);

		bool isRGBAlive = false;
		if (detectRes == 0 && liveNessInfo.num > 0)
		{
			if (liveNessInfo.isLive[0] == 1)
			{
				isRGBAlive = true;
			}
			else if (liveNessInfo.isLive[0] == 0)
			{
				dialog->m_curVideoShowString = "RGB假体";
			}
			else
			{
				//-1：不确定；-2:传入人脸数>1； -3: 人脸过小；-4: 角度过大；-5: 人脸超出边界 
				dialog->m_curVideoShowString = "unknown";  
			}
		}
		else
		{
			dialog->m_curVideoShowString = "";
		}

		if (!(isRGBAlive && isIRAlive))
		{
			if (isRGBAlive && !isIRAlive)
			{
				dialog->m_curVideoShowString = "RGB活体";
			}
			cvReleaseImage(&tempImage);
			continue;
		}

		//特征提取
		detectRes = dialog->m_videoFaceEngine.PreExtractFeature(tempImage,
			faceFeature, dialog->m_curFaceInfo);

		cvReleaseImage(&tempImage);

		if (MOK != detectRes)
		{
			continue;
		}

		int maxIndex = 0;
		MFloat maxThreshold = 0.0;
		int curIndex = 0;

		if (dialog->m_bLoadIconThreadRunning ||
			dialog->m_bClearFeatureThreadRunning)
		{
			continue;
		}

		for each (auto regisFeature in dialog->m_featuresVec)
		{
			curIndex++;
			MFloat confidenceLevel = 0;
			MRESULT pairRes = dialog->m_videoFaceEngine.FacePairMatching(confidenceLevel, faceFeature, regisFeature);

			if (MOK == pairRes && confidenceLevel > maxThreshold)
			{
				maxThreshold = confidenceLevel;
				maxIndex = curIndex;
			}
		}

		if (atof(dialog->m_strEditThreshold) >= 0 &&
			maxThreshold >= atof(dialog->m_strEditThreshold) &&
			isRGBAlive && isIRAlive)
		{
			CString resStr;
			resStr.Format("%d号 :%.2f", maxIndex, maxThreshold);
			dialog->m_curVideoShowString = resStr + ",RGB活体";
		}
		else if (isRGBAlive)
		{
			dialog->m_curVideoShowString = "RGB活体";
		}
	}
	
	SafeFree(multiFaceInfo.faceOrient);
	SafeFree(multiFaceInfo.faceRect);
	SafeFree(faceFeature.feature);
	return 0;
}


//双缓存画图
void CArcFaceDemoDlg::IplDrawToHDC(BOOL isVideoMode, IplImage* rgbImage, CRect& strShowRect, UINT ID)
{
	if (!rgbImage)
	{
		return;
	}

	CDC MemDC;

	CClientDC pDc(GetDlgItem(ID));

	//创建与窗口DC兼容的内存DC（MyDC）
	MemDC.CreateCompatibleDC(&pDc);
	
	IplImage* cutImg;
	if (m_curIrVideoImage)
	{
		//红外图像的缩放并拷贝
		IplImage* shrinkIrImage = cvCreateImage(cvSize(m_curIrVideoImage->width / 3, m_curIrVideoImage->height / 3), m_curIrVideoImage->depth, m_curIrVideoImage->nChannels);
		cvResize(m_curIrVideoImage, shrinkIrImage, CV_INTER_AREA);

		//将IR图像融合到RGB图像上
		cv::Mat matRGBImage = cv::cvarrToMat(rgbImage);
		cv::Mat matIRImage = cv::cvarrToMat(shrinkIrImage);
		cv::Mat imageROI = matRGBImage(cv::Rect(10, 10, matIRImage.cols, matIRImage.rows));
		matIRImage.copyTo(imageROI);
		IplImage* roiImage = &IplImage(matRGBImage);	//浅拷贝

		//裁剪图片
		cutImg = cvCreateImage(cvSize(roiImage->width - (roiImage->width % 4), roiImage->height), IPL_DEPTH_8U, roiImage->nChannels);
		PicCutOut(roiImage, cutImg, 0, 0);
		cvReleaseImage(&shrinkIrImage);
	}
	else
	{
		cutImg = cvCreateImage(cvSize(rgbImage->width - (rgbImage->width % 4), rgbImage->height), IPL_DEPTH_8U, rgbImage->nChannels);
		PicCutOut(rgbImage, cutImg, 0, 0);
	}

	CBitmap* bmp = IplImage2CBitmap(cutImg);

	//把内存位图选进内存DC中用来保存在内存DC中绘制的图形
	CBitmap *oldbmp = MemDC.SelectObject(bmp);

	CPen pen(PS_SOLID, 4, RGB(255, 0, 0));
	pDc.SelectStockObject(NULL_BRUSH);

	pDc.SetBkMode(TRANSPARENT);
	pDc.SetTextColor(RGB(0, 0, 255));

	CRect rect;
	GetDlgItem(ID)->GetClientRect(&rect);

	//把内存DC中的图形粘贴到窗口中；
	pDc.SetStretchBltMode(HALFTONE);

	strShowRect.left = strShowRect.left < 0 ? 0 : strShowRect.left;
	strShowRect.top = strShowRect.top < 0 ? 0 : strShowRect.top;
	strShowRect.right = strShowRect.right > rect.right ? 0 : strShowRect.right;
	strShowRect.bottom = strShowRect.bottom > rect.bottom ? rect.bottom : strShowRect.bottom;

	if (isVideoMode)
	{
		pDc.StretchBlt(0, 0, rect.Width(), rect.Height(), &MemDC, 0, 0, VIDEO_FRAME_DEFAULT_WIDTH, VIDEO_FRAME_DEFAULT_HEIGHT, SRCCOPY);

		//为了让文字不贴边
		strShowRect.left += 4;
		strShowRect.top += 4;

		//让文字不超出视频框
		GetDlgItem(ID)->SetFont(m_Font);

		SIZE size;
		GetTextExtentPoint32A(pDc, m_curVideoShowString, (int)strlen(m_curVideoShowString), &size);

		if (strShowRect.left + size.cx > rect.Width())
		{
			strShowRect.left = rect.Width() - size.cx;
		}
		if (strShowRect.top + size.cy > rect.Height())
		{
			strShowRect.top = rect.Height() - size.cy;
		}

		//画比对信息
		if (m_curVideoShowString == "RGB假体")
		{
			pDc.SetTextColor(RGB(255, 242, 0));
		}
		pDc.DrawText(m_curVideoShowString, &strShowRect, DT_TOP | DT_LEFT | DT_NOCLIP);

		if (m_curIRVideoShowString == "IR假体")
		{
			pDc.SetTextColor(RGB(255, 242, 0));
		}
		pDc.DrawText(m_curIRVideoShowString, CRect(20,20,100,100), DT_TOP | DT_LEFT | DT_NOCLIP);
	}
	else
	{
		//图片由于尺寸不一致 ，需要重绘背景
		HBRUSH hBrush = ::CreateSolidBrush(RGB(255, 255, 255));
		::FillRect(pDc.m_hDC, CRect(0, 0, m_windowViewRect.Width(), m_windowViewRect.Height()), hBrush);

		pDc.StretchBlt(m_curImageShowRect.X + 2, m_curImageShowRect.Y + 2,
			m_curImageShowRect.Width - 2, m_curImageShowRect.Height - 5, &MemDC, 0, 0, cutImg->width, cutImg->height, SRCCOPY);

		Gdiplus::Graphics graphics(pDc.m_hDC);
		Gdiplus::Pen pen(Gdiplus::Color::Red, 2);
		graphics.DrawRectangle(&pen, m_curFaceShowRect);

		//画age gender信息
		pDc.DrawText(m_curStaticShowAgeGenderString, &strShowRect, DT_TOP | DT_LEFT | DT_NOCLIP);

		//将比对信息放在age gender信息下
		strShowRect.top += 20;
		strShowRect.bottom += 20;

		//让文字不超出视频框
		GetDlgItem(ID)->SetFont(m_Font);

		SIZE size;
		GetTextExtentPoint32A(pDc, m_curVideoShowString, (int)strlen(m_curVideoShowString), &size);

		if (strShowRect.left + size.cx > rect.Width())
		{
			strShowRect.left = rect.Width() - size.cx;
		}
		if (strShowRect.top + size.cy > rect.Height())
		{
			strShowRect.top = rect.Height() - size.cy;
		}

		//画比对信息
		pDc.DrawText(m_curStaticShowCmpString, &strShowRect, DT_TOP | DT_LEFT | DT_NOCLIP);
	}

	cvReleaseImage(&cutImg);

	//选进原来的位图，删除内存位图对象和内存DC
	MemDC.SelectObject(oldbmp);
	bmp->DeleteObject();
	MemDC.DeleteDC();

}


//图片格式转换
CBitmap* IplImage2CBitmap(const IplImage *img)
{
	if (!img)
	{
		return NULL;
	}

	CBitmap* bitmap = new CBitmap;//new一个CWnd对象
	BITMAPINFO bmpInfo;  //创建位图        
	bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmpInfo.bmiHeader.biWidth = img->width;
	bmpInfo.bmiHeader.biHeight = img->origin ? abs(img->height) : -abs(img->height);//img->height;//高度
	bmpInfo.bmiHeader.biPlanes = 1;
	bmpInfo.bmiHeader.biBitCount = 24;
	bmpInfo.bmiHeader.biCompression = BI_RGB;

	void *pArray = NULL;
	HBITMAP hbmp = CreateDIBSection(NULL, &bmpInfo, DIB_RGB_COLORS, &pArray, NULL, 0);//创建DIB，可直接写入、与设备无关，相当于创建一个位图窗口
	ASSERT(hbmp != NULL);
	UINT uiTotalBytes = img->width * img->height * 3;
	memcpy(pArray, img->imageData, uiTotalBytes);

	bitmap->Attach(hbmp);// 一个CWnd对象的HWND成员指向这个窗口句柄

	return bitmap;
}

void CArcFaceDemoDlg::OnEnChangeEditThreshold()
{
	//更新阈值
	UpdateData(TRUE);
	if (atof(m_strEditThreshold) < 0)
	{
		AfxMessageBox(_T("阈值必须大于0！"));
		SetDlgItemTextA(IDC_EDIT_THRESHOLD, "1.141");
	}

}


void CArcFaceDemoDlg::OnClose()
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	if (m_videoOpened)
	{
		AfxMessageBox(_T("请先关闭摄像头！"));
		return;
	}

	CDialogEx::OnClose();

	m_bLoadIconThreadRunning = FALSE;
	TerminateLoadThread();
	m_bClearFeatureThreadRunning = FALSE;
	ClearRegisterImages();

	m_videoOpened = false;
	Sleep(500);

	m_bFDThreadRunning = FALSE;
	::CloseHandle(m_hFDThread);
	::CloseHandle(m_hFRThread);

	Sleep(500);

	m_imageFaceEngine.UnInitEngine();
	m_videoFaceEngine.UnInitEngine();
}

void CArcFaceDemoDlg::ClearShowWindow()
{
	//清空背景
	CDC* pCDC = GetDlgItem(IDC_STATIC_VIEW)->GetDC();
	HDC hDC = pCDC->m_hDC;
	HBRUSH hBrush = ::CreateSolidBrush(RGB(255, 255, 255));
	::FillRect(hDC, CRect(0, 0, m_windowViewRect.Width(), m_windowViewRect.Height()), hBrush);
	DeleteObject(hBrush);
}


unsigned long _stdcall ClearFaceFeatureOperation(LPVOID lpParam)
{
	CArcFaceDemoDlg* dialog = (CArcFaceDemoDlg*)(lpParam);

	if (dialog == nullptr)
	{
		return 1;
	}

	int iImageCount = dialog->m_IconImageList.GetImageCount();

	dialog->m_IconImageList.Remove(-1);

	dialog->m_ImageListCtrl.DeleteAllItems();

	iImageCount = dialog->m_IconImageList.SetImageCount(0);

	//清除特征
	for (auto feature : dialog->m_featuresVec)
	{
		free(feature.feature);
	}

	dialog->m_featuresVec.clear();

	dialog->m_bClearFeatureThreadRunning = FALSE;

	return 0;
}


BOOL SetTextFont(CFont* font, int fontHeight, int fontWidth, string fontStyle)
{
	return font->CreateFont(
		fontHeight,					// nHeight
		fontWidth,					// nWidth
		0,							// nEscapement
		0,							// nOrientation
		FW_BOLD,					// nWeight
		FALSE,						// bItalic
		FALSE,						// bUnderline
		0,							// cStrikeOut
		DEFAULT_CHARSET,				// nCharSet
		OUT_DEFAULT_PRECIS,			// nOutPrecision
		CLIP_DEFAULT_PRECIS,			// nClipPrecision
		DEFAULT_QUALITY,				// nQuality
		DEFAULT_PITCH | FF_SWISS,		// nPitchAndFamily
		fontStyle.c_str());			// lpszFacename
}

//列出硬件设备
int listDevices(vector<string>& list)
{
	ICreateDevEnum *pDevEnum = NULL;
	IEnumMoniker *pEnum = NULL;
	int deviceCounter = 0;
	CoInitialize(NULL);

	HRESULT hr = CoCreateInstance(
		CLSID_SystemDeviceEnum,
		NULL,
		CLSCTX_INPROC_SERVER,
		IID_ICreateDevEnum,
		reinterpret_cast<void**>(&pDevEnum)
	);

	if (SUCCEEDED(hr))
	{
		hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnum, 0);
		if (hr == S_OK) {

			IMoniker *pMoniker = NULL;
			while (pEnum->Next(1, &pMoniker, NULL) == S_OK)
			{
				IPropertyBag *pPropBag;
				hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
					(void**)(&pPropBag));

				if (FAILED(hr)) {
					pMoniker->Release();
					continue; // Skip this one, maybe the next one will work.
				}

				VARIANT varName;
				VariantInit(&varName);
				hr = pPropBag->Read(L"Description", &varName, 0);
				if (FAILED(hr))
				{
					hr = pPropBag->Read(L"FriendlyName", &varName, 0);
				}

				if (SUCCEEDED(hr))
				{
					hr = pPropBag->Read(L"FriendlyName", &varName, 0);
					int count = 0;
					char tmp[255] = { 0 };
					while (varName.bstrVal[count] != 0x00 && count < 255)
					{
						tmp[count] = (char)varName.bstrVal[count];
						count++;
					}
					list.push_back(tmp);
				}

				pPropBag->Release();
				pPropBag = NULL;
				pMoniker->Release();
				pMoniker = NULL;

				deviceCounter++;
			}

			pDevEnum->Release();
			pDevEnum = NULL;
			pEnum->Release();
			pEnum = NULL;
		}
	}
	return deviceCounter;
}

void ReadSetting(char* appID, char* sdkKey, char* activeKey, char* tag, 
	char* rgbLiveThreshold, char* irLiveThreshold, char* rgbCameraId, char* irCameraId)
{
	CString iniPath = _T(".\\setting.ini");

	char resultStr[MAX_PATH] = "";

	GetPrivateProfileStringA("tag", _T("tag"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(tag, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("APPID"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(appID, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("SDKKEY"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(sdkKey, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("ACTIVE_KEY"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(activeKey, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("rgbLiveThreshold"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(rgbLiveThreshold, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("irLiveThreshold"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(irLiveThreshold, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("rgbCameraId"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(rgbCameraId, resultStr, MAX_PATH);

	GetPrivateProfileStringA(tag, _T("irCameraId"), NULL, resultStr, MAX_PATH, iniPath);
	memcpy(irCameraId, resultStr, MAX_PATH);
}



vector<dlib::matrix<float, 0, 1>> Get_Database() {
	cv::Mat II;
	vector<dlib::matrix<float, 0, 1>> vec;        //定义一个向量组，用于存放每一个人脸的编码；
	float vec_error[30];                         //定义一个浮点型的数组，用于存放一个人脸编码与人脸库的每一个人脸编码的差值；

	//cout << "Enter the path of picture set：";
	//string dir_path = "C:\\Users\\lzx\\Desktop\\ArcSoft_x86\\demo\\ArcfaceDemo\\ArcFaceDemo";
	string test_path;
	vector<string> fileNames;
	fileNames.push_back("C:\\Users\\lzx\\Desktop\\zjl.jpg");


	//我们要做的第一件事是加载所有模型。首先，因为我们需要在图像中查找人脸我们需要人脸检测器：

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	//我们还将使用人脸标记模型将人脸与标准姿势对齐：（有关介绍，请参见Face_Landmark_Detection_ex.cpp）
	dlib::shape_predictor sp;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//终于我们加载Resnet模型进行人脸识别
	anet_type net;
	dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	dlib::matrix<dlib::rgb_pixel> img, img1, img3;            //定义dlib型图片，彩色


 /*-------------------------------------------------------------------------*/
//此下为建立人脸编码库代码
	for (int k = 0; k < fileNames.size(); k++)  //依次加载完图片库里的文件
	{

		string fileFullName = REGIST_FACE_IMAGE;//"C:\\Users\\lzx\\Desktop\\lzx.jpg";//图片地址+文件名
		load_image(img, fileFullName);//load picture      //加载图片
									  // Display the raw image on the screen

		vector<dlib::rectangle> dets = detector(img);  //用dlib自带的人脸检测器检测人脸，然后将人脸位置大小信息存放到dets中
		img1 = img;
		cv::Mat I = dlib::toMat(img1);                     //dlib->opencv
		vector<dlib::full_object_detection> shapes;
		if (dets.size() < 1);
		else if (dets.size() > 1);
		else
		{
			shapes.push_back(sp(img, dets[0]));             //画人脸轮廓，68点

			if (!shapes.empty()) {
				for (int j = 0; j < 68; j++) {
					circle(I, cvPoint(shapes[0].part(j).x(), shapes[0].part(j).y()), 3, cv::Scalar(255, 0, 0), -1);

					//	shapes[0].part(i).x();//68¸ö
				}
			}

			dlib::cv_image<dlib::rgb_pixel> dlib_img(I);//dlib<-opencv


												  // Run the face detector on the image of our action heroes, and for each face extract a
												  // copy that has been normalized to 150x150 pixels in size and appropriately rotated
												  // and centered.


												  //复制已规格化为150x150像素并适当旋转的

												  //居中。
			vector<dlib::matrix<dlib::rgb_pixel>> faces;//定义存放截取人脸数据组

			auto shape = sp(img, dets[0]);
			dlib::matrix<dlib::rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);//截取人脸部分，并将大小调为150*150
			faces.push_back(move(face_chip));
			dlib::image_window win1(img); //显示原图

			win1.add_overlay(dets[0]);//在原图上框出人脸
			dlib::image_window win2(dlib_img);  //显示68点图

			dlib::image_window win3(faces[0]);//显示截取的人脸图像
			vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);                             
			vec.emplace_back(face_descriptors[k]);                             //保存这一个人脸的特征向量到vec向量的对应位置
			//cout << "The vector of picture " << fileNames[k] << "is:" << trans(face_descriptors[0]) << endl;//打印该人脸的标签和特征向量
			//vec[k] += 8;
																											/*-----------------------------------------------------------------------------------*/
		}
	}

	return vec;
}

vector<dlib::matrix<float, 0, 1>> Get_Client() {
	cv::Mat II;
	vector<dlib::matrix<float, 0, 1>> match;        //定义一个向量组，用于存放每一个人脸的编码；
	float vec_error[30];                         //定义一个浮点型的数组，用于存放一个人脸编码与人脸库的每一个人脸编码的差值；

	//cout << "Enter the path of picture set：";
	//string dir_path = "C:\\Users\\lzx\\Desktop\\ArcSoft_x86\\demo\\ArcfaceDemo\\ArcFaceDemo";
	string test_path;
	vector<string> fileNames;
	fileNames.push_back("C:\\Users\\lzx\\Desktop\\zjl.jpg");


	//我们要做的第一件事是加载所有模型。首先，因为我们需要在图像中查找人脸我们需要人脸检测器：

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	//我们还将使用人脸标记模型将人脸与标准姿势对齐：（有关介绍，请参见Face_Landmark_Detection_ex.cpp）
	dlib::shape_predictor sp;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//终于我们加载Resnet模型进行人脸识别
	anet_type net;
	dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	dlib::matrix<dlib::rgb_pixel> img, img1, img3;            //定义dlib型图片，彩色


 /*-------------------------------------------------------------------------*/
//此下为建立人脸编码库代码
	for (int k = 0; k < fileNames.size(); k++)  //依次加载完图片库里的文件
	{

		string fileFullName = MATCH_FACE_IMAGE;//"C:\\Users\\lzx\\Desktop\\lzx2.jpg";//图片地址+文件名
		load_image(img, fileFullName);//load picture      //加载图片
									  // Display the raw image on the screen

		vector<dlib::rectangle> dets = detector(img);  //用dlib自带的人脸检测器检测人脸，然后将人脸位置大小信息存放到dets中
		img1 = img;
		cv::Mat I = dlib::toMat(img1);                     //dlib->opencv
		vector<dlib::full_object_detection> shapes;
		if (dets.size() < 1);
		else if (dets.size() > 1);
		else
		{
			shapes.push_back(sp(img, dets[0]));             //画人脸轮廓，68点

			if (!shapes.empty()) {
				for (int j = 0; j < 68; j++) {
					circle(I, cvPoint(shapes[0].part(j).x(), shapes[0].part(j).y()), 3, cv::Scalar(255, 0, 0), -1);

					//	shapes[0].part(i).x();//68¸ö
				}
			}

			dlib::cv_image<dlib::rgb_pixel> dlib_img(I);//dlib<-opencv


												  // Run the face detector on the image of our action heroes, and for each face extract a
												  // copy that has been normalized to 150x150 pixels in size and appropriately rotated
												  // and centered.


												  //复制已规格化为150x150像素并适当旋转的

												  //居中。
			vector<dlib::matrix<dlib::rgb_pixel>> faces;//定义存放截取人脸数据组

			auto shape = sp(img, dets[0]);
			dlib::matrix<dlib::rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);//截取人脸部分，并将大小调为150*150
			faces.push_back(move(face_chip));
			dlib::image_window win1(img); //显示原图

			win1.add_overlay(dets[0]);//在原图上框出人脸
			dlib::image_window win2(dlib_img);  //显示68点图

			dlib::image_window win3(faces[0]);//显示截取的人脸图像
										// Also put some boxes on the faces so we can see that the detector is finding
										// them.
										//同时在表面放置一些盒子，这样我们可以看到探测器正在寻找他们。

										// This call asks the DNN to convert each face image in faces into a 128D vector.
										// In this 128D vector space, images from the same person will be close to each other
										// but vectors from different people will be far apart.  So we can use these vectors to
										// identify if a pair of images are from the same person or from different people.  
										//此调用要求dnn将面中的每个面图像转换为128d矢量。

										//在这个128d向量空间中，同一个人的图像会彼此靠近

										//但是来自不同人群的向量会相差很远。所以我们可以用这些向量

										//标识一对图像是来自同一个人还是来自不同的人。
			vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);//将150*150人脸图像载入Resnet残差网络，返回128D人脸特征存于face_descriptors
																			//sprintf(vec, "%f", (double)length(face_descriptors[0]);
																			//printf("%f\n", length(face_descriptors[0]));
																			//vec[0] = face_descriptors[0];                              
			match.emplace_back(face_descriptors[k]);                             //保存这一个人脸的特征向量到vec向量的对应位置
																											/*-----------------------------------------------------------------------------------*/
		}
	}
	return match;
}

void CArcFaceDemoDlg::OnBnClickedBtnEncryptEnvIni()
{
	//计时器
	CString str;
	long t1 = 0, t2 = 0;
	//while (1) {
		t1 = GetTickCount64();
		// TODO: 在此添加控件通知处理程序代码
		EncryptionParameters parms(scheme_type::CKKS);
		CArcFaceDemoDlg::poly_modulus_degree = 8192;
		parms.set_poly_modulus_degree(CArcFaceDemoDlg::poly_modulus_degree);
		parms.set_coeff_modulus(CoeffModulus::Create(CArcFaceDemoDlg::poly_modulus_degree, { 45,40,40,45 }));

		CArcFaceDemoDlg::context = SEALContext::Create(parms);
		KeyGenerator keygen(context);
		CArcFaceDemoDlg::galois_keys = keygen.galois_keys();
		CArcFaceDemoDlg::secret_key = keygen.secret_key();
		CArcFaceDemoDlg::public_key = keygen.public_key();
		CArcFaceDemoDlg::relin_keys = keygen.relin_keys();
		t2 = GetTickCount64();
		//if (t2 - t1 < 12000) break;
	//}
	str.Format("加密环境初始化共耗时:%dms", t2 - t1);
	EditOut(str, TRUE);

	GetDlgItem(IDC_BTN_ENCRYPT)->EnableWindow(TRUE);
	GetDlgItem(IDC_BTN_REGIST_ENCRYPT)->EnableWindow(TRUE);
	GetDlgItem(IDC_BTN_DECRYPT)->EnableWindow(TRUE);
}


Ciphertext get_encrypt_probe(CKKSEncoder& ckks_encoder, Encryptor& encryptor, vector<double> v_input) {

	/*
	* 加密获得probe_p;
	*/
	Plaintext v_plaintext;
	// v编码
	ckks_encoder.encode(v_input, scale, v_plaintext);
	Ciphertext probe_p;
	// 加密
	encryptor.encrypt(v_plaintext, probe_p);
	return probe_p;
}


/*
 get (Ci - Pi)^2
*/

Ciphertext get_sub_square(CKKSEncoder& ckks_encoder, Evaluator& evaluator, Decryptor& decryptor, 
	Ciphertext encrypt_E_matrix, Ciphertext probe_p, RelinKeys& relin_keys) {

		/*
		*  Calculate begin
		*/
		Plaintext plain_sub_cache, plain_mult_cache;
		Ciphertext encrypt_sub_cache, encrypt_multiply_cache;
		//vector<double> result_sub_cache, result_mult_cache;

		evaluator.sub(probe_p, encrypt_E_matrix, encrypt_sub_cache);

		evaluator.square(encrypt_sub_cache, encrypt_multiply_cache);
		evaluator.relinearize_inplace(encrypt_multiply_cache, relin_keys);
		evaluator.rescale_to_next_inplace(encrypt_multiply_cache);

		return encrypt_multiply_cache;
}

long summm = 0;

Ciphertext get_sum_rotate(std::shared_ptr<seal::SEALContext>& context, CKKSEncoder & ckks_encoder, 
	Evaluator & evaluator, Encryptor & encryptor, Decryptor & decryptor, Ciphertext encrypt_multiply_cache, 
	GaloisKeys & galois_keys, RelinKeys & relin_keys) {
		//vector<Ciphertext> encrypt_RR_matrix;

	/*
	* get encryptor of vector K{1,0,0,....0}
	* begin
	*/
	size_t slot_count = ckks_encoder.slot_count();
	vector<double> vector_k(slot_count, 0ULL);
	vector_k[0] = 1ULL;
	//print_vector(vector_k, 3, 13);

	Plaintext plain_vector_k;
	Ciphertext encrypt_vector_k;
	ckks_encoder.encode(vector_k, scale, plain_vector_k);
	encryptor.encrypt(plain_vector_k, encrypt_vector_k);
	evaluator.mod_switch_to_next_inplace(encrypt_vector_k);

	/*
	*  Calculate begin
	*/
	//for (auto it = encrypt_R_matrix.begin(); it != encrypt_R_matrix.end(); it++) {

		Plaintext plain_rotated_cache, plain_sum_cache;
		Ciphertext encrypt_rotated_cache, encrypt_sum_cache;
		vector<double> result_rotated_cache, result_sum_cache;

		encrypt_sum_cache = encrypt_multiply_cache;
		/*
		* rotated & add to get sum of them
		*/
		long t1 = 0, t2 = 0;
		for (auto i = 0; i < (int)log2(number_n); i++) {

			t1 = GetTickCount64();
			evaluator.rotate_vector(encrypt_sum_cache, pow(2.0, 1.0 * i), galois_keys, encrypt_rotated_cache);
			t2 = GetTickCount64();
			summm += (t2 - t1);
			encrypt_multiply_cache = encrypt_rotated_cache;

			evaluator.add_inplace(encrypt_sum_cache, encrypt_multiply_cache);
			//evaluator.relinearize_inplace(encrypt_sum_cache, relin_keys);
		}

		evaluator.multiply_inplace(encrypt_sum_cache, encrypt_vector_k);
		evaluator.relinearize_inplace(encrypt_sum_cache, relin_keys);
		evaluator.rescale_to_next_inplace(encrypt_sum_cache);
	/*
	*  Calculate end;
	*/
		return encrypt_sum_cache;//encrypt_RR_matrix;
}

    /*
    * Core
    * get dist(Ci,P)
    * dist(c,p)
    */
Ciphertext get_dist(std::shared_ptr<seal::SEALContext>& context, CKKSEncoder& ckks_encoder, 
	Evaluator& evaluator, Ciphertext encrypt_E_matrix, Ciphertext probe_p, Encryptor& encryptor, 
	Decryptor& decryptor, RelinKeys& relin_keys, GaloisKeys& galois_keys) {

	Ciphertext encrypt_R_matrix;

	/*
	* get (Ci - Pi)^2
	*/

	Ciphertext encrypt_R_matrix_cache = get_sub_square(ckks_encoder, evaluator, decryptor, encrypt_E_matrix, probe_p, relin_keys);

	/*
	* get sum
	*/
	encrypt_R_matrix = get_sum_rotate(context, ckks_encoder, evaluator, encryptor, decryptor, encrypt_R_matrix_cache, galois_keys, relin_keys);

	return encrypt_R_matrix;
}

Ciphertext get_up_cosine_dist_square(std::shared_ptr<seal::SEALContext>& context, CKKSEncoder& ckks_encoder,
	Evaluator& evaluator, Ciphertext encrypt_E_matrix, Ciphertext probe_p, Encryptor& encryptor,
	Decryptor& decryptor, RelinKeys& relin_keys, GaloisKeys& galois_keys) {

	Ciphertext encrypt_up_distance;

	/*
	* get (Ci · Pi)
	*/
	Ciphertext encrypt_multiply_cache;
	//vector<double> result_sub_cache, result_mult_cache;

	//evaluator.sub(probe_p, encrypt_E_matrix, encrypt_sub_cache);

	evaluator.multiply(encrypt_E_matrix, probe_p, encrypt_multiply_cache);
	evaluator.relinearize_inplace(encrypt_multiply_cache, relin_keys);
	evaluator.rescale_to_next_inplace(encrypt_multiply_cache);

	/*
	* get sum
	*/
	encrypt_up_distance = 
		get_sum_rotate(context, ckks_encoder, evaluator, encryptor, decryptor, encrypt_multiply_cache, galois_keys, relin_keys);

	return encrypt_up_distance;
}

Ciphertext get_down_cosine_dist_square(std::shared_ptr<seal::SEALContext>& context, CKKSEncoder& ckks_encoder,
	Evaluator& evaluator, Ciphertext encrypt_E_matrix, Ciphertext probe_p, Encryptor& encryptor,
	Decryptor& decryptor, RelinKeys& relin_keys, GaloisKeys& galois_keys) {

	Ciphertext encrypt_down_distance;

	/*
	* get (Ci · Pi)
	*/
	Ciphertext encrypt_a_multiply_cache, encrypt_b_multiply_cache,
		encrypt_a_down_distance, encrypt_b_down_distance;
	//vector<double> result_sub_cache, result_mult_cache;

	//a
	evaluator.square(probe_p, encrypt_a_multiply_cache);
	evaluator.relinearize_inplace(encrypt_a_multiply_cache, relin_keys);
	evaluator.rescale_to_next_inplace(encrypt_a_multiply_cache);

	//b
	evaluator.square(encrypt_E_matrix, encrypt_b_multiply_cache);
	evaluator.relinearize_inplace(encrypt_b_multiply_cache, relin_keys);
	evaluator.rescale_to_next_inplace(encrypt_b_multiply_cache);

	/*
	* get sum
	*/
	encrypt_a_down_distance =
		get_sum_rotate(context, ckks_encoder, evaluator, encryptor, decryptor, 
			encrypt_a_multiply_cache, galois_keys, relin_keys);

	encrypt_b_down_distance =
		get_sum_rotate(context, ckks_encoder, evaluator, encryptor, decryptor, 
			encrypt_b_multiply_cache, galois_keys, relin_keys);

	evaluator.multiply(encrypt_a_multiply_cache, encrypt_b_multiply_cache, encrypt_down_distance);
	evaluator.relinearize_inplace(encrypt_down_distance, relin_keys);
	evaluator.rescale_to_next_inplace(encrypt_down_distance);

	return encrypt_down_distance;
}

/*
* shifting the ri vectors
* Through shifting the rivectors, a structure conceptually akin to a diagonal matrix is reached,
*/
vector<Ciphertext> get_shifting_ri(std::shared_ptr<seal::SEALContext>& context, CKKSEncoder& ckks_encoder, 
	Evaluator& evaluator, vector<Ciphertext> encrypt_R_matrix, GaloisKeys& galois_keys, Decryptor& decryptor) {

	vector<Ciphertext> encrypt_RR_matrix;
	Plaintext plain_shift_cache;
	vector<double>result_shift_cache;

	int step = 0;
	for (auto it = encrypt_R_matrix.begin(); it != encrypt_R_matrix.end(); it++) {
		Ciphertext after_shift;
		// 右移

		evaluator.rotate_vector((*it), step, galois_keys, after_shift);


		// reslut output test
		decryptor.decrypt(after_shift, plain_shift_cache);
		result_shift_cache.clear();
		ckks_encoder.decode(plain_shift_cache, result_shift_cache);

		//cout << "rotated step: "<<step << endl;

		std::cout << std::fixed << std::setprecision(13); // 设置输出保留位数
		for (auto i = 0; i < 10; i++)
		{
			cout << result_shift_cache[i] << " ";
		}
		cout << endl;
		//print_vector(result_shift_cache, 5, 13);

		step--;
		encrypt_RR_matrix.push_back(after_shift);

	}

	return encrypt_RR_matrix;
}

/*
*  Those vectors are combined by adding them together,
*  thereby producing a single encrypted vector holding the comparison scores of p against E,
*  i.e. a mapping was introduced so that R →(r1,1, r2,1, . . . rN,1).
*/
Ciphertext get_combined_R(std::shared_ptr<seal::SEALContext>& context, CKKSEncoder& ckks_encoder, Evaluator& evaluator, 
	vector<Ciphertext> encrypt_R_matrix, Encryptor& encryptor, Decryptor& decryptor, RelinKeys& relin_keys) {
	Ciphertext encrypt_R_sum_cache;

	cout << " combined together: " << endl;
	int step = 0;
	for (auto it = encrypt_R_matrix.begin(); it != encrypt_R_matrix.end(); it++) {

		if (step == 0)
		{
			encrypt_R_sum_cache = (*it);
			step++;
			continue;
		}
		else {
			parms_id_type last_parms_id = (*it).parms_id();
			evaluator.add_inplace(encrypt_R_sum_cache, (*it));
			evaluator.relinearize_inplace(encrypt_R_sum_cache, relin_keys);
			step++;
		}
	}
	return encrypt_R_sum_cache;
}

vector<double> p, q;
double x_sum = 0, y_sum = 0;
void CArcFaceDemoDlg::OnBnClickedBtnEncryptor()
{
	//计时器
	CString str;
	//加密器
	Encryptor encryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::public_key);
	//计算器
	Evaluator evaluator(CArcFaceDemoDlg::context);
	//解密器
	Decryptor decryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::secret_key);

	CKKSEncoder ckks_encoder(CArcFaceDemoDlg::context);
	if (!m_curStaticImageFRSucceed)
	{
		AfxMessageBox(_T("人脸比对失败，请重新选择识别照!"));
		return;
	}

	vector<dlib::matrix<float, 0, 1>> test_descriptors = Get_Client();

	vector<double> v_input;
	number_n = test_descriptors[0].size();

	for (auto it = test_descriptors[0].begin(); it != test_descriptors[0].end(); it++) {
		v_input.emplace_back(*it);p.emplace_back(*it);
	}

	//double y_sum = 0;
	for (int i = 0; i < v_input.size(); i++) y_sum += pow(v_input[i], 2);

	vector<double> y_alter; //y_alter.emplace_back(0.5);
	for (int i = 0; i < v_input.size(); i++) y_alter.emplace_back(-2 * v_input[i]);
	//y_alter.emplace_back(y_sum / 2);

	long t1 = GetTickCount64();
	CArcFaceDemoDlg::encrypt_y_alter = get_encrypt_probe(ckks_encoder, encryptor, y_alter);
	long t2 = GetTickCount64();

	long ty1 = GetTickCount64();
	CArcFaceDemoDlg::encrypt_probe_p = get_encrypt_probe(ckks_encoder, encryptor, v_input);
	long ty2 = GetTickCount64();

	/* BASELINE
	long t1 = GetTickCount64();
	CArcFaceDemoDlg::encrypt_probe_p = get_encrypt_probe(ckks_encoder, encryptor, v_input);
	long t2 = GetTickCount64();
	*/

	ofstream OutFile_p; OutFile_p.open("C:\\Users\\lzx\\Desktop\\Cloud\\666.txt");
	CArcFaceDemoDlg::encrypt_probe_p.save(OutFile_p);
	OutFile_p.close();

    OutFile_p.open("C:\\Users\\lzx\\Desktop\\Cloud\\777.txt");
	string spara;
	//parms_id_type parr = CArcFaceDemoDlg::encrypt_E_vector.parms_id();
	for (auto it = p.begin();it != p.end(); it++) {
		OutFile_p << *it;
	}
	//OutFile_p << spara;
	OutFile_p.close();
 
	//str.Format("加密识别照共耗时:%dms", t2 - t1);
	//EditOut(str, TRUE);
	str.Format("加密识别照共耗时:%dms", ty2 - ty1);
	EditOut(str, TRUE);
	AfxMessageBox(_T("所选识别人脸加密完成！"));
	 
}

vector<Ciphertext> UN_SIMD_VECTOR;
void CArcFaceDemoDlg::OnBnClickedBtnRegistEncrypt()
{
	//计时器
	CString str, str1;

	//加密器
	Encryptor encryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::public_key);
	//计算器
	Evaluator evaluator(CArcFaceDemoDlg::context);
	//解密器
	Decryptor decryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::secret_key);

	CKKSEncoder ckks_encoder(CArcFaceDemoDlg::context);

	//纯粹时间种子
	default_random_engine e;
	// TODO: 在此添加控件通知处理程序代码
	vector<dlib::matrix<float, 0, 1>> vec = Get_Database();
	if (vec.size() > 0) m_curRegistImageFRSucceed = true;
	if (!m_curRegistImageFRSucceed)
	{
		AfxMessageBox(_T("注册图片未检测到人脸，请重新选择图片！"));
		return;
	}
	vector<double> E_vector;
	for (auto it = vec[0].begin(); it != vec[0].end(); it++) {
		E_vector.emplace_back(*it);q.emplace_back(*it);//“库向量”
	}
	//向量变形
	//x_sum = 0;

	for (int i = 0; i < E_vector.size(); i++) x_sum += pow(E_vector[i], 2);
	CString temp;
	vector<double> x_alter; //x_alter.emplace_back(x_sum / 2);
	for (int i = 0; i < E_vector.size(); i++) x_alter.emplace_back(E_vector[i]);
	//x_alter.emplace_back(0.5);

	long tx1 = GetTickCount64();
	CArcFaceDemoDlg::encrypt_E_vector = get_encrypt_probe(ckks_encoder, encryptor, E_vector);
	long tx2 = GetTickCount64();

	long t1 = GetTickCount64();
	CArcFaceDemoDlg::encrypt_x_alter = get_encrypt_probe(ckks_encoder, encryptor, x_alter);
	long t2 = GetTickCount64();

	long uid = e() % 1000000; 
	temp.Format("成功注册，您的用户认证序列号为:%d，请妥善保存！", uid);
	EditOut(temp, TRUE);
	string s_uid = to_string(uid);
	string folderPath = "C:\\Users\\lzx\\Desktop\\Cloud\\" + s_uid;
	string command;
	command = "mkdir -p " + folderPath;
	system(command.c_str());
	ofstream OutFile_c; OutFile_c.open(folderPath + "\\Ciphertext.txt");
	string cipher,cipher_compression;
	for (auto it = CArcFaceDemoDlg::encrypt_E_vector.data(); it != CArcFaceDemoDlg::encrypt_E_vector.data_end(); it++) {
		cipher_compression += to_string(*it); cipher += to_string(*it); if (it != CArcFaceDemoDlg::encrypt_E_vector.data_end() - 1) cipher += "*";
	}
 
	OutFile_c << cipher;
	OutFile_c.close();

	OutFile_c.open("C:\\Users\\lzx\\Desktop\\888.txt");
	OutFile_c << cipher_compression;
	OutFile_c.close();
	//CArcFaceDemoDlg::encrypt_E_vector.data_release();
	/*
	ofstream OutFile_p; OutFile_p.open(folderPath + "\\Parms_Info.txt");
	string para;
	//parms_id_type parr = CArcFaceDemoDlg::encrypt_E_vector.parms_id();
	for (auto it = CArcFaceDemoDlg::encrypt_E_vector.parms_id().begin();
		it != CArcFaceDemoDlg::encrypt_E_vector.parms_id().end(); it++) {
		para += to_string(*it); if (it != CArcFaceDemoDlg::encrypt_E_vector.parms_id().end() - 1) para += "*";
	}
	OutFile_p << para;
	OutFile_p.close();

	ofstream OutFile_m; OutFile_m.open(folderPath + "\\Main_Info.txt");
	string main_info =
		to_string(CArcFaceDemoDlg::encrypt_E_vector.size()) + "*" +
		to_string(CArcFaceDemoDlg::encrypt_E_vector.coeff_mod_count()) + "*" +
		to_string(CArcFaceDemoDlg::encrypt_E_vector.is_ntt_form()) + "*" +
		to_string(CArcFaceDemoDlg::encrypt_E_vector.poly_modulus_degree()) + "*" +
		to_string(CArcFaceDemoDlg::encrypt_E_vector.scale());

	OutFile_m << main_info;
	OutFile_m.close();
	//CArcFaceDemoDlg::encrypt_E_vector.release();
	//加密传输控制插口
	/*
	MYSQL* con = new MYSQL;
	mysql_init(con);
	if (!mysql_real_connect(con, HOST, USERNAME, PASSWORD, DATABASE, PORT, NULL, 0)) {
		AfxMessageBox(_T("连接人脸数据库失败！"));
		exit(1);
	}
	MYSQL_STMT* stmt = mysql_stmt_init(con);

	char MainInsertSql[2000] = 
		"insert into control_message(uid,size,poly_modulus_degree,coeff_mod_count,is_ntt_form,scale)values(?,?,?,?,?,?)";
	if (mysql_stmt_prepare(stmt, MainInsertSql, strlen(MainInsertSql)) != 0) {
		AfxMessageBox(_T("普通参数库初始化失败！"));
		exit(1);
	}
	long uid = rand() / 10000000;
	long size = CArcFaceDemoDlg::encrypt_E_vector.size();
	long coeff = CArcFaceDemoDlg::encrypt_E_vector.coeff_mod_count();
	long poly = CArcFaceDemoDlg::encrypt_E_vector.poly_modulus_degree();
	bool isntt = CArcFaceDemoDlg::encrypt_E_vector.is_ntt_form();
	double _scale_ = CArcFaceDemoDlg::encrypt_E_vector.scale();

	CString temp;
    temp.Format("成功注册，您的用户认证序列号为:%d，请妥善保存！", uid);
	EditOut(temp, TRUE);

	MYSQL_BIND bindData[6];
	memset(bindData, 0, sizeof(bindData));

	bindData[0].buffer_type = MYSQL_TYPE_LONG;
	bindData[0].buffer = &uid;
	bindData[0].is_null = 0;

	bindData[1].buffer_type = MYSQL_TYPE_LONG;
	bindData[1].buffer = &size;
	bindData[1].is_null = 0;

	bindData[2].buffer_type = MYSQL_TYPE_LONG;
	bindData[2].buffer = &poly;
	bindData[2].is_null = 0;

	bindData[3].buffer_type = MYSQL_TYPE_LONG;
	bindData[3].buffer = &coeff;
	bindData[3].is_null = 0;

	bindData[4].buffer_type = MYSQL_TYPE_BIT;
	bindData[4].buffer = &isntt;
	bindData[4].is_null = 0;

	bindData[5].buffer_type = MYSQL_TYPE_DOUBLE;
	bindData[5].buffer = &_scale_;
	bindData[5].is_null = 0;

	if (!mysql_stmt_bind_param(stmt, bindData)) {
		AfxMessageBox(_T("普通密文控制信息绑定失败！"));
		exit(1);
	}

	string cipher;
	for (auto it = CArcFaceDemoDlg::encrypt_E_vector.data(); it != CArcFaceDemoDlg::encrypt_E_vector.data_end(); it++) {
		cipher += to_string(*it); cipher += "*";
	}

	char* ct = const_cast<char*>(cipher.data());
	long ct_len = strlen(ct);
	char CipherInsertSql[60000001] =
		"insert into ciphertext_message(uid,ciphertext)values(?,?)";
	if (mysql_stmt_prepare(stmt, CipherInsertSql, strlen(CipherInsertSql)) != 0) {
		AfxMessageBox(_T("密文库初始化失败！"));
		exit(1);
	}

	MYSQL_BIND CipherData[2];
	memset(CipherData, 0, sizeof(CipherData));

	CipherData[0].buffer_type = MYSQL_TYPE_LONG;
	CipherData[0].buffer = &uid;
	CipherData[0].is_null = 0;

	CipherData[1].buffer_type = MYSQL_TYPE_MEDIUM_BLOB;
	CipherData[1].buffer = (char*)ct;
	CipherData[1].buffer_length = ct_len;
	CipherData[1].is_null = 0;
	CipherData[1].length = (unsigned long*)&ct_len;

	if (!mysql_stmt_bind_param(stmt, CipherData)) {
		AfxMessageBox(_T("密文流绑定失败！"));
		exit(1);
	}

	string para;
	//parms_id_type parr = CArcFaceDemoDlg::encrypt_E_vector.parms_id();
	for (auto it = CArcFaceDemoDlg::encrypt_E_vector.parms_id().begin(); 
		it != CArcFaceDemoDlg::encrypt_E_vector.parms_id().end(); it++) {
		para += to_string(*it); para += "*";
	}

	char* para_array = const_cast<char*>(para.data());
	long para_len = strlen(para_array);
	char ParmsInsertSql[2000001] =
		"insert into parms_message(uid,parameters)values(?,?)";
	if (mysql_stmt_prepare(stmt, ParmsInsertSql, strlen(ParmsInsertSql)) != 0) {
		AfxMessageBox(_T("参数库初始化失败！"));
		exit(1);
	}

	MYSQL_BIND ParmsData[2];
	memset(ParmsData, 0, sizeof(ParmsData));

	ParmsData[0].buffer_type = MYSQL_TYPE_LONG;
	ParmsData[0].buffer = &uid;
	ParmsData[0].is_null = 0;

	ParmsData[1].buffer_type = MYSQL_TYPE_MEDIUM_BLOB;
	ParmsData[1].buffer = (char*)para_array;
	ParmsData[1].buffer_length = para_len;
	ParmsData[1].is_null = 0;
	ParmsData[1].length = (unsigned long*)&para_len;
 
	if (!mysql_stmt_bind_param(stmt, ParmsData)) {
		AfxMessageBox(_T("参数流绑定失败！"));
		exit(1);
	}

	mysql_stmt_execute(stmt);
	CArcFaceDemoDlg::encrypt_E_vector.release();
	mysql_stmt_close(stmt);
	mysql_close(con);
	*/

    //str.Format("加密注册照共耗时:%dms", tx2 - tx1);
    //EditOut(str, TRUE);

	str.Format("加密注册照共耗时:%dms", t2 - t1);
	EditOut(str, TRUE);

	AfxMessageBox(_T("所选注册人脸加密完成！"));



	/*
	delete[]ParmsData;
	delete[]CipherData;
	delete[]bindData;
	delete[]ct;
	delete[]para_array;
	delete[]MainInsertSql;
	delete[]CipherInsertSql;
	delete[]ParmsInsertSql;
	*/
	string record;
	for (int i = 1; i <= 22; i++) {
		OutFile_c.open("C:\\Users\\lzx\\Desktop\\999.txt");
		string dst; //const int compressionlevel = 22;
		const char* cipher_char = cipher_compression.data();
		size_t const cBuffSize = ZSTD_compressBound(cipher_compression.size());
		dst.resize(cBuffSize);
		char* char_compress = new char[cBuffSize];
		long yy1 = GetTickCount64();
		size_t const cSize = ZSTD_compress(char_compress, cBuffSize, cipher_char, cipher_compression.size(), i);
		long yy2 = GetTickCount64();
		dst.resize(cSize);
		//OutFile_c.open("C:\\Users\\lzx\\Desktop\\999.txt");
		dst = char_compress;
		OutFile_c << dst;
		OutFile_c.close();
		//OutFile_c.open("C:\\Users\\lzx\\Desktop\\" + to_string(i) + ".txt");
		//OutFile_c << dst;
		//OutFile_c.close();
		unsigned long long decom_buf_size = ZSTD_getFrameContentSize(char_compress, cSize);
		char* decom_ptr = new char[decom_buf_size];

		ifstream infile; infile.open("C:\\Users\\lzx\\Desktop\\999.txt", ios::in); infile >> char_compress;
		long yyy2 = GetTickCount64();
		size_t decom_size = ZSTD_decompress(decom_ptr, decom_buf_size, char_compress, cSize);
		long yy3 = GetTickCount64();

		/*
		OutFile_c.open("C:\\Users\\lzx\\Desktop\\1999.txt");
		dst = decom_ptr;
		OutFile_c << dst;
		OutFile_c.close();
		*/
		DeleteFile("C:\\Users\\lzx\\Desktop\\999.txt");
		//OutFile_c.open("C:\\Users\\lzx\\Desktop\\ZSTD.txt");
		record += to_string(i) + " compress:" + to_string(yy2 - yy1) + " decompress:" + to_string(yy3 - yyy2) + " ";
		//OutFile_c << endl;
		/*
		str.Format("压缩加密模板共耗时:%dms", yy2 - yy1);
		EditOut(str, TRUE);

		str.Format("解压缩加密模板共耗时:%dms", yy3 - yy2);
		EditOut(str, TRUE);
		*/
		//OutFile_c.close();
	}
	OutFile_c.open("C:\\Users\\lzx\\Desktop\\ZSTD.txt");
	OutFile_c << record;
	OutFile_c.close();
}

long T2 = 0, T1 = 0;
void CArcFaceDemoDlg::OnBnClickedBtnCompare()
{
	// TODO:  在此添加控件通知处理程序代码
	//计时器
	CString str, str1;

	//加密器
	Encryptor encryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::public_key);
	//计算器
	Evaluator evaluator(CArcFaceDemoDlg::context);
	//解密器
	Decryptor decryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::secret_key);

	CKKSEncoder ckks_encoder(CArcFaceDemoDlg::context);

	ofstream OutFile_p; OutFile_p.open("C:\\Users\\lzx\\Desktop\\Cloud\\test.txt");
	string para;
	//parms_id_type parr = CArcFaceDemoDlg::encrypt_E_vector.parms_id();
	for (auto it = CArcFaceDemoDlg::encrypt_probe_p.parms_id().begin();
		it != CArcFaceDemoDlg::encrypt_probe_p.parms_id().end(); it++) {
		para += to_string(*it); if (it != CArcFaceDemoDlg::encrypt_probe_p.parms_id().end() - 1) para += "*";
	}
	OutFile_p << para;
	OutFile_p.close();

	double sum = 0.0;
	for (int i = 0; i < p.size(); i++) {
		sum += pow(p[i] - q[i], 2);
	}
	double ssum = sqrt(sum);

	str1.Format("加密前人脸匹配率:%2f%%", 100 / (1 + exp(14.0879341576 * ssum - 7.2786390745)));
	EditOut(str1, TRUE);

	Ciphertext encrypt_distance;
	Ciphertext encrypt_multiply_cache;

	long t1 = GetTickCount64();
	evaluator.multiply(CArcFaceDemoDlg::encrypt_x_alter, CArcFaceDemoDlg::encrypt_y_alter, encrypt_multiply_cache);
	evaluator.relinearize_inplace(encrypt_multiply_cache, relin_keys);
	evaluator.rescale_to_next_inplace(encrypt_multiply_cache);
	long tt1 = GetTickCount64();
	encrypt_distance =
		get_sum_rotate(context, ckks_encoder, evaluator,
			encryptor, decryptor, encrypt_multiply_cache, galois_keys, relin_keys);
	long tt2 = GetTickCount64();
	Plaintext plain_add_cache;
	vector<double> result_add_cache;

	decryptor.decrypt(encrypt_distance, plain_add_cache);
	ckks_encoder.decode(plain_add_cache, result_add_cache);
	long t2 = GetTickCount64();
	//str.Format("变形乘法操作共耗时:%dms", tt1 - t1);
	//EditOut(str, TRUE);
	//str.Format("变形循环移位共耗时:%dms", summm);
	//EditOut(str, TRUE);
	T2 = t2 - t1;
	//str.Format("采用本方案共耗时:%dms", t2 - t1);
	//EditOut(str, TRUE);
	//str.Format("变形解密解码共耗时:%dms", t2 - tt2);
	//EditOut(str, TRUE);

	//CString resStr1, resStr2, resStr3;
	//resStr1.Format("变形人脸匹配率: %2f%%", 100 / (1 + exp(14.0879341576 * sqrt(x_sum + y_sum + result_add_cache[0]) - 7.2786390745)));
	//EditOut(resStr1, TRUE);

}

void CArcFaceDemoDlg::OnBnClickedBtnDecryptor()
{   
	//计时器
	CString str;
	//加密器
	Encryptor encryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::public_key);
	//计算器
	Evaluator evaluator(CArcFaceDemoDlg::context);
	//解密器
	Decryptor decryptor(CArcFaceDemoDlg::context, CArcFaceDemoDlg::secret_key);

	CKKSEncoder ckks_encoder(CArcFaceDemoDlg::context);
	// TODO: 在此添加控件通知处理程序代码
	/*
	long u1 = GetTickCount64();
	Ciphertext encrypt_sum_cache;
	AfxMessageBox(_T("1！"));
	for (int i = 0; i < 1; i++) {
		Ciphertext encrypt_R_matrix_cache =
			get_sub_square(ckks_encoder, evaluator, decryptor, UN_SIMD_E_VECTOR[i], UN_SIMD_VECTOR[i], relin_keys);
		evaluator.add_inplace(encrypt_sum_cache, encrypt_R_matrix_cache);
		evaluator.relinearize_inplace(encrypt_sum_cache, relin_keys);
	}
	AfxMessageBox(_T("2！"));
	long u2 = GetTickCount64();
	str.Format("原始方法求解共耗时:%dms", u2 - u1);
	EditOut(str, TRUE);*/
	/*
	//密文恢复插口
	MYSQL* con = new MYSQL;
	mysql_init(con);
	if (!mysql_real_connect(con, HOST, USERNAME, PASSWORD, DATABASE, PORT, NULL, 0)) {
		AfxMessageBox(_T("连接人脸数据库失败！"));
		exit(1);
	}
	MYSQL_STMT* stmt = mysql_stmt_init(con);

	char SelectMainSql[2000] =
		"select uid,size,poly_modulus_degree,coeff_mod_count,is_ntt_form,scale from control_message";
	if (mysql_stmt_prepare(stmt, SelectMainSql, strlen(SelectMainSql))) {
		AfxMessageBox(_T("普通参数库初始化失败！"));
		exit(1);
	}

	CString temp;
	GetDlgItem(IDC_SEQUENCE_EDIT)->GetWindowText(temp);
	long input = atoi(temp);//客户端输入/////////////////

	long uid;long size;
	long coeff;long poly;
	bool isntt;double _scale_;

	MYSQL_BIND bindData[6];
	memset(bindData, 0, sizeof(bindData));

	bindData[0].buffer_type = MYSQL_TYPE_LONG;
	bindData[0].buffer = &uid;

	bindData[1].buffer_type = MYSQL_TYPE_LONG;
	bindData[1].buffer = &size;

	bindData[2].buffer_type = MYSQL_TYPE_LONG;
	bindData[2].buffer = &poly;

	bindData[3].buffer_type = MYSQL_TYPE_LONG;
	bindData[3].buffer = &coeff;

	bindData[4].buffer_type = MYSQL_TYPE_BIT;
	bindData[4].buffer = &isntt;
	//bindData[4].is_null = 0;

	bindData[5].buffer_type = MYSQL_TYPE_DOUBLE;
	bindData[5].buffer = &_scale_;
//	bindData[5].is_null = 0;

	if (!mysql_stmt_bind_param(stmt, bindData)) {
		AfxMessageBox(_T("普通密文控制信息绑定失败！"));
		exit(1);
	}

	if (!mysql_stmt_bind_result(stmt, bindData)) {
		AfxMessageBox(_T("普通密文控制信息绑定结果异常！"));
		exit(1);
	}

	while (mysql_stmt_fetch(stmt) == 0) {
		if (uid == input) {
			CArcFaceDemoDlg::encrypt_E_vector.re_scale(_scale_);
			CArcFaceDemoDlg::encrypt_E_vector.re_size(size);
			CArcFaceDemoDlg::encrypt_E_vector.re_ntt(isntt);
			CArcFaceDemoDlg::encrypt_E_vector.re_coeff(coeff);
			CArcFaceDemoDlg::encrypt_E_vector.re_poly(poly);
		}
	}

	char* ct = new char[60000001]; const char s[2] = "*";
	long ct_len = strlen(ct);
	char SelectCipherSql[60000001] =
		"select uid,ciphertext from ciphertext_message";
	if (mysql_stmt_prepare(stmt, SelectCipherSql, strlen(SelectCipherSql)) != 0) {
		AfxMessageBox(_T("密文库初始化失败！"));
		exit(1);
	}

	MYSQL_BIND CipherData[2];
	memset(CipherData, 0, sizeof(CipherData));

	CipherData[0].buffer_type = MYSQL_TYPE_LONG;
	CipherData[0].buffer = &uid;

	CipherData[1].buffer_type = MYSQL_TYPE_MEDIUM_BLOB;
	CipherData[1].buffer = ct;
	CipherData[1].buffer_length = sizeof(ct);
	
	if (!mysql_stmt_bind_param(stmt, CipherData)) {
		AfxMessageBox(_T("密文流绑定失败！"));
		exit(1);
	}
	if (!mysql_stmt_bind_result(stmt, CipherData)) {
		AfxMessageBox(_T("密文流绑定结果异常！"));
		exit(1);
	}
	IntArray<uint64_t> rep;
	char* c_token; c_token = strtok(ct, s);
	for (int i = 0; i < CArcFaceDemoDlg::encrypt_probe_p.uint64_count()||!c_token; i++) {
		rep[i] = atoi(c_token);
		c_token = strtok(NULL, s);
	}
	while (mysql_stmt_fetch(stmt) == 0) {
		if (uid == input) {
			CArcFaceDemoDlg::encrypt_E_vector.re_ciphertext(rep);
		}
	}

	char* para_array = new char[2000001];
	long para_len = strlen(para_array);
	char SelectParmsSql[2000001] =
		"select uid,parameters from parms_message";
	if (mysql_stmt_prepare(stmt, SelectParmsSql, strlen(SelectParmsSql)) != 0) {
		AfxMessageBox(_T("参数库初始化失败！"));
		exit(1);
	}

	MYSQL_BIND ParmsData[2];
	memset(ParmsData, 0, sizeof(ParmsData));

	ParmsData[0].buffer_type = MYSQL_TYPE_LONG;
	ParmsData[0].buffer = &uid;

	ParmsData[1].buffer_type = MYSQL_TYPE_MEDIUM_BLOB;
	ParmsData[1].buffer = para_array;
	ParmsData[1].buffer_length = para_len;

	if (!mysql_stmt_bind_param(stmt, ParmsData)) {
		AfxMessageBox(_T("参数流绑定失败！"));
		exit(1);
	}
	if (!mysql_stmt_bind_result(stmt, ParmsData)) {
		AfxMessageBox(_T("参数流绑定结果异常！"));
		exit(1);
	}
	parms_id_type pt;
	char* p_token; p_token = strtok(para_array, s);
	for (int i = 0; i < CArcFaceDemoDlg::encrypt_probe_p.parms_id().size() || !p_token; i++) {
		pt[i] = atoi(p_token);
		p_token = strtok(NULL, s);
	}
	while (mysql_stmt_fetch(stmt) == 0) {
		if (uid == input) {
			CArcFaceDemoDlg::encrypt_E_vector.re_parms_id(pt);
		}
	}
	mysql_stmt_close(stmt);
	mysql_close(con);

	delete[]ParmsData;
	delete[]CipherData;
	delete[]bindData;
	delete[]ct;
	delete[]c_token;
	delete[]p_token;
	delete[]para_array;
	delete[]SelectMainSql;
	delete[]SelectCipherSql;
	delete[]SelectParmsSql;
	*/
    CString temp;
    GetDlgItem(IDC_SEQUENCE_EDIT)->GetWindowText(temp);
	if(temp.IsEmpty()){
		AfxMessageBox(_T("请输入您的注册序列码！"));
		return;
	}
	long input;
	string g = temp;
	stringstream geek(g); geek >> input;
	string s_uid = to_string(input);
	string folderPath = "C:\\Users\\lzx\\Desktop\\Cloud\\" + s_uid;

	ifstream infile;

	/*
	CArcFaceDemoDlg::encrypt_E_vector.re_size(CArcFaceDemoDlg::encrypt_probe_p.size());
	CArcFaceDemoDlg::encrypt_E_vector.re_coeff(CArcFaceDemoDlg::encrypt_probe_p.coeff_mod_count());
	CArcFaceDemoDlg::encrypt_E_vector.re_ntt(CArcFaceDemoDlg::encrypt_probe_p.is_ntt_form());
	CArcFaceDemoDlg::encrypt_E_vector.re_poly(CArcFaceDemoDlg::encrypt_probe_p.poly_modulus_degree());
	CArcFaceDemoDlg::encrypt_E_vector.re_scale(CArcFaceDemoDlg::encrypt_probe_p.scale());
	*//*
	infile.open(folderPath + "\\Main_Info.txt", ios::in);
	if (!infile.is_open())
	{
		AfxMessageBox(_T("没有拥有该注册序列码的用户信息或您没有获取该信息的权限！")); 
		return;
	}

	//long size;
	//long coeff; long poly;
	//bool isntt; double _scale_;
	*/
	const char s[2] = "*";
	/*char* m_token = new char[2000]; char* m_temp = new char[2000]; int count = 0;
	while (!infile.eof()) {
		infile >> m_token;
	}
	m_temp = strtok(m_token, s);
	while (m_temp) {
		count++;
		switch (count) {
		case 1: {
			CArcFaceDemoDlg::encrypt_E_vector.re_size(atoi(m_temp));
			temp.Format("恢复测试：%s", m_temp);
			m_temp = strtok(NULL, s);
			EditOut(temp, TRUE);
			break;
		}
		case 2: {
			CArcFaceDemoDlg::encrypt_E_vector.re_coeff(atoi(m_temp));
			temp.Format("恢复测试：%s", m_temp);
			m_temp = strtok(NULL, s);
			EditOut(temp, TRUE);
			break;
		}
		case 3: {
			CArcFaceDemoDlg::encrypt_E_vector.re_ntt(atoi(m_temp));
			temp.Format("恢复测试：%s", m_temp);
			m_temp = strtok(NULL, s);
			EditOut(temp, TRUE);
			break;
		}
		case 4: {
			CArcFaceDemoDlg::encrypt_E_vector.re_poly(atoi(m_temp));
			temp.Format("恢复测试：%s", m_temp);
			m_temp = strtok(NULL, s);
			EditOut(temp, TRUE);
			break;
		}
		case 5: {
			CArcFaceDemoDlg::encrypt_E_vector.re_scale(atoi(m_temp));
			temp.Format("恢复测试：%s", m_temp);
			EditOut(temp, TRUE);
			break;
		}
		default:break;
		}
	}
	infile.close();
	
	infile.open(folderPath + "\\Parms_Info.txt", ios::in);
	if (!infile.is_open())
	{
		AfxMessageBox(_T("没有拥有该注册序列码的用户信息或您没有获取该信息的权限！"));
		return;
	}
	*/
	/*
	char* p_token = new char[2000]; char* p_temp = new char[2000];
	while (!infile.eof()) {
		infile >> p_token;
	}
	p_temp = strtok(p_token, s);
	for (int i = 0; i < CArcFaceDemoDlg::encrypt_probe_p.parms_id().size(); i++) {
		pt[i] = (uint64_t)atoi(p_temp);
		temp.Format("恢复测试：%s", p_temp);
		EditOut(temp, TRUE);
		p_temp = strtok(NULL, s);
	}*/
	//CArcFaceDemoDlg::encrypt_E_vector.re_parms_id(CArcFaceDemoDlg::encrypt_probe_p.parms_id());
	
	//infile.close();
	infile.open(folderPath + "\\Ciphertext.txt", ios::in);
	if (!infile.is_open())
	{
		AfxMessageBox(_T("没有拥有该注册序列码的用户信息或您没有获取该信息的权限！"));
		return;
	}
	IntArray<uint64_t> rep;
	char* c_token = new char[60000001]; char* c_temp = new char[60000001];
	while (!infile.eof()) {
		infile >> c_token;
	}
	string re = c_token;
	string ree = "";
	string reee = "";
	ofstream OutFile; OutFile.open(folderPath + "\\Ciphertext2.txt");
	OutFile << re;
	OutFile.close();
	
	int t_count = 0;
	for (int i = 0;; i++) {
		if (c_token[i] == '*') t_count++;
		if (c_token[i] == '\0') break;
	}
	rep.resize(t_count + 1);
	OutFile.open(folderPath + "\\Ciphertext3.txt");
    c_temp = strtok(c_token, s);
	vector<uint64_t> vec;
	for (int i = 0; i < t_count + 1; i++) {
		uint64_t ttrr = 0;
		string g = "";
		g = c_temp;
		reee += c_temp;
		stringstream geek(g);geek >> ttrr;
		vec.emplace_back(ttrr);
		ree += to_string(vec[i]);
		c_temp = strtok(NULL, s);
	}
	/*
	for (auto it = rep.begin(); it != rep.end(); it++) {
		//ree += c_temp;
		*it = (uint64_t)atoi(c_temp);
		ree += to_string(*it);
		c_temp = strtok(NULL, s);
	}*/
	OutFile << ree;
	OutFile.close();

	OutFile.open(folderPath + "\\Ciphertext4.txt");
	OutFile << reee;
	OutFile.close();
	/*
	string give = "";
	int ooo = 0;
	for (int i = 0; i < re.size(); i++) {
		if (c_token[i] == '*') {
			rep[ooo++] = (uint64_t)atoi(give.c_str());
			give = "";
			continue;
		}
		give += c_token[i];
	}
	str.Format("变换前密文数组长度:%d,变换后长度:%d", t_count + 1, ooo + 1);
	EditOut(str, TRUE);
	AfxMessageBox(_T("1"));
	*/
	CArcFaceDemoDlg::encrypt_E_vector.re_ciphertext(vec);
	infile.close();
	long t1 = GetTickCount64();
	/*Ciphertext encrypt_up =
		get_up_cosine_dist_square(CArcFaceDemoDlg::context, ckks_encoder, evaluator, CArcFaceDemoDlg::encrypt_E_vector,
		CArcFaceDemoDlg::encrypt_probe_p, encryptor, decryptor, CArcFaceDemoDlg::relin_keys, 
			CArcFaceDemoDlg::galois_keys);
	Plaintext plain_a_add_cache;
	vector<double> result_a_add_cache;
	decryptor.decrypt(encrypt_up, plain_a_add_cache);
	ckks_encoder.decode(plain_a_add_cache, result_a_add_cache);

	Ciphertext encrypt_down =
		get_down_cosine_dist_square(CArcFaceDemoDlg::context, ckks_encoder, evaluator, CArcFaceDemoDlg::encrypt_E_vector,
			CArcFaceDemoDlg::encrypt_probe_p, encryptor, decryptor, CArcFaceDemoDlg::relin_keys,
			CArcFaceDemoDlg::galois_keys);
	Plaintext plain_b_add_cache;
	vector<double> result_b_add_cache;
	decryptor.decrypt(encrypt_down, plain_b_add_cache);
	ckks_encoder.decode(plain_b_add_cache, result_b_add_cache);
	*/

	Ciphertext encrypt_R = get_dist(CArcFaceDemoDlg::context, ckks_encoder, evaluator, CArcFaceDemoDlg::encrypt_E_vector,
		CArcFaceDemoDlg::encrypt_probe_p, encryptor, decryptor, CArcFaceDemoDlg::relin_keys, CArcFaceDemoDlg::galois_keys);
	//long t2 = GetTickCount64();
	Plaintext plain_add_cache;
	vector<double> result_add_cache;
	decryptor.decrypt(encrypt_R, plain_add_cache);
	ckks_encoder.decode(plain_add_cache, result_add_cache);
	long t2 = GetTickCount64();
	//str.Format("未采用向量变形共耗时:%dms", t2 - t1);
	//EditOut(str, TRUE);

	str.Format("系统耗时:%dms", T2);
	EditOut(str, TRUE);
	CString resStr1, resStr2, resStr3; 
	resStr1.Format("人脸匹配率: %2f%%", 100 / (1 + exp(14.0879341576 * sqrt(result_add_cache[0]) - 7.2786390745)));
	EditOut(resStr1, TRUE);

	if (result_add_cache[0] < 0.23) AfxMessageBox(_T("认证成功，是同一个人"));
	else AfxMessageBox(_T("不是同一个人"));

	/*
	long t2 = GetTickCount64();
	str.Format("求解加密特征向量距离并解密共耗时:%dms", t2 - t1);
	EditOut(str, TRUE);

	CString resStr1, resStr2, resStr3;
	resStr2.Format("分母: %f，分子：%f", result_a_add_cache[0] , result_b_add_cache[0]);
	EditOut(resStr2, TRUE);

	resStr1.Format("人脸匹配率: %2f%%", 100 * (1 - result_b_add_cache[0] / sqrt(result_a_add_cache[0])));
	EditOut(resStr1, TRUE);

	if (100 * (1 - result_b_add_cache[0] / sqrt(result_a_add_cache[0])) < 0.23) AfxMessageBox(_T("认证成功，是同一个人"));
	else AfxMessageBox(_T("不是同一个人"));
	*/
	delete[]c_token;
	delete[]c_temp;
}
