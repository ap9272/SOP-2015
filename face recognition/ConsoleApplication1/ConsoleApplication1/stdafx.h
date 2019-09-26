// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include<dirent.h>
#include<string.h>
# include <cstdlib>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <cstring>

#define LENGTH 92
#define BREADTH 112
#define EPS 10e-08

using namespace std;
using namespace cv;

struct imageMat
{
	uchar *im;
	int length;
	int breadth;
};

struct imageMat_f
{
	double *im;
	int length;
	int breadth;
};

typedef imageMat_f imageCovMat;

/*
struct imageCovMat
{
	double **mat;
	int length;
	int breadth;
};
*/

// TODO: reference additional headers your program requires here
