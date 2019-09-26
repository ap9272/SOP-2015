#ifndef FUNCTIONS_HPP_
#define FUNCTIONS_HPP_

#include "stdafx.h"

using namespace std;
using namespace cv;

void read_csv(const string& filename, vector<imageMat>& images, vector<int>& labels);

void compute_Eigenfaces(vector<imageMat>& images, vector<int>& labels);

imageMat_f compute_mean(vector<imageMat>& images, int Dim);

imageMat_f compute_mean(vector<imageMat_f>& images, int Dim);

void normalize_images(vector<imageMat>& images, imageMat_f meanImage, vector<imageMat_f>& normalizedImages, int Dim);

void normalize_images(vector<imageMat_f>& images, imageMat_f meanImage, vector<imageMat_f>& normalizedImages, int Dim);

void normalize_image(imageMat image, imageMat_f meanImage, imageMat_f normalizedImage, int Dim);

void get_covariance_matrix(vector<imageMat_f>& normalizedImage, imageCovMat covarianceMat, int Dim);

void getEigenfaces(vector<imageMat_f>& normalizedImages, vector<imageMat_f>& eigenFaces, imageCovMat eigenVectorMat);

void jacobi_eigenvalue(int n, double a[], int it_max, double v[],double d[], int &it_num, int &rot_num);

void r8mat_diag_get_vector(int n, double a[], double v[]);

void r8mat_identity(int n, double a[]);

double r8mat_is_eigen_right(int n, int k, double a[], double x[],double lambda[]);

double r8mat_norm_fro(int m, int n, double a[]);

void r8mat_print(int m, int n, double a[], string title);

void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi,int jhi, string title);

void r8vec_print(int n, double a[], string title);

void timestamp();

int getMinEuclideanDist(double ** weights, double * w, int no_of_images, int size);

double getEuclideanDistance(double * w1, double * w2, int size);

void getWeightOneImage(imageMat_f img, vector<imageMat_f>& eigenFaces, double * weights);

void getWeights(vector<imageMat_f>& normalizedImages, vector<imageMat_f>& eigenFaces, double ** weights);

void clear_memory(vector<imageMat>& images);

void clear_memory(vector<imageMat_f>& images);

void clear_memory(imageMat_f image);

#endif