#include "stdafx.h"
#include "functions.hpp"

void read_csv(const string& filename, vector<imageMat>& images, vector<int>& labels)
{
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file)
		throw std::exception();
	std::string line, path, classlabel;
	// for each line
	while (std::getline(file, line)) 
	{
		// get current line
		std::stringstream liness(line);
		// split line

		//cout << "shita";
		std::getline(liness, path, ';');
		std::getline(liness, classlabel);
		//cout << path << endl;
		// push pack the data
		
		Mat im = cvLoadImage(path.c_str(), 0);

		if (im.empty())
		{
			cout << "Could not read image";
			return;
		}
		//cout << path << endl;
		//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
		//imshow("Display window", im);                   // Show our image inside it.

		//cout << im.type()<<im.rows<<im.cols;
		
		imageMat m;
		m.length = im.rows;
		m.breadth = im.cols;
		m.im = new uchar[im.rows*im.cols];
		//cout << "shitshit";
		for (int r = 0; r < im.rows; ++r)
		{
			for (int c = 0; c < im.cols; ++c)
			{
				m.im[r*im.cols + c] = im.at<uchar>(r, c);
				//cout <<r<<c<<endl;
			}
		}
		
		//cout << (double)im.at<uchar>(im.rows-1,im.cols-1);
		images.push_back(m);
		//cout << "shit";
		labels.push_back(atoi(classlabel.c_str()));
		//cout << "shit";
		
	}
}

void get_covariance_matrix(vector<imageMat_f>& normalizedImage, imageCovMat covarianceMat, int Dim)
{
	for (int n = 0; n < covarianceMat.length; n++)
	{
		for (int m = 0; m < covarianceMat.breadth; ++m)
		{
			covarianceMat.im[n*covarianceMat.breadth + m] = 0;
			for (int i = 0; i < Dim; ++i)
			{
				covarianceMat.im[n*covarianceMat.breadth + m] += normalizedImage[n].im[i] * normalizedImage[m].im[i];
			}
			covarianceMat.im[n*covarianceMat.breadth + m] /= covarianceMat.length;
		}
	}
}

imageMat_f compute_mean(vector<imageMat>& images,  int Dim)
{
	imageMat_f mean;
	mean.length = images[0].length;
	mean.breadth = images[0].breadth;

	mean.im = new double[Dim];
	for (int i = 0; i < Dim; ++i)
	{
		mean.im[i] = 0;
		for (unsigned int k = 0; k < images.size(); ++k)
		{
			mean.im[i] += (double)images[k].im[i];
		}
		mean.im[i] /= images.size();
	}

	return mean;
}

imageMat_f compute_mean(vector<imageMat_f>& images, int Dim)
{
	imageMat_f mean;
	mean.length = images[0].length;
	mean.breadth = images[0].breadth;

	mean.im = new double[Dim];
	for (int i = 0; i < Dim; ++i)
	{
		mean.im[i] = 0;
		for (unsigned int k = 0; k < images.size(); ++k)
		{
			mean.im[i] += images[k].im[i];
		}
		mean.im[i] /= images.size();
	}

	return mean;
}

void normalize_images(vector<imageMat>& images, imageMat_f meanImage, vector<imageMat_f>& normalizedImages, int Dim)
{
	for (unsigned int i = 0; i < images.size(); ++i)
	{
		for (int j = 0; j < Dim; ++j)
		{
			normalizedImages[i].im[j] = images[i].im[j] - meanImage.im[j];
		}
	}
}

void normalize_image(imageMat image, imageMat_f meanImage, imageMat_f normalizedImage, int Dim)
{

	for (int j = 0; j < Dim; ++j)
	{
		normalizedImage.im[j] = image.im[j] - meanImage.im[j];
	}

}

void normalize_images(vector<imageMat_f>& images, imageMat_f meanImage, vector<imageMat_f>& normalizedImages, int Dim)
{
	for (unsigned int i = 0; i < images.size(); ++i)
	{
		for (int j = 0; j < Dim; ++j)
		{
			normalizedImages[i].im[j] = images[i].im[j] - meanImage.im[j];
		}
	}
}

void clear_memory(vector<imageMat>& images)
{
	for (unsigned int i = 0; i < images.size(); ++i)
	{
		delete[] images[i].im;
	}
}

void clear_memory(vector<imageMat_f>& images)
{
	for (unsigned int i = 0; i < images.size(); ++i)
	{
		delete[] images[i].im;
	}
}

void clear_memory(imageMat_f image)
{
	delete[] image.im;
}

void getEigenfaces(vector<imageMat_f>& normalizedImages, vector<imageMat_f>& eigenFaces, imageCovMat eigenVectorMat)
{
	//double * temp;
	for (unsigned int i = 0; i < eigenFaces.size(); i++)
	{
		//temp = eigenVectorMat.im[no_of_images*(no_of_images - i)];
		//temp[j]

		for (int j = 0; j < eigenFaces[i].length*eigenFaces[i].breadth; j++)
		{
			eigenFaces[i].im[j] = 0;
			for (unsigned int k = 0; k < normalizedImages.size(); k++)
			{
				eigenFaces[i].im[j] += normalizedImages[k].im[j] * eigenVectorMat.im[(int)(eigenVectorMat.breadth*(eigenVectorMat.length - i - 1) + k)];
			}
		}
	}
}

void getWeightOneImage(imageMat_f img, vector<imageMat_f>& eigenFaces, double * weights){
	//weights = (double *)malloc(eigenFaces.size()*sizeof(double));
	for (int i = 0; i < eigenFaces.size(); i++)
	{
		weights[i] = 0;
		for (int j = 0; j < img.length*img.breadth; j++)
		{
			weights[i] += img.im[j] * eigenFaces[i].im[j];
		}
	}

}

void getWeights(vector<imageMat_f>& normalizedImages, vector<imageMat_f>& eigenFaces, double ** weights)
{
	for (int i = 0; i <normalizedImages.size(); i++)
	{
		*(weights + i) = (double *)malloc(eigenFaces.size()*sizeof(double));
		getWeightOneImage(normalizedImages[i], eigenFaces, *(weights + i));
	}
}

double getEuclideanDistance(double * w1, double * w2,int size){
	double dist = 0;
	for (int i = 0; i < size; i++)
	{
		//cout<<w2[i]<<w2[i]<< "**" << endl;
		dist += fabs((w1[i] - w2[i])*(w1[i] - w2[i]));
	}
	dist = sqrt(dist);
	return dist;
}

int getMinEuclideanDist(double ** weights, double * w,int no_of_images,int size){
	double m = DBL_MAX,dist=0;
	int mi = 0;
	for (int i = 0; i < no_of_images; i++)
	{
		//cout << "**" <<weights[i][i]<<endl;
		dist = getEuclideanDistance(weights[i], w,size);
		//cout << dist << "**\n";
		if (dist < m){ m = dist; mi = i; }
	}
	return mi;
}

void jacobi_eigenvalue(int n, double a[], int it_max, double v[],double d[], int &it_num, int &rot_num)
//  Parameters:
//
//    Input, int N, the order of the matrix.
//
//    Input, double A[N*N], the matrix, which must be square, real,
//    and symmetric.
//
//    Input, int IT_MAX, the maximum number of iterations.
//
//    Output, double V[N*N], the matrix of eigenvectors.
//
//    Output, double D[N], the eigenvalues, in descending order.
//
//    Output, int &IT_NUM, the total number of iterations.
//
//    Output, int &ROT_NUM, the total number of rotations.
//
{
	double *bw;
	double c;
	double g;
	double gapq;
	double h;
	int i;
	int j;
	int k;
	int l;
	int m;
	int p;
	int q;
	double s;
	double t;
	double tau;
	double term;
	double termp;
	double termq;
	double theta;
	double thresh;
	double w;
	double *zw;

	r8mat_identity(n, v);

	r8mat_diag_get_vector(n, a, d);

	bw = new double[n];
	zw = new double[n];

	for (i = 0; i < n; i++)
	{
		bw[i] = d[i];
		zw[i] = 0.0;
	}
	it_num = 0;
	rot_num = 0;

	while (it_num < it_max)
	{
		it_num = it_num + 1;
		//
		//  The convergence threshold is based on the size of the elements in
		//  the strict upper triangle of the matrix.
		//
		thresh = 0.0;
		for (j = 0; j < n; j++)
		{
			for (i = 0; i < j; i++)
			{
				thresh = thresh + a[i + j*n] * a[i + j*n];
			}
		}

		thresh = sqrt(thresh) / (double)(4 * n);

		if (thresh == 0.0)
		{
			break;
		}

		for (p = 0; p < n; p++)
		{
			for (q = p + 1; q < n; q++)
			{
				gapq = 10.0 * fabs(a[p + q*n]);
				termp = gapq + fabs(d[p]);
				termq = gapq + fabs(d[q]);
				//
				//  Delete tiny offdiagonal elements.
				//
				if (4 < it_num &&
					termp == fabs(d[p]) &&
					termq == fabs(d[q]))
				{
					a[p + q*n] = 0.0;
				}
				//
				//  Otherwise, apply a rotation.
				//
				else if (thresh <= fabs(a[p + q*n]))
				{
					h = d[q] - d[p];
					term = fabs(h) + gapq;

					if (term == fabs(h))
					{
						t = a[p + q*n] / h;
					}
					else
					{
						theta = 0.5 * h / a[p + q*n];
						t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
						if (theta < 0.0)
						{
							t = -t;
						}
					}
					c = 1.0 / sqrt(1.0 + t * t);
					s = t * c;
					tau = s / (1.0 + c);
					h = t * a[p + q*n];
					//
					//  Accumulate corrections to diagonal elements.
					//
					zw[p] = zw[p] - h;
					zw[q] = zw[q] + h;
					d[p] = d[p] - h;
					d[q] = d[q] + h;

					a[p + q*n] = 0.0;
					//
					//  Rotate, using information from the upper triangle of A only.
					//
					for (j = 0; j < p; j++)
					{
						g = a[j + p*n];
						h = a[j + q*n];
						a[j + p*n] = g - s * (h + g * tau);
						a[j + q*n] = h + s * (g - h * tau);
					}

					for (j = p + 1; j < q; j++)
					{
						g = a[p + j*n];
						h = a[j + q*n];
						a[p + j*n] = g - s * (h + g * tau);
						a[j + q*n] = h + s * (g - h * tau);
					}

					for (j = q + 1; j < n; j++)
					{
						g = a[p + j*n];
						h = a[q + j*n];
						a[p + j*n] = g - s * (h + g * tau);
						a[q + j*n] = h + s * (g - h * tau);
					}
					//
					//  Accumulate information in the eigenvector matrix.
					//
					for (j = 0; j < n; j++)
					{
						g = v[j + p*n];
						h = v[j + q*n];
						v[j + p*n] = g - s * (h + g * tau);
						v[j + q*n] = h + s * (g - h * tau);
					}
					rot_num = rot_num + 1;
				}
			}
		}

		for (i = 0; i < n; i++)
		{
			bw[i] = bw[i] + zw[i];
			d[i] = bw[i];
			zw[i] = 0.0;
		}
	}
	//
	//  Restore upper triangle of input matrix.
	//
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < j; i++)
		{
			a[i + j*n] = a[j + i*n];
		}
	}
	//
	//  Ascending sort the eigenvalues and eigenvectors.
	//
	for (k = 0; k < n - 1; k++)
	{
		m = k;
		for (l = k + 1; l < n; l++)
		{
			if (d[l] < d[m])
			{
				m = l;
			}
		}

		if (m != k)
		{
			t = d[m];
			d[m] = d[k];
			d[k] = t;
			for (i = 0; i < n; i++)
			{
				w = v[i + m*n];
				v[i + m*n] = v[i + k*n];
				v[i + k*n] = w;
			}
		}
	}

	delete[] bw;
	delete[] zw;

	return;
}

void r8mat_diag_get_vector(int n, double a[], double v[])
//  Parameters:
//
//    Input, int N, the number of rows and columns of the matrix.
//
//    Input, double A[N*N], the N by N matrix.
//
//    Output, double V[N], the diagonal entries
//    of the matrix.
//
{
	int i;

	for (i = 0; i < n; i++)
	{
		v[i] = a[i + i*n];
	}

	return;
}

void r8mat_identity(int n, double a[])
//  Parameters:
//
//    Input, int N, the order of A.
//
//    Output, double A[N*N], the N by N identity matrix.
//
{
	int i;
	int j;
	int k;

	k = 0;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < n; i++)
		{
			if (i == j)
			{
				a[k] = 1.0;
			}
			else
			{
				a[k] = 0.0;
			}
			k = k + 1;
		}
	}

	return;
}

double r8mat_is_eigen_right(int n, int k, double a[], double x[],double lambda[])
//  Parameters:
//
//    Input, int N, the order of the matrix.
//
//    Input, int K, the number of eigenvectors.
//    K is usually 1 or N.
//
//    Input, double A[N*N], the matrix.
//
//    Input, double X[N*K], the K eigenvectors.
//
//    Input, double LAMBDA[K], the K eigenvalues.
//
//    Output, double R8MAT_IS_EIGEN_RIGHT, the Frobenius norm
//    of the difference matrix A * X - X * LAMBDA, which would be exactly zero
//    if X and LAMBDA were exact eigenvectors and eigenvalues of A.
//
{
	double *c;
	double error_frobenius;
	int i;
	int j;
	int l;

	c = new double[n*k];

	for (j = 0; j < k; j++)
	{
		for (i = 0; i < n; i++)
		{
			c[i + j*n] = 0.0;
			for (l = 0; l < n; l++)
			{
				c[i + j*n] = c[i + j*n] + a[i + l*n] * x[l + j*n];
			}
		}
	}

	for (j = 0; j < k; j++)
	{
		for (i = 0; i < n; i++)
		{
			c[i + j*n] = c[i + j*n] - lambda[j] * x[i + j*n];
		}
	}

	error_frobenius = r8mat_norm_fro(n, k, c);

	delete[] c;

	return error_frobenius;
}

double r8mat_norm_fro(int m, int n, double a[])
//  Parameters:
//
//    Input, int M, the number of rows in A.
//
//    Input, int N, the number of columns in A.
//
//    Input, double A[M*N], the matrix whose Frobenius
//    norm is desired.
//
//    Output, double R8MAT_NORM_FRO, the Frobenius norm of A.
//
{
	int i;
	int j;
	double value;

	value = 0.0;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < m; i++)
		{
			value = value + pow(a[i + j*m], 2);
		}
	}
	value = sqrt(value);

	return value;
}

void r8mat_print(int m, int n, double a[], string title)
//  Parameters:
//
//    Input, int M, the number of rows in A.
//
//    Input, int N, the number of columns in A.
//
//    Input, double A[M*N], the M by N matrix.
//
//    Input, string TITLE, a title.
//
{
	r8mat_print_some(m, n, a, 1, 1, m, n, title);

	return;
}

void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi,int jhi, string title)
//  Parameters:
//
//    Input, int M, the number of rows of the matrix.
//    M must be positive.
//
//    Input, int N, the number of columns of the matrix.
//    N must be positive.
//
//    Input, double A[M*N], the matrix.
//
//    Input, int ILO, JLO, IHI, JHI, designate the first row and
//    column, and the last row and column to be printed.
//
//    Input, string TITLE, a title.
//
{
# define INCX 5

	int i;
	int i2hi;
	int i2lo;
	int j;
	int j2hi;
	int j2lo;

	cout << "\n";
	cout << title << "\n";

	if (m <= 0 || n <= 0)
	{
		cout << "\n";
		cout << "  (None)\n";
		return;
	}
	//
	//  Print the columns of the matrix, in strips of 5.
	//
	for (j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX)
	{
		j2hi = j2lo + INCX - 1;
		if (n < j2hi)
		{
			j2hi = n;
		}
		if (jhi < j2hi)
		{
			j2hi = jhi;
		}
		cout << "\n";
		//
		//  For each column J in the current range...
		//
		//  Write the header.
		//
		cout << "  Col:    ";
		for (j = j2lo; j <= j2hi; j++)
		{
			cout << setw(7) << j - 1 << "       ";
		}
		cout << "\n";
		cout << "  Row\n";
		cout << "\n";
		//
		//  Determine the range of the rows in this strip.
		//
		if (1 < ilo)
		{
			i2lo = ilo;
		}
		else
		{
			i2lo = 1;
		}
		if (ihi < m)
		{
			i2hi = ihi;
		}
		else
		{
			i2hi = m;
		}

		for (i = i2lo; i <= i2hi; i++)
		{
			//
			//  Print out (up to) 5 entries in row I, that lie in the current strip.
			//
			cout << setw(5) << i - 1 << ": ";
			for (j = j2lo; j <= j2hi; j++)
			{
				cout << setw(12) << a[i - 1 + (j - 1)*m] << "  ";
			}
			cout << "\n";
		}
	}

	return;
# undef INCX
}

void r8vec_print(int n, double a[], string title)
//  Parameters:
//
//    Input, int N, the number of components of the vector.
//
//    Input, double A[N], the vector to be printed.
//
//    Input, string TITLE, a title.
//
{
	int i;

	cout << "\n";
	cout << title << "\n";
	cout << "\n";
	for (i = 0; i < n; i++)
	{
		cout << "  " << setw(8) << i
			<< ": " << setw(14) << a[i] << "\n";
	}

	return;
}

void timestamp()
{
# define TIME_SIZE 40

	static char time_buffer[TIME_SIZE];
	const struct std::tm *tm_ptr;
	size_t len;
	std::time_t now;

	now = std::time(NULL);
	tm_ptr = std::localtime(&now);

	len = std::strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr);

	std::cout << time_buffer << "\n";

	return;
# undef TIME_SIZE
}