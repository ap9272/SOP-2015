#include "stdafx.h"
#include "functions.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	vector<imageMat> images;
	vector<int> labels;
	// check for command line arguments
	if (argc != 2) {
		cout << "usage: " << argv[0] << " <csv.ext>" << endl;
		return 1;
	}

	// path to your CSV
	string fn_csv = string(argv[1]);
	// read in the images
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\"." << endl;
		return 1;
	}
	clock_t begin, end;
	double time_spent;

	begin = clock();
	
	//cout << "main" << images.size();
	//get test instances
	imageMat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	// ... and delete them from the vector
	images.pop_back();
	labels.pop_back();

	if (images.size() == 0)
	{
		cout << "Please give more than 1 image";
		return -1;
	}
	if (images.size() != labels.size())
	{
		cout << "No of labels should be equal to number of images";
		return -1;
	}

	int Dim = images[0].length*images[0].breadth;

	imageMat_f meanImage = compute_mean(images, Dim);

	vector<imageMat_f> normalizedImages;

	for (unsigned int i = 0; i < images.size(); ++i)
	{
		imageMat_f m;
		m.length = images[i].length;
		m.breadth = images[i].breadth;
		m.im = new double[Dim];
		normalizedImages.push_back(m);
	}

	normalize_images(images, meanImage, normalizedImages, Dim);

	imageCovMat covarianceMat;
	covarianceMat.length = covarianceMat.breadth = images.size();
	covarianceMat.im = new double[images.size()*images.size()];

	get_covariance_matrix(normalizedImages, covarianceMat, Dim);

	//cout << "Covariance Matrix computed";

	imageCovMat eigenVectorMat;
	eigenVectorMat.length = eigenVectorMat.breadth = images.size();
	eigenVectorMat.im = new double[images.size()*images.size()];

	double *eigenValues;
	eigenValues = (double*)malloc(sizeof(double)*images.size());

	int no_of_iter, no_of_rot;

	jacobi_eigenvalue(images.size(), (double*)covarianceMat.im, 1000, (double*)eigenVectorMat.im, (double*)eigenValues, no_of_iter, no_of_rot);

	vector<imageMat_f> eigenFaces;
	int no_of_images = images.size();

	for (int i = 0; i < no_of_images; ++i)
	{
		imageMat_f m;
		m.length = images[i].length;
		m.breadth = images[i].breadth;
		m.im = new double[Dim];
		eigenFaces.push_back(m);
	}

	getEigenfaces(normalizedImages, eigenFaces, eigenVectorMat);

	imageMat_f mean_eigenfaces = compute_mean(eigenFaces, Dim);

	vector<imageMat_f> normalizedEigen;

	for (unsigned int i = 0; i < images.size(); ++i)
	{
		imageMat_f m;
		m.length = images[i].length;
		m.breadth = images[i].breadth;
		m.im = new double[Dim];
		normalizedEigen.push_back(m);
	}

	normalize_images(eigenFaces, mean_eigenfaces, normalizedEigen, Dim);

	double** weights;
	weights = (double **)malloc(images.size()*sizeof(double *));

	getWeights(normalizedImages, normalizedEigen, weights);
	//cout << weights[0][0];
	//
	imageMat_f normalizedImage;
	normalizedImage.length = images[0].length;
	normalizedImage.breadth = images[0].breadth;
	normalizedImage.im = new double[Dim];

	normalize_image(testSample, meanImage, normalizedImage, Dim);

	double * w = (double *)malloc(90*sizeof(double)); 
	//cout<< normalizedEigen.size();
	getWeightOneImage(normalizedImage, normalizedEigen, w);
	//r8vec_print(images.size(), w, "Weightofimg=");

	int predicted_label = getMinEuclideanDist(weights, w, images.size(), no_of_images);
	cout << testLabel << " predicted --> " << labels[predicted_label]<<"\n";


	//cout << eigenVectorMat.im[9];
	//r8mat_print(images.size(), images.size(), (double*)eigenVectorMat.im, "eigenvectorMat = ");

	//r8vec_print(images.size(), eigenValues, "eigenvalues =");


	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "\nTime spent is : " << time_spent;



	clear_memory(images);
	waitKey(0);
	return 0;
}