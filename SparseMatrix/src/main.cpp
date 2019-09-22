#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <Core/Time.hpp>
#include "header.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Eigen;
using namespace std;
using namespace cv;

#define ROUND_UINT(d) ( (unsigned int) ((d) + ((d) > 0 ? 0.5 : -0.5)) )

void motionMat(std::vector<Mat>& motionVec, size_t image_count, size_t rfactor, bool clockwise)
{

	size_t quotient, remainder;

	if(clockwise)
	{
			for( size_t i = 0; i < image_count; i ++)
			{
				Mat motionvec =  Mat::zeros(3,3,CV_32F);
				motionvec.at<float>(0,0) = 1;
				motionvec.at<float>(0,1) = 0;
				motionvec.at<float>(1,0) = 0;
				motionvec.at<float>(1,1) = 1;
				motionvec.at<float>(2,0) = 0;
				motionvec.at<float>(2,1) = 0;
				motionvec.at<float>(2,2) = 1;

				quotient = floor(i/1.0/rfactor);
				remainder = i%rfactor;

				if(quotient%2 == 0)
					motionvec.at<float>(0,2) = remainder/1.0/rfactor;

				else
					motionvec.at<float>(0,2) = (rfactor - remainder -1)/1.0/rfactor;

				motionvec.at<float>(1,2) = quotient/1.0/rfactor;

                std::cout << "motionvec = " << std::endl << motionvec << std::endl << std::endl;
                
				motionVec.push_back(motionvec);

				std::cout<<"image i = "<<i<<", x motion = "<<motionvec.at<float>(0,2)<<", y motion = "<<motionvec.at<float>(1,2)<<std::endl;
			}
	}
	else
	{
			for( size_t i = 0; i < image_count; i ++)
			{
				Mat motionvec = Mat::zeros(3,3,CV_32F);
				motionvec.at<float>(0,0) = 1;
				motionvec.at<float>(0,1) = 0;
				motionvec.at<float>(1,0) = 0;
				motionvec.at<float>(1,1) = 1;
				motionvec.at<float>(2,0) = 0;
				motionvec.at<float>(2,1) = 0;
				motionvec.at<float>(2,2) = 1;

				quotient = floor(i/1.0/rfactor);
				remainder = i%rfactor;
				if(quotient%2 == 0)
					motionvec.at<float>(1,2) = remainder/1.0/rfactor;

				else
					motionvec.at<float>(1,2) = (rfactor - remainder -1)/1.0/rfactor;

				motionvec.at<float>(0,2) = quotient/1.0/rfactor;

				motionVec.push_back(motionvec);

			}
	}

}


Eigen::SparseMatrix<float, Eigen::RowMajor,int> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor)
{
    /* 
    // Debug Info 
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Inside function Dmatrix" << std::endl;
    //std::cout << "rfactor = " << rfactor << std::endl;
    //std::cout << "Src = " << std::endl << Src << std::endl << std::endl;
    //std::cout << "Dest = "<< std::endl << Dest << std::endl << std::endl;
    */

    // CPU computation times
	Core::TimeSpan time1 = Core::getCurrentTime();

	int dim_srcvec = Src.rows * Src.cols;
    int dim_dstvec = Dest.rows * Dest.cols;
    
	Eigen::SparseMatrix<float,Eigen::RowMajor, int> _Dmatrix(dim_srcvec, dim_dstvec);
	
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j< Src.cols; j++)
		{
			int LRindex = i*Src.cols + j;
			for (int m = rfactor*i; m < (i+1)*rfactor; m++)
			{
				for (int n = rfactor*j; n < (j+1)*rfactor; n++)
				{
					int HRindex = m*Dest.cols + n;
					_Dmatrix.coeffRef(LRindex,HRindex) = 1.0/rfactor/rfactor;
					//std::cout<<"_Dmatrix.coeffRef(LRindex,HRindex) = "<<1.0/rfactor/rfactor<<", rfactor = "<<rfactor<<std::endl;
				}
			}
		}
	}

	Core::TimeSpan time2 = Core::getCurrentTime();

/*
    // Debug Info
	Core::TimeSpan timed = time2 - time1;
	std::cout << "Time taken to execute _Dmatrix - " << timed << std::endl;

    ofstream myfile1;
    myfile1.open ("DmatrixCPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing D Matrix" << "\n";
    // myfile1 << "Inside function Dmatrix" << "\n";
    // myfile1 << "rfactor = " << rfactor << std::endl;
    // myfile1 << "Src = " << "\n" << Src << std::endl;
    // myfile1 << "Dest = "<< "\n" << Dest << std::endl << std::endl;
	// myfile1 << "Number of non zeros in _Dmatrix - "<< _Dmatrix.nonZeros() << "\n";
	// myfile1 << "Size of _Dmatrix - " << _Dmatrix.rows() << " X " << _Dmatrix.cols() << std::endl << std::endl;
    // myfile1 << "_Dmatrix is - " << std::endl << MatrixXf(_Dmatrix) << std::endl << std::endl;

    for (int k=0; k<_Dmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Dmatrix,k); it; ++it)
        {
            myfile1 <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();
        }
        myfile1 << "\n";
    }
    myfile1.close();
*/
	return _Dmatrix;
}

Eigen::SparseMatrix<float, Eigen::RowMajor,int> Hmatrix(cv::Mat & Dest, const cv::Mat& kernel)
{
    /* 
    // Debug Info 
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Inside function Hmatrix" << std::endl;
    //std::cout << "kernel - " << std::endl << kernel << std::endl << std::endl;
    //std::cout << "Dest - "<< std::endl << Dest << std::endl << std::endl;
    */

    // CPU computation times
	Core::TimeSpan time1 = Core::getCurrentTime();

    int dim_dstvec = Dest.rows * Dest.cols;

    Eigen::SparseMatrix <float,Eigen::RowMajor, int> _Hmatrix(dim_dstvec, dim_dstvec);
    
    int radius_y = int((kernel.rows-1)/2);
    int radius_x = int((kernel.cols-1)/2);

    for (int i = 0; i < Dest.rows; i++)
    {
        for (int j = 0; j< Dest.cols; j++)
        {
            int index = i*Dest.cols + j;
            for (int m = 0; m < kernel.rows; m++)
            {
                for (int n = 0; n < kernel.cols; n++)
                {
                    int loc = (i-radius_y+m)*Dest.cols + (j-radius_x+n);
                    if (i-radius_y+m >= 0 && i-radius_y+m < Dest.rows && j-radius_x+n >= 0 && j-radius_x+n < Dest.cols && loc < dim_dstvec)
                        _Hmatrix.coeffRef(index,loc) = kernel.at<float>(m,n);
                }
            }
        }
    }

	Core::TimeSpan time2 = Core::getCurrentTime();

/*
    // Debug Info

	Core::TimeSpan timed = time2 - time1;
	std::cout << "Time taken to execute _Hmatrix - " << timed << std::endl;

    ofstream myfile1;
    myfile1.open ("HmatrixCPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing A" << "\n";
    // myfile1 << "Inside function Hmatrix" << "\n";
    // myfile1 << "kernel - " << "\n" << kernel << std::endl;
    // myfile1 << "Dest - "<< "\n" << Dest << std::endl << std::endl;
	// myfile1 << "Number of non zeros in _Hmatrix - "<< _Hmatrix.nonZeros() << "\n";
	// myfile1 << "Size of _Hmatrix - " << _Hmatrix.rows() << " X " << _Hmatrix.cols() << std::endl << std::endl;
    // myfile1 << "_Hmatrix is - " << "\n" << MatrixXf(_Hmatrix) << std::endl << std::endl;

    for (int k=0; k<_Hmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Hmatrix,k); it; ++it)
        {
            myfile1 <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();
        }
        myfile1 << "\n";
    }
    myfile1.close();
*/

	return _Hmatrix;
}

Eigen::SparseMatrix<float, Eigen::RowMajor,int> Mmatrix(cv::Mat &Dest, float deltaX, float deltaY)
{
    /* 
    // Debug Info   
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "deltaX - " << deltaX << std::endl;
    std::cout << "deltaY - " << deltaY << std::endl << std::endl;
    std::cout << "Dest - "<< std::endl << Dest << std::endl << std::endl;
    */   
 
    // CPU computation times
	Core::TimeSpan time1 = Core::getCurrentTime();

	int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Mmatrix(dim_dstvec, dim_dstvec);

	for (int i = 0; i < Dest.rows; i++)
	{
		for(int j = 0; j < Dest.cols; j++)
		{
			if(i < (Dest.rows-std::floor(deltaY)) && j< (Dest.cols-std::floor(deltaX)) && (i+std::floor(deltaY) >= 0) && (j+std::floor(deltaX) >= 0))
			{
				int index = i*Dest.cols + j;
				int neighborUL = (i+std::floor(deltaY))*Dest.cols + (j+std::floor(deltaX));
				int neighborUR = (i+std::floor(deltaY))*Dest.cols + (j+std::floor(deltaX)+1);
				int neighborBR = (i+std::floor(deltaY)+1)*Dest.cols + (j+std::floor(deltaX)+1);
				int neighborBL = (i+std::floor(deltaY)+1)*Dest.cols + (j+std::floor(deltaX));

				if(neighborUL >= 0 && neighborUL < dim_dstvec)
				{
				    float tmp = (i+std::floor(deltaY)+1-(i+deltaY))*(j+std::floor(deltaX)+1-(j+deltaX));
				    if(tmp != 0)
					    _Mmatrix.coeffRef(index, neighborUL) = tmp;
                        //std::cout<< "index is " <<  index << ", neighborUL is "<< neighborUL << "\n";
		        }
				if(neighborUR >= 0 && neighborUR < dim_dstvec)
				{
		            float tmp = (i+std::floor(deltaY)+1-(i+deltaY))*(j+deltaX-(j+std::floor(deltaX)));
                    //std::cout<< "index is " <<  index << ", neighborUR is "<< neighborUR << "\n";
		            if(tmp != 0)
			            _Mmatrix.coeffRef(index, neighborUR) = tmp;
	            }
				if(neighborBR >= 0 && neighborBR < dim_dstvec)
				{
		            float tmp = (i+deltaY-(i+std::floor(deltaY)))*(j+deltaX-(j+std::floor(deltaX)));
                    //std::cout<< "index is " <<  index << ", neighborBR is "<< neighborBR << "\n";
		            if(tmp != 0)				
					    _Mmatrix.coeffRef(index, neighborBR) = tmp;
				}
				if(neighborBL >= 0 && neighborBL < dim_dstvec)
				{
		            float tmp = (i+deltaY-(i+std::floor(deltaY)))*(j+std::floor(deltaX)+1-(j+deltaX));
                    //std::cout<< "index is " <<  index << ", neighborBL is "<< neighborBL << "\n";
		            if(tmp != 0)				
					    _Mmatrix.coeffRef(index, neighborBL) = tmp;
				}
			}

		}
	}

	Core::TimeSpan time2 = Core::getCurrentTime();

/*
    // Debug Info

	Core::TimeSpan timed = time2 - time1;
	std::cout << "Time taken to execute _Mmatrix - " << timed << std::endl << std::endl;

    ofstream myfile1;
    myfile1.open ("MmatrixCPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing M Matrix" << "\n";
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    // myfile << "Inside function Mmatrix" << "\n";
    myfile1 << "deltaX - " << deltaX << std::endl;
    myfile1 << "deltaY - " << deltaY << std::endl;
    myfile1 << "Dest - "<< "\n" << Dest << std::endl << std::endl;
	myfile1 << "Number of non zeros in _Mmatrix - "<< _Mmatrix.nonZeros() << "\n";
	myfile1 << "Size of _Mmatrix - " << _Mmatrix.rows() << " X " << _Mmatrix.cols() << std::endl << std::endl;
    myfile1 << "_Mmatrix is - " << std::endl << MatrixXf(_Mmatrix) << std::endl << std::endl;

    for (int k=0; k<_Mmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Mmatrix,k); it; ++it)
        {
            myfile1 <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();
        }
        myfile1 << "\n";
    }
    myfile1.close();

*/
	return _Mmatrix;
}


Eigen::SparseMatrix<float,Eigen::RowMajor, int> ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix)
{

	int dim_srcvec = Src.rows * Src.cols;
    int dim_dstvec = Dest.rows * Dest.cols;

    //float maxPsfRadius = 3 * rfactor * psfWidth;

    Eigen::SparseMatrix<float,Eigen::RowMajor, int> _DHF(dim_srcvec, dim_dstvec);

    // CPU computation times
	Core::TimeSpan time1 = Core::getCurrentTime();

    DMatrix = Dmatrix(Src, Dest, rfactor);

    // CPU computation times
	Core::TimeSpan time2 = Core::getCurrentTime();

    HMatrix = Hmatrix(Dest, kernel);

    // CPU computation times
	Core::TimeSpan time3 = Core::getCurrentTime();

    MMatrix = Mmatrix(Dest, delta.x, delta.y);

    // CPU computation times
	Core::TimeSpan time4 = Core::getCurrentTime();

    _DHF = DMatrix * (HMatrix * MMatrix);

    // CPU computation times
	Core::TimeSpan time5 = Core::getCurrentTime();

    std::cout << std::endl << std::endl << std::endl;

	Core::TimeSpan timet1 = time2 - time1;
	std::cout << "Time taken to execute _Dmatrix - " << timet1 << std::endl << std::endl;

	Core::TimeSpan timet2 = time3 - time2;
	std::cout << "Time taken to execute _Hmatrix - " << timet2 << std::endl << std::endl;

	Core::TimeSpan timet3 = time4 - time3;
	std::cout << "Time taken to execute _Mmatrix - " << timet3 << std::endl << std::endl;

	Core::TimeSpan timet4 = time5 - time4;
	std::cout << "Time taken for multiplication  - " << timet4 << std::endl << std::endl;

    std::cout << "-----------------------------------------------------------------" << std::endl << std::endl;
    Core::TimeSpan timet = timet1 + timet2 + timet3 + timet4;
	std::cout << "Total time taken to execute _DHFmatrix - " << timet << " seconds" << std::endl << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl << std::endl;

    ofstream myfile1;
    myfile1.open ("DHFmatrixCPU.txt");
  
/*
    // Debug Info

    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing DHF" << "\n";
//	myfile1 << "Number of non zeros in _DHF - "<< _DHF.nonZeros() << "\n";
//	myfile1 << "Size of _DHF - " << _DHF.rows() << " X " << _DHF.cols() << std::endl << std::endl;
//  myfile1 << "_DHF is - " << std::endl << MatrixXf(_DHF) << std::endl << std::endl;
    for (int k=0; k<_DHF.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_DHF,k); it; ++it)
        {
            myfile1 <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();
        }
        myfile1 << "\n";
    }
    myfile1.close();

    _DHF.makeCompressed();
*/
    return _DHF;
}


void Normalization(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& src, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& dst)
{
	for(Eigen::Index c = 0; c < src.rows(); ++c)
	{
		float colsum = 0.0;
		for(typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator itL(src, c); itL; ++itL)
			 colsum += itL.value();

		for(typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator itl(src, c); itl; ++itl)
			dst.coeffRef(itl.row(), itl.col()) = src.coeffRef(itl.row(), itl.col())/colsum;
	}
}

void Gaussiankernel(cv::Mat& dst)
{
	int klim = int((dst.rows-1)/2);

	for(int i = -klim; i <= klim; i++)
	{
		for (int j = -klim; j <= klim; j++)
		{
			float dist = i*i+j*j;
			dst.at<float>(i+klim, j+klim) = 1/(2*M_PI)*exp(-dist/2);
		}
	}

	float normF = cv::sum(dst)[0];
	dst = dst/normF;
}


Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatSq(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& src)
{

	Eigen::SparseMatrix<float,Eigen::RowMajor, int> A2(src.rows(), src.cols());

	for (int k = 0; k < src.outerSize(); ++k)
	{
	   	for (typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator innerit(src,k); innerit; ++innerit)
	   	{
	   		//A2.insert(innerit.row(), innerit.col()) = innerit.value() * innerit.value();
	   		A2.insert(k, innerit.index()) = innerit.value() * innerit.value();
	   		//A2.insert(innerit.row(), innerit.col()) = 0;
	   	}
	}
	A2.makeCompressed();
	return A2;

}


void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex, std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& A, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& AT, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& A2, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& AT2, std::vector<viennacl::compressed_matrix<float> >& DHF, std::vector<viennacl::compressed_matrix<float> >& DHFT, std::vector<viennacl::compressed_matrix<float> > &DHF2, std::vector<viennacl::compressed_matrix<float> > &DHFT2)
{

	Gaussiankernel(kernel);

	cv::Point2f Shifts;
	Shifts.x = motionVec[imgindex].at<float>(0,2)*rfactor;
	Shifts.y = motionVec[imgindex].at<float>(1,2)*rfactor;

	A = ComposeSystemMatrix(Src, Dest, Shifts, rfactor, kernel, DMatrix, HMatrix, MMatrix);

	Normalization(A, A);

	A2 = sparseMatSq(A);

	AT = A.transpose();

	AT2 = A2.transpose();

	viennacl::compressed_matrix<float>tmp_vcl(A.rows(), A.cols(), A.nonZeros());
	viennacl::compressed_matrix<float>tmp_vclT(AT.rows(), AT.cols(), AT.nonZeros());

	viennacl::copy(A, tmp_vcl);
	viennacl::copy(AT, tmp_vclT);

	DHF.push_back(tmp_vcl);
	DHFT.push_back(tmp_vclT);

	viennacl::copy(A2, tmp_vcl);
	viennacl::copy(AT2, tmp_vclT);

	DHF2.push_back(tmp_vcl);
	DHFT2.push_back(tmp_vclT);
	
/*    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing A" << "\n";
//	myfile1 << "Number of non zeros in A - "<< A.nonZeros() << "\n";
//	myfile1 << "Size of A - " << A.rows() << " X " << A.cols() << std::endl << std::endl;
//  myfile1 << "A is - " << std::endl << MatrixXf(A) << std::endl << std::endl;

    for (int k=0; k<A.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(A,k); it; ++it)
        {
            myfile1 <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();
        }
        myfile1 << "\n";
    }
    myfile1.close();
*/

}


int main(int argc, char** argv)
{

    size_t image_count = 1;// M
    int rfactor = 2;//magnification factor
    float psfWidth = 3;


    std::vector<cv::Mat> Src(image_count);
    cv::Mat dest;
    cv::Mat kernel = cv::Mat::zeros(cv::Size(psfWidth, psfWidth), CV_32F);


    std::vector<viennacl::compressed_matrix<float> > DHF; 
    std::vector<viennacl::compressed_matrix<float> > DHFT; 
    std::vector<viennacl::compressed_matrix<float> > DHF2; 
    std::vector<viennacl::compressed_matrix<float> > DHFT2; 
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> DMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> HMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> MMatrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> AT;  // transpose of matrix A_i
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> A;  // matrix A_i
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> AT2; //transpose of matrix B_i
    Eigen::SparseMatrix<float, Eigen::RowMajor, int> A2; // matrix B_i


    /***** Generate motion parameters ******/

    std::vector<cv::Mat> motionvec;
    motionMat(motionvec, image_count, rfactor, true);


    for (size_t i = 0;i < image_count;i++)
    {
        Src[i] = cv::imread("../Images/Test/LR_000" + boost::lexical_cast<std::string> (i+1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);

	    if(! Src[i].data)
            std::cerr<<"No files can be found!"<<std::endl;

        Src[i].convertTo(Src[i], CV_32F);


	    dest = cv::Mat(Src[0].rows * rfactor, Src[0].cols * rfactor, CV_16UC1);
	    cv::resize(Src[0], dest, dest.size(), 0, 0, INTER_CUBIC);

        /***** Generate Matrices A = DHF, inverse A = DHFT and B = DHF2, invere B = DHFT2 ******/
	    GenerateAT(Src[i], dest, i, motionvec, kernel, rfactor, DMatrix, HMatrix, MMatrix, A, AT, A2, AT2, DHF, DHFT, DHF2, DHFT2);

	    std::cout<<"Matrices of image "<<(i+1)<<" done."<<std::endl;
    }

    std::cout<<"CPU calculation is done."<<std::endl;

    return 0;
}
