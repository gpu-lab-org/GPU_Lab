#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void SuperAwesome_D_Matrix () {

    Eigen::SparseMatrix<float, Eigen::RowMajor,int> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor)
{
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


}

__kernel void SuperAwesome_H_Matrix () {

}

__kernel void SuperAwesome_M_Matrix () {

}