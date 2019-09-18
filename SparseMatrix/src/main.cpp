
/*

//TODO Clean up the code
//TODO Make the buffers common or dealloc buffers after usage
//TODO Send and get back H matrix
//TODO Timing analysis and comparision
//TODO Write MATRIX in files and compare GPU result with CPU result using diff command
//TODO Add as many comments as possible to make code readable

*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "header.hpp"


#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

// ViennaCl includes
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/tools/timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;
using namespace cv;

#define ROUND_UINT(d) ( (unsigned int) ((d) + ((d) > 0 ? 0.5 : -0.5)) )

//=======================================================================
//  
// GLOBAL VARIABLES
//
//=======================================================================
cl::Program program;
cl::Context context;
cl::CommandQueue que;
cl::Device device;
std::vector<cl::Platform> platforms;

float zero = 0.0f;

//=======================================================================
//
// MOTION MAT FUNC
//
//=======================================================================
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


//=======================================================================
//
// D __ MATRIX FUNC
//
//=======================================================================

// TODO change return type and function prototype so as to return a viennacl compressed matrix

Eigen::SparseMatrix<float, Eigen::RowMajor,int> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor)
{
    // For Timimg Analysis
    cl::Event event_kernel1;
    cl::Event event_read1;
    cl::Event event_read2;
    cl::Event event_read3;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;

    // Parameters to compute total buffer size to be allocated to V, C and R buffers
    std::size_t wgSize      = 16;                                       // Work Group size
	std::size_t dim_srcvec  = Src.rows * Src.cols;                      // Number of Rows in D Matrix
    std::size_t dim_dstvec  = Dest.rows * Dest.cols;                    // Number of Columns in D Matrix
    std::size_t count       = Src.rows * Src.cols* rfactor * rfactor;   // Number of non zero elements

    // Create a new Eigen Matrix
    Eigen::SparseMatrix<float,Eigen::RowMajor, int> _Dmatrix(Src.rows*Src.cols, Dest.rows*Dest.cols);

    // Create host memory
    std::vector<float>      h_GpuV (count);
    std::vector<cl_uint>    h_GpuC (count);
    std::vector<cl_uint>    h_GpuR (count);

    // Initilaization of host memory
    memset(h_GpuV.data(),  0, count *sizeof (float));
    memset(h_GpuC.data(),  0, count *sizeof (cl_uint));
    memset(h_GpuR.data(),  0, count *sizeof (cl_uint));

    // Create device memory 
    cl::Buffer NZ_values        (context, CL_MEM_READ_WRITE, count * sizeof(float));
    cl::Buffer NZ_Columns       (context, CL_MEM_READ_WRITE, count * sizeof(cl_int));
    cl::Buffer NZ_Rows          (context, CL_MEM_READ_WRITE, count * sizeof(cl_int));

    // Copy from host to device memory
    que.enqueueWriteBuffer(NZ_values,     true, 0, count *sizeof (float),   h_GpuV.data(),  NULL, &event_write1);
    que.enqueueWriteBuffer(NZ_Columns,    true, 0, count *sizeof (cl_uint), h_GpuC.data(),  NULL, &event_write2);
    que.enqueueWriteBuffer(NZ_Rows,       true, 0, count *sizeof (cl_uint), h_GpuR.data(),  NULL, &event_write3);

    // Wait until all write events complete
    event_write1.wait();
    event_write2.wait();
    event_write3.wait();

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_D_kernel(program, "SuperAwesome_D_Matrix");

    // Kernel2 arguments
    SuperAwesome_D_kernel.setArg<cl_uint>       (0, rfactor); 
    SuperAwesome_D_kernel.setArg<cl::Buffer>    (1, NZ_values);    
    SuperAwesome_D_kernel.setArg<cl::Buffer>    (2, NZ_Columns);
    SuperAwesome_D_kernel.setArg<cl::Buffer>    (3, NZ_Rows);

    // Launch the kernel2
    que.enqueueNDRangeKernel(SuperAwesome_D_kernel, 0,cl::NDRange(Src.rows, Src.cols),cl::NDRange(wgSize, wgSize), NULL, &event_kernel1);

    // Wait until kernel2 is complete
    event_kernel1.wait();

    // Read values from host to GPU
    que.enqueueReadBuffer(NZ_values,  true, 0, count *sizeof (float),   h_GpuV.data(), NULL, &event_read1);
    que.enqueueReadBuffer(NZ_Columns, true, 0, count *sizeof (cl_uint), h_GpuC.data(), NULL, &event_read2);
    que.enqueueReadBuffer(NZ_Rows,    true, 0, count *sizeof (cl_uint), h_GpuR.data(), NULL, &event_read3);

    // Wait until all copy and writes events complete
    event_read1.wait();
    event_read2.wait();
    event_read3.wait();

    // Compute Eigen from Triple

    // CPU computation time for Triplet
	Core::TimeSpan time_trip1 = Core::getCurrentTime();

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(count);
    for(int k=0; k< count; k++ )
    {
        // Put the values on the stack        
        tripletList.push_back(T(h_GpuR[k],h_GpuC[k],h_GpuV[k])); 
    }
 
    // Populate the Sparse MAtrix
    _Dmatrix.setFromTriplets(tripletList.begin(), tripletList.end());

	Core::TimeSpan time_trip2 = Core::getCurrentTime();

    // Total time consumed by Eigen setFromTriplets function
    Core::TimeSpan timed1 = time_trip2 - time_trip1;

    /* -------------------------------------- DEBUG INFO -------------------------------------- */   

    /* TIMING ANALYSIS START */

    // Timing Analysis
	Core::TimeSpan time1    = OpenCL::getElapsedTime(event_kernel1);
	Core::TimeSpan time2    = OpenCL::getElapsedTime(event_read1);
	Core::TimeSpan time3    = OpenCL::getElapsedTime(event_read2);
	Core::TimeSpan time4    = OpenCL::getElapsedTime(event_read3);
	Core::TimeSpan time5    = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time6    = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time7    = OpenCL::getElapsedTime(event_write3);


    std::cout <<"Kernel time for Val, Columns and Row computation       : " << time1 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix Values      : " << time2 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix Columns     : " << time3 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix RowPointers : " << time4 << std::endl;
    std::cout <<"Write and que time for Values Buffer                   : " << time5 << std::endl;
    std::cout <<"Write and que time for Columns Buffer                  : " << time6 << std::endl;
    std::cout <<"Write and que time for RowPointers Buffer              : " << time7 << std::endl;
    std::cout <<"Time taken to execute Triplet                          : " << timed1 << std::endl;

	Core::TimeSpan timeGPU = time1+time2+time3+time4+time5+time6+time7+timed1;

    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of D matrix on the GPU      : "<< timeGPU << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;

    /* PRINT MATRIX */
    /*
    for (int i = 0; i < count; i++)
    {
        std::cout << "(" << h_GpuR[i] << "," << h_GpuC[i] <<")" << "-->" << h_GpuV[i];
        k++;
        if (k==Dest.cols)
        {
            std::cout << "\n";
            k=0;
        }
    } 
    */

    /*
    for (k=0; k<_Dmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Dmatrix,k); it; ++it)
            {
                std::cout <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();

            }
            std::cout << "\n";
    } 
    */

	return _Dmatrix;
}


//=======================================================================
//
// H __ MATRIX
//
//=======================================================================
Eigen::SparseMatrix<float, Eigen::RowMajor,int> Hmatrix(cv::Mat & Dest, const cv::Mat& kernel)
{
    // For timimg Analysis
    cl::Event event_kernel;
    cl::Event event_read1;
    cl::Event event_read2;
    cl::Event event_read3;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;
	cl::Event event_write4;

    // Parameters to compute total buffer size to be allocated to V, C and R buffers
    int dim_dstvec = Dest.rows * Dest.cols;
    std::size_t wgSize = 16;
	int kernelSize = kernel.rows*kernel.cols;
    std::size_t count  = dim_dstvec*kernelSize;    
    std::size_t size   = count *sizeof (float);
	std::size_t kerSize= kernelSize*sizeof(float);

    // Create a new Eigen Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Hmatrix(dim_dstvec, dim_dstvec);

    // Allocate space for output data from GPU on the host
    std::vector<float>          h_GpuV (count);
    std::vector<unsigned int>   h_GpuR (count);
    std::vector<unsigned int>   h_GpuC (count);

    // kernel matrix to vector conversion (Size 3X3 matrix)
    std::vector<float> vkernel(kernelSize);

//    if (kernel.isContinuous())

		//TODO Really really bad way, but this only works. HAVE TO ASK BHAGYA why above is not working

        vkernel[0] = kernel.at<float>(0,0);
        vkernel[1] = kernel.at<float>(0,1);
        vkernel[2] = kernel.at<float>(0,2);
        vkernel[3] = kernel.at<float>(1,0);
        vkernel[4] = kernel.at<float>(1,2);
        vkernel[5] = kernel.at<float>(2,0);
        vkernel[6] = kernel.at<float>(2,1);
        vkernel[7] = kernel.at<float>(2,2);
		vkernel[8] = kernel.at<float>(1,1);
//    }


    //     

    // Allocate space (buffers) for output data on the device 
    cl::Buffer NZ_Values    (context, CL_MEM_READ_WRITE, size);
    cl::Buffer NZ_Rows      (context, CL_MEM_READ_WRITE, size);
    cl::Buffer NZ_Columns   (context, CL_MEM_READ_WRITE, size);
	cl::Buffer GPU_Kernel   (context, CL_MEM_READ_WRITE, kerSize);

    // Initialize host memory to 0 (0 being useful to debug if errors in V)
    memset(h_GpuV.data(), 0, size);
    memset(h_GpuR.data(), 0, size);
    memset(h_GpuC.data(), 0, size);

    // Initialize device memory
    que.enqueueWriteBuffer(NZ_Values,   true, 0, size, h_GpuV.data(),    NULL, &event_write1);
	que.enqueueWriteBuffer(NZ_Rows,     true, 0, size, h_GpuR.data(),    NULL, &event_write2);
    que.enqueueWriteBuffer(NZ_Columns,  true, 0, size, h_GpuC.data(),    NULL, &event_write3);
	que.enqueueWriteBuffer(GPU_Kernel,  true, 0, kerSize, vkernel.data(),NULL, &event_write4);

    // New kernel object for computation of H matrix
	cl::Kernel SuperAwesome_H_kernel(program, "SuperAwesome_H_Matrix");

    // Set kernel arguments as per required order of semantics
    SuperAwesome_H_kernel.setArg<cl::Buffer>(0,GPU_Kernel);
    SuperAwesome_H_kernel.setArg<cl::Buffer>(1,NZ_Values);   
    SuperAwesome_H_kernel.setArg<cl::Buffer>(2,NZ_Rows); 
    SuperAwesome_H_kernel.setArg<cl::Buffer>(3,NZ_Columns);

    // Launch the kernel
    que.enqueueNDRangeKernel(SuperAwesome_H_kernel, 0, cl::NDRange(Dest.rows, Dest.cols), cl::NDRange(wgSize, wgSize), NULL, &event_kernel);

    // Copy output data from device to host
    que.enqueueReadBuffer(NZ_Rows,      true, 0, size, h_GpuR.data(), NULL, &event_read1);
    que.enqueueReadBuffer(NZ_Values,    true, 0, size, h_GpuV.data(), NULL, &event_read2);
    que.enqueueReadBuffer(NZ_Columns,   true, 0, size, h_GpuC.data(), NULL, &event_read3);
    
    // CPU computation time for Triplet
	Core::TimeSpan time_trip1 = Core::getCurrentTime();

    // Triplet to Eigen-Sparse Conversion
    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(count);

    for(int k=0; k< count; k++ )
    {
        // Put the values on the stack        
        tripletList.push_back(T(h_GpuR[k],h_GpuC[k],h_GpuV[k])); 
    }

    // Populate the Sparse Matrix
    _Hmatrix.setFromTriplets(tripletList.begin(), tripletList.end());

	Core::TimeSpan time_trip2 = Core::getCurrentTime();

    // Total time consumed by Eigen setFromTriplets function
    Core::TimeSpan timeh1 = time_trip2 - time_trip1;

    /* -------------------------------------- DEBUG INFO -------------------------------------- */   

    /* TIMING ANALYSIS START */

    // Timing Analysis
	Core::TimeSpan time1 = OpenCL::getElapsedTime(event_kernel);
	Core::TimeSpan time2 = OpenCL::getElapsedTime(event_read1);
	Core::TimeSpan time3 = OpenCL::getElapsedTime(event_read2);
	Core::TimeSpan time4 = OpenCL::getElapsedTime(event_read3);
	Core::TimeSpan time5 = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time6 = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time7 = OpenCL::getElapsedTime(event_write3);
	Core::TimeSpan time8 = OpenCL::getElapsedTime(event_write4);

    std::cout <<"Kernel time for Val, Columns and Row computation       : " << time1 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix Values      : " << time2 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix Columns     : " << time3 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix RowPointers : " << time4 << std::endl;
    std::cout <<"Write and que time for Values Buffer                   : " << time5 << std::endl;
    std::cout <<"Write and que time for Columns Buffer                  : " << time6 << std::endl;
    std::cout <<"Write and que time for Row Buffer                      : " << time7 << std::endl;
    std::cout <<"Write and que time for Gaussian Kernel Buffer          : " << time8 << std::endl;
    std::cout <<"Time taken to execute Triplet                          : " << timeh1 << std::endl;

	Core::TimeSpan timeGPU = time1+time2+time3+time4+time5+time6+time7+time8+timeh1;

    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of H matrix on the GPU      : "<< timeGPU << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;

    /* PRINT MATRIX */
    /*
    for (int i = 0; i < count; i++)
    {
        std::cout << "(" << h_GpuR[i] << "," << h_GpuC[i] <<")" << "-->" << h_GpuV[i];
        k++;
        if (k==Dest.cols)
        {
            std::cout << "\n";
            k=0;
        }
    } 
    */

    /*
    for (int k=0; k<_Hmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Mmatrix,k); it; ++it)
        {
            std::cout <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();

        }
        std::cout << "\n";
    }
    */

	return _Hmatrix;
}


//=======================================================================
//
// M__MATRIX
//
//=======================================================================
Eigen::SparseMatrix<float, Eigen::RowMajor,int> Mmatrix(cv::Mat &Dest, float deltaX, float deltaY)
{
    // For timimg Analysis
    cl::Event event_kernel;
    cl::Event event_read1;
    cl::Event event_read2;
    cl::Event event_read3;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;

    // Parameters to compute total buffer size to be allocated to V, C and R buffers
	int dim_dstvec      = Dest.rows * Dest.cols;
    std::size_t wgSize  = 16;
    std::size_t count   = Dest.rows * (Dest.cols*4);
    std::size_t size    = count *sizeof (float);

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Mmatrix(dim_dstvec, dim_dstvec);

    // Create host memory
    std::vector<float>  h_GpuV (count);
    std::vector<int>    h_GpuC (count);
    std::vector<int>    h_GpuR (count);
   
    // Initilaization of host memory
    memset(h_GpuR.data(), 0, size);
    memset(h_GpuV.data(), 0, size);
    memset(h_GpuC.data(), 0, size);

    // Create device memory 
    cl::Buffer NZ_Rows(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_values(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Columns(context,CL_MEM_READ_WRITE,size);

    // Copy from host to device memory
	que.enqueueWriteBuffer(NZ_Rows,     true, 0, size, h_GpuR.data(), NULL, &event_write1);
    que.enqueueWriteBuffer(NZ_values,   true, 0, size, h_GpuV.data(), NULL, &event_write2);
    que.enqueueWriteBuffer(NZ_Columns,  true, 0, size, h_GpuC.data(), NULL, &event_write3);

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_M_kernel(program, "SuperAwesome_M_Matrix");

    //kernel arguments
    SuperAwesome_M_kernel.setArg<cl_int>    (0, Dest.rows);
	SuperAwesome_M_kernel.setArg<cl_int>    (1, Dest.cols);
    SuperAwesome_M_kernel.setArg<cl_float>  (2, deltaX); 
    SuperAwesome_M_kernel.setArg<cl_float>  (3, deltaY);
    SuperAwesome_M_kernel.setArg<cl::Buffer>(4, NZ_Rows);
    SuperAwesome_M_kernel.setArg<cl::Buffer>(5, NZ_values);    
    SuperAwesome_M_kernel.setArg<cl::Buffer>(6, NZ_Columns);
    SuperAwesome_M_kernel.setArg<cl_int>    (7, dim_dstvec);

    //launch the kernel
    que.enqueueNDRangeKernel(SuperAwesome_M_kernel, 0,cl::NDRange(Dest.rows, Dest.cols),cl::NDRange(wgSize, wgSize), NULL, &event_kernel);

    que.enqueueReadBuffer(NZ_Rows,      true, 0, size, h_GpuR.data(), NULL, &event_read1);
    que.enqueueReadBuffer(NZ_values,    true, 0, size, h_GpuV.data(), NULL, &event_read2);
    que.enqueueReadBuffer(NZ_Columns,   true, 0, size, h_GpuC.data(), NULL, &event_read3);

    // Compute Eigen from Triple

    // CPU computation time for Triplet
	Core::TimeSpan time_trip1 = Core::getCurrentTime();

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(count);
    int k;

    for(k=0; k< count; k++ )
    {
        // Put the values on the stack        
        tripletList.push_back(T(h_GpuR[k],h_GpuC[k],h_GpuV[k])); 
    }

    // Populate the Sparse MAtrix
    _Mmatrix.setFromTriplets(tripletList.begin(), tripletList.end());

	Core::TimeSpan time_trip2 = Core::getCurrentTime();

    // Total time consumed by Eigen setFromTriplets function
    Core::TimeSpan timem1 = time_trip2 - time_trip1;

    /* -------------------------------------- DEBUG INFO -------------------------------------- */   

    /* TIMING ANALYSIS START */

    // Timing Analysis
	Core::TimeSpan time1    = OpenCL::getElapsedTime(event_kernel);
	Core::TimeSpan time2    = OpenCL::getElapsedTime(event_read1);
	Core::TimeSpan time3    = OpenCL::getElapsedTime(event_read2);
	Core::TimeSpan time4    = OpenCL::getElapsedTime(event_read3);
	Core::TimeSpan time5    = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time6    = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time7    = OpenCL::getElapsedTime(event_write3);

    std::cout <<"Kernel time for Val, Columns and Row computation       : " << time1 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix Values      : " << time2 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix Columns     : " << time3 << std::endl;
    std::cout <<"Read time from kernel to compressed matrix RowPointers : " << time4 << std::endl;
    std::cout <<"Write and que time for Values Buffer                   : " << time5 << std::endl;
    std::cout <<"Write and que time for Columns Buffer                  : " << time6 << std::endl;
    std::cout <<"Write and que time for Row Buffer                      : " << time7 << std::endl;
    std::cout <<"Time taken to execute Triplet                          : " << timem1 << std::endl;

	Core::TimeSpan timeGPU = time1+time2+time3+time4+time5+time6+time7+timem1;

    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of D matrix on the GPU      : "<< timeGPU << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;

    /* PRINT MATRIX */
    /*
    for (int i = 0; i < count; i++)
    {
        std::cout << "(" << h_GpuR[i] << "," << h_GpuC[i] <<")" << "-->" << h_GpuV[i];
        k++;
        if (k==Dest.cols)
        {
            std::cout << "\n";
            k=0;
        }
    } 
    */

    /*
    for (int k=0; k<_Mmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Mmatrix,k); it; ++it)
        {
            std::cout <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();

        }
        std::cout << "\n";
    }
    */

	return _Mmatrix;
}


//=======================================================================
//
// COMPOSE SYSTEM MATRIX FUNC
//
//=======================================================================
Eigen::SparseMatrix<float,Eigen::RowMajor, int> ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel, Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix, Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix)
{

	int dim_srcvec = Src.rows * Src.cols;
    int dim_dstvec = Dest.rows * Dest.cols;

    //float maxPsfRadius = 3 * rfactor * psfWidth;

    Eigen::SparseMatrix<float,Eigen::RowMajor, int> _DHF(dim_srcvec, dim_dstvec);

	// Do calculation on the host side
	Core::TimeSpan time1 = Core::getCurrentTime();
    DMatrix = Dmatrix(Src, Dest, rfactor);
    Core::TimeSpan time2 = Core::getCurrentTime();

	Core::TimeSpan timed = time2 - time1;
    std::cout <<"==================================================" << std::endl;
	std::cout << "Time taken to execute DMatrix - " << timed << std::endl;
    std::cout <<"==================================================" << std::endl;

	// Do calculation on the host side
	Core::TimeSpan time5 = Core::getCurrentTime();
    HMatrix = Hmatrix(Dest, kernel);
    Core::TimeSpan time6 = Core::getCurrentTime();

	Core::TimeSpan timeh = time6 - time5;
    std::cout <<"==================================================" << std::endl;
	std::cout << "Time taken to execute HMatrix - " << timeh << std::endl;
    std::cout <<"==================================================" << std::endl;

	// Do calculation on the host side
	Core::TimeSpan time3 = Core::getCurrentTime();
    MMatrix = Mmatrix(Dest, delta.x, delta.y);
    Core::TimeSpan time4 = Core::getCurrentTime();

	Core::TimeSpan timem = time4 - time3;
    std::cout <<"==================================================" << std::endl;
	std::cout << "Time taken to execute MMatrix - " << timem << std::endl;
    std::cout <<"==================================================" << std::endl;

    //_DHF = DMatrix * (HMatrix * MMatrix);

    //_DHF.makeCompressed();

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

	// viennacl::compressed_matrix<float>tmp_vcl(A.rows(), A.cols(), A.nonZeros());
	// viennacl::compressed_matrix<float>tmp_vclT(AT.rows(), AT.cols(), AT.nonZeros());

	// viennacl::copy(A, tmp_vcl);
	// viennacl::copy(AT, tmp_vclT);

	// DHF.push_back(tmp_vcl);
	// DHFT.push_back(tmp_vclT);

	// viennacl::copy(A2, tmp_vcl);
	// viennacl::copy(AT2, tmp_vclT);

	// DHF2.push_back(tmp_vcl);
	// DHFT2.push_back(tmp_vclT);


}

//=======================================================================
//
// MAIN
//
//=======================================================================
int main(int argc, char** argv)
{

	// ===========================================================
	// Initialization of the GPU
	//
	//============================================================
	
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	size_t i;
	for (i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}

	std::cout << platformId << "\n";
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	
    context = cl::Context(CL_DEVICE_TYPE_GPU, prop);
	//	Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	//	Create a command que
	que = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    //*******************************************************************
    // TODO viennacl::init();
    viennacl::ocl::setup_context(0, context(), device(), que());
    viennacl::ocl::switch_context(0);
    // The above lines are not really required. CAN BE REMOVED!!
	//*******************************************************************

	// Load the source code
	program = OpenCL::loadProgramSource(context, "Kernel.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	//********************************************************************

    size_t image_count = 1;// M //TODO///////////// PLEASE CHANGE THIS WHEN USING
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
        Src[i] = cv::imread("../Images/Test/LR_0001.tif", CV_LOAD_IMAGE_ANYDEPTH);
   
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

