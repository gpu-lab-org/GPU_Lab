/*
--------------------- GPU LAB --------------------------
                University of Stuttgart

Team Members:
- Bhagyalaxmi Soumyaji
- Hasan Mahmood
- Mohammed Muddasser

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

viennacl::compressed_matrix<float> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor)
{
    // For Timimg Analysis
    cl::Event event_kernel1;
    cl::Event event_kernel2;
    cl::Event event_copy1;
    cl::Event event_copy2;
    cl::Event event_copy3;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;

    // Parameters to compute total buffer size to be allocated to V, C and R buffers
    std::size_t wgSize      = 16;                                       // Work Group size
	std::size_t dim_srcvec  = Src.rows * Src.cols;                      // Number of Rows in D Matrix
    std::size_t dim_dstvec  = Dest.rows * Dest.cols;                    // Number of Columns in D Matrix
    std::size_t count       = Src.rows * Src.cols * rfactor * rfactor;  // Number of non zero elements

    // Create host memory
    std::vector<float>      h_GpuV (count);
    std::vector<cl_uint>    h_GpuC (count);
    std::vector<cl_uint>    h_GpuRp(dim_srcvec + 1);

    // Initilaization of host memory
    memset(h_GpuV.data(),  0, count *sizeof (float));
    memset(h_GpuC.data(),  0, count *sizeof (cl_uint));
    memset(h_GpuRp.data(), 0, (dim_srcvec + 1) * sizeof(cl_uint));

    // Create device memory 
    cl::Buffer NZ_values        (context, CL_MEM_READ_WRITE, count * sizeof(float));
    cl::Buffer NZ_Columns       (context, CL_MEM_READ_WRITE, count * sizeof(cl_uint));
    cl::Buffer NZ_Row_Pointer   (context, CL_MEM_READ_WRITE,(dim_srcvec + 1) * sizeof(cl_int));

    // Copy from host to device memory
    que.enqueueWriteBuffer(NZ_values,     true, 0, count *sizeof (float),   h_GpuV.data(),  NULL, &event_write1);
    que.enqueueWriteBuffer(NZ_Columns,    true, 0, count *sizeof (cl_uint), h_GpuC.data(),  NULL, &event_write2);
    que.enqueueWriteBuffer(NZ_Row_Pointer,true, 0, (dim_srcvec + 1) *sizeof (cl_uint), h_GpuRp.data(),  NULL, &event_write3);

    // Wait until all write events complete
    event_write1.wait();
    event_write2.wait();
    event_write3.wait();

    // New kernel object for computation of each matrix
	cl::Kernel Kernel_D_Row_Kernel(program, "Kernel_D_Row_Pointer");

    // Kernel1 arguments
    Kernel_D_Row_Kernel.setArg<cl_uint>       (0, rfactor); 
    Kernel_D_Row_Kernel.setArg<cl::Buffer>    (1, NZ_Row_Pointer);

    // Launch the kernel1
    que.enqueueNDRangeKernel(Kernel_D_Row_Kernel, 0, cl::NDRange((Src.rows * Src.cols) + 1 ), cl::NDRange(wgSize * wgSize), NULL, &event_kernel1);

    // Wait until kernel1 is complete
    event_kernel1.wait();

    // ---- To Debug, test and Validate ----
    // Read into buffer to print
    // que.enqueueReadBuffer(NZ_Row_Pointer, true, 0, (dim_srcvec + 1) *sizeof (cl_uint), h_GpuRp.data(),NULL, NULL);

    // New kernel object for computation of each matrix
	cl::Kernel Kernel_D_kernel(program, "Kernel_D_Matrix");

    // Kernel2 arguments
    Kernel_D_kernel.setArg<cl_uint>       (0, rfactor); 
    Kernel_D_kernel.setArg<cl::Buffer>    (1, NZ_values);    
    Kernel_D_kernel.setArg<cl::Buffer>    (2, NZ_Columns);

    // Launch the kernel2
    que.enqueueNDRangeKernel(Kernel_D_kernel, 0,cl::NDRange(Src.rows, Src.cols),cl::NDRange(wgSize, wgSize), NULL, &event_kernel2);

    // Wait until kernel2 is complete
    event_kernel2.wait();

    // ---- To Debug, test and Validate ----
    // que.enqueueReadBuffer(NZ_values,  true, 0, count *sizeof (float), h_GpuV.data(),  NULL, NULL);
    // que.enqueueReadBuffer(NZ_Columns, true, 0, count *sizeof (cl_uint), h_GpuC.data(),NULL, NULL);

    // Initialise the compressed D matrix
    viennacl::compressed_matrix<float> _Dmatrix(dim_srcvec, dim_dstvec, count);

    // Get handles of the compressed D matrix
    cl_int error;
    cl_mem cValues = _Dmatrix.handle().opencl_handle().get();
    error = clRetainMemObject (cValues);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Dmatrix.handle().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Dmatrix_Values (cValues);

    cl_mem cCol_indices = _Dmatrix.handle2().opencl_handle().get();
    error = clRetainMemObject (cCol_indices);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Dmatrix.handle2().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Dmatrix_Col_indices (cCol_indices);

    cl_mem cRow_pointer = _Dmatrix.handle1().opencl_handle().get();
    error = clRetainMemObject (cRow_pointer);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Dmatrix.handle1().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Dmatrix_Row_pointer (cRow_pointer);

    // Read values from host to GPU
    cl_int copyh1  =  que.enqueueCopyBuffer(NZ_values,        _Dmatrix_Values,      0, 0, count * sizeof(float),              NULL, &event_copy1);
    cl_int copyh2  =  que.enqueueCopyBuffer(NZ_Columns,       _Dmatrix_Col_indices, 0, 0, count * sizeof(cl_uint),            NULL, &event_copy2);
    cl_int copyh3  =  que.enqueueCopyBuffer(NZ_Row_Pointer,   _Dmatrix_Row_pointer, 0, 0, (dim_srcvec + 1) * sizeof(cl_uint), NULL, &event_copy3);

    // Wait until all copy and writes events complete
    event_copy1.wait();
    event_copy2.wait();
    event_copy3.wait();

    /* -------------------------------------- DEBUG INFO -------------------------------------- */   

    /* TIMING ANALYSIS START */
    /*
    // Timing Analysis
	Core::TimeSpan time1    = OpenCL::getElapsedTime(event_kernel1);
	Core::TimeSpan time2    = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time3    = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time4    = OpenCL::getElapsedTime(event_write3);
	Core::TimeSpan time5    = OpenCL::getElapsedTime(event_kernel1);
	Core::TimeSpan time6    = OpenCL::getElapsedTime(event_copy1);
	Core::TimeSpan time7    = OpenCL::getElapsedTime(event_copy2);
	Core::TimeSpan time8    = OpenCL::getElapsedTime(event_copy3);

    std::cout << std::endl << std::endl;
    std::cout <<"Kernel time for Row Pointer computation                : " << time1 << std::endl;
    std::cout <<"Write and que time for Values Buffer                   : " << time2 << std::endl;
    std::cout <<"Write and que time for Columns Buffer                  : " << time3 << std::endl;
    std::cout <<"Write and que time for RowPointers Buffer              : " << time4 << std::endl;
    std::cout <<"Kernel time for Val and Columns computation            : " << time5 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Values      : " << time6 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Columns     : " << time7 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix RowPointers : " << time8 << std::endl;

	Core::TimeSpan timeGPUD = time1+time2+time3+time4+time5+time6+time7+time8;

    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of D matrix on the GPU      : "<< timeGPUD << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl << std::endl << std::endl;

    /* DEBUG CODE START */
    /*
    if(copyh1 == CL_SUCCESS)
        std::cout << "copyh1 value " << copyh1 << "\n";

    if(copyh2 == CL_SUCCESS)
        std::cout << "copyh2 value " << copyh2 << "\n";

    if(copyh3 == CL_SUCCESS)
        std::cout << "copyh3 value " << copyh3 << "\n";

    //Test Vector
    std::vector<float> test_values1 (count);
    memset(test_values1.data(), 0, count * sizeof(float));
    que.enqueueReadBuffer(_Dmatrix_Values,  true, 0, count * sizeof(float), test_values1.data(), NULL, NULL);

    std::vector<unsigned int> test_values2 (count);
    memset(test_values2.data(), 0, count * sizeof(cl_uint));
    que.enqueueReadBuffer(_Dmatrix_Col_indices,  true, 0, count * sizeof(cl_uint), test_values2.data(), NULL, NULL);

    std::vector<unsigned int> test_values3 (count);
    memset(test_values3.data(), 0, (dim_srcvec + 1) * sizeof(cl_uint));
    que.enqueueReadBuffer(_Dmatrix_Row_pointer,  true, 0, (dim_srcvec + 1) * sizeof(cl_uint), test_values3.data(), NULL, NULL);
    */

    /* PRINT MATRIX */
    /*
    // Print on conole
    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_DMat.begin(); it != cpu_DMat.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            std::cout << "( " << it_count << " , " << it1->first << ")" << " --> " << it1->second << "\t"; 
        }
        std::cout << "\n";
        it_count++;
    }
    */

    /*
    // Print in file

    // Setup CPU Result matrix
    std::vector< std::map< unsigned int, float> > cpu_DMat(dim_srcvec);

    //Copy from returned device to host memory
    copy(_Dmatrix, cpu_DMat );

    ofstream myfile1;
    myfile1.open ("DmatrixGPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing D" << "\n";

    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_DMat.begin(); it != cpu_DMat.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            myfile1 << "  (" << it_count << "," << it1->first << ")--" << it1->second; 
        }
        myfile1 << "\n";
        it_count++;
    }

    myfile1.close();

    */

	return _Dmatrix;
}


//=======================================================================
//
// H __ MATRIX
//
//=======================================================================
viennacl::compressed_matrix<float> Hmatrix(cv::Mat & Dest, const cv::Mat& kernel)
{
    // For timimg Analysis
    cl::Event event_kernel1;
    cl::Event event_kernel2;
    cl::Event event_copy1;
    cl::Event event_copy2;
    cl::Event event_copy3;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;
    cl::Event event_write4;

    std::size_t dim_dstvec = Dest.rows * Dest.cols;

 	uint kernelSize = kernel.rows * kernel.cols;
	int radius_y = int((kernel.rows-1)/2);
    int radius_x = int((kernel.cols-1)/2);
    // kernel matrix to vector conversion (Size 3X3 matrix)
    std::vector<float> vkernel(kernelSize);

    // Compute total buffer size to be allocated to V, C and R buffers  
    std::size_t wgSize  = 16;
    std::size_t count   = dim_dstvec * kernelSize;   
    std::size_t size    = count *sizeof (float);
	std::size_t kerSize = kernelSize*sizeof(float);

    // Allocate space for output data from GPU on the host
    std::vector<float>      h_GpuV (count);
    std::vector<cl_uint>    h_GpuC (count);
    std::vector<cl_uint>    h_GpuRp(dim_dstvec + 1);

    // Allocate space (buffers) for output data on the device
    cl::Buffer NZ_Values        (context,CL_MEM_READ_WRITE,count * sizeof (float));
    cl::Buffer NZ_Columns       (context,CL_MEM_READ_WRITE,count * sizeof(cl_int));
    cl::Buffer NZ_Row_Pointer   (context,CL_MEM_READ_WRITE,(dim_dstvec + 1) * sizeof(cl_uint));
	cl::Buffer GPU_Kernel       (context,CL_MEM_READ_WRITE,kerSize);

    // Initialize host memory to 0 (0 being useful to debug if errors in V)
    memset(h_GpuV.data(), 0, count * sizeof (float));
    memset(h_GpuC.data(), 0, count * sizeof(cl_uint));
    memset(h_GpuRp.data(), 0, (dim_dstvec + 1) * sizeof(cl_uint));
    
    // Serialize the Gauss Kernel
	for (int m = 0; m < kernel.rows; m++)
    {
        for (int n = 0; n < kernel.cols; n++)
        {
            vkernel[m*kernel.cols +n] = kernel.at<float>(m,n);
            //std::cout<< vkernel[m*kernel.cols +n];            
        }
    } 

    // Initialize device memory
    que.enqueueWriteBuffer(NZ_Row_Pointer,  true, 0, (dim_dstvec + 1) *sizeof (cl_uint), h_GpuRp.data(),    NULL,   &event_write1);
    que.enqueueWriteBuffer(NZ_Values,       true, 0, count *sizeof (float),  h_GpuV.data(),                 NULL,   &event_write2);
    que.enqueueWriteBuffer(NZ_Columns,      true, 0, count * sizeof(cl_uint), h_GpuC.data(),                NULL,   &event_write3);
	que.enqueueWriteBuffer(GPU_Kernel,      true, 0, kerSize, vkernel.data(),                               NULL,   &event_write4);

    // Wait until all write events complete
    event_write1.wait();
    event_write2.wait();
    event_write3.wait();
    event_write4.wait();

    // New kernel object for computation of each matrix
	cl::Kernel Kernel_H_Row_Kernel(program, "Kernel_H_Row_Pointer");

    // Kernel1 arguments
    Kernel_H_Row_Kernel.setArg<cl_uint>       (0, kernel.rows); 
    Kernel_H_Row_Kernel.setArg<cl_uint>       (1, kernel.cols); 
    Kernel_H_Row_Kernel.setArg<cl::Buffer>    (2, NZ_Row_Pointer);

    // Launch the kernel1
    que.enqueueNDRangeKernel(Kernel_H_Row_Kernel, 0, cl::NDRange( dim_dstvec + 1 ), cl::NDRange(wgSize * wgSize), NULL, &event_kernel1);

    // Wait until kernel1 is complete
    event_kernel1.wait();

    // ---- To Debug, test and Validate ----
    // Read into buffer to compute Rowoffset buffer
    // que.enqueueReadBuffer(NZ_Row_Pointer, true, 0, (dim_dstvec + 1) *sizeof (cl_uint), h_GpuRp.data(),NULL, NULL); 

    // New kernel object for computation of H matrix
	cl::Kernel Kernel_H_kernel(program, "Kernel_H_Matrix");

    // Set kernel arguments as per required order of semantics
    Kernel_H_kernel.setArg<cl::Buffer>(0,GPU_Kernel);
    Kernel_H_kernel.setArg<cl::Buffer>(1,NZ_Values);   
    Kernel_H_kernel.setArg<cl::Buffer>(2,NZ_Columns);
    Kernel_H_kernel.setArg<cl_int>    (3,kernel.rows);
	Kernel_H_kernel.setArg<cl_int>    (4,kernel.cols);

    // Launch the kernel
    que.enqueueNDRangeKernel(Kernel_H_kernel, 0, cl::NDRange(Dest.rows, Dest.cols), cl::NDRange(wgSize, wgSize), NULL, &event_kernel2);

    // Wait until kernel2 is complete
    event_kernel2.wait();

    // ---- To Debug, test and Validate ----
    // que.enqueueReadBuffer(NZ_Values,  true, 0, count *sizeof (float),   h_GpuV.data(),  NULL, NULL);
    // que.enqueueReadBuffer(NZ_Columns, true, 0, count *sizeof (cl_uint), h_GpuC.data(),  NULL, NULL);
    
    // Initialise the compressed D matrix
    viennacl::compressed_matrix<float> _Hmatrix(dim_dstvec, dim_dstvec, count);

    // Get handles of the compressed D matrix
    cl_int error;
    cl_mem cValues = _Hmatrix.handle().opencl_handle().get();
    error = clRetainMemObject (cValues);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Hmatrix.handle().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Hmatrix_Values (cValues);

    cl_mem cCol_indices = _Hmatrix.handle2().opencl_handle().get();
    error = clRetainMemObject (cCol_indices);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Hmatrix.handle2().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Hmatrix_Col_indices (cCol_indices);

    cl_mem cRow_pointer = _Hmatrix.handle1().opencl_handle().get();
    error = clRetainMemObject (cRow_pointer);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Hmatrix.handle1().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Hmatrix_Row_pointer (cRow_pointer);

    // Read values from host to GPU
    cl_int copyh1  =  que.enqueueCopyBuffer(NZ_Values,        _Hmatrix_Values,      0, 0, count * sizeof(float),              NULL, &event_copy1);
    cl_int copyh2  =  que.enqueueCopyBuffer(NZ_Columns,       _Hmatrix_Col_indices, 0, 0, count * sizeof(cl_uint),            NULL, &event_copy2);
    cl_int copyh3  =  que.enqueueCopyBuffer(NZ_Row_Pointer,   _Hmatrix_Row_pointer, 0, 0, (dim_dstvec + 1) * sizeof(cl_uint), NULL, &event_copy3);

    // Wait until all copy and writes events complete
    event_copy1.wait();
    event_copy2.wait();
    event_copy3.wait();

    /* -------------------------------------- DEBUG INFO -------------------------------------- */  
    /*
    /* TIMING ANALYSIS START */
    /*
    // Timing Analysis

	Core::TimeSpan time1    = OpenCL::getElapsedTime(event_kernel1);
	Core::TimeSpan time2    = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time3    = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time4    = OpenCL::getElapsedTime(event_write3);
	Core::TimeSpan time5    = OpenCL::getElapsedTime(event_write4);
	Core::TimeSpan time6    = OpenCL::getElapsedTime(event_kernel2);
	Core::TimeSpan time7    = OpenCL::getElapsedTime(event_copy1);
	Core::TimeSpan time8    = OpenCL::getElapsedTime(event_copy2);
	Core::TimeSpan time9    = OpenCL::getElapsedTime(event_copy3);

    std::cout << std::endl << std::endl;
    std::cout <<"Kernel time for Row Pointer computation                : " << time1 << std::endl;
    std::cout <<"Write and que time for Row Pointer Buffer              : " << time2 << std::endl;
    std::cout <<"Write and que time for Values Buffer                   : " << time3 << std::endl;
    std::cout <<"Write and que time for Columns Buffer                  : " << time4 << std::endl;
    std::cout <<"Write and que time for Kernel Buffer                   : " << time5 << std::endl;
    std::cout <<"Kernel time for Val, Columns and Row computation       : " << time6 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Values      : " << time7 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Columns     : " << time8 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix RowPointers : " << time9 << std::endl;

	Core::TimeSpan timeGPUH = time1+time2+time3+time4+time5+time6+time7+time8+time9;

    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of H matrix on the GPU      : "<< timeGPUH << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;

    /*
    // DEBUG MATRIX
    if(copyh1 == CL_SUCCESS)
        std::cout << "copyh1 value " << copyh1 << "\n";

    if(copyh2 == CL_SUCCESS)
        std::cout << "copyh2 value " << copyh2 << "\n";

    if(copyh3 == CL_SUCCESS)
        std::cout << "copyh3 value " << copyh3 << "\n";

    //Test Vector
    std::vector<float> test_values1 (count);
    memset(test_values1.data(), 0, count * sizeof(float));
    que.enqueueReadBuffer(_Hmatrix_Values,  true, 0, count * sizeof(float), test_values1.data(), NULL, NULL);

    std::vector<unsigned int> test_values2 (count);
    memset(test_values2.data(), 0, count * sizeof(cl_uint));
    que.enqueueReadBuffer(_Hmatrix_Col_indices,  true, 0, count * sizeof(cl_uint), test_values2.data(), NULL, NULL);

    std::vector<unsigned int> test_values3 (count);
    memset(test_values3.data(), 0, (dim_dstvec + 1) * sizeof(cl_uint));
    que.enqueueReadBuffer(_Hmatrix_Row_pointer,  true, 0, (dim_dstvec + 1) * sizeof(cl_uint), test_values3.data(), NULL, NULL); 
    */

    /* PRINT MATRIX */
    /*

    // Print on console
    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_HMat.begin(); it != cpu_HMat.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            std::cout << "( " << it_count << " , " << it1->first << ")" << " --> " << it1->second << "\t"; 
        }
        std::cout << "\n";
        it_count++;
    }
    */

    /*
    // Print in file
    // Setup CPU Result matrix
    std::vector< std::map< unsigned int, float> > cpu_HMat(dim_dstvec);

    //Copy from returned device to host memory
    copy(_Hmatrix, cpu_HMat );

    ofstream myfile1;
    myfile1.open ("HmatrixGPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing H" << "\n";

    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_HMat.begin(); it != cpu_HMat.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            myfile1 << "  (" << it_count << "," << it1->first << ")--" << it1->second; 
        }
        myfile1 << "\n";
        it_count++;
    }

    myfile1.close();
    */

	return _Hmatrix;
}


//=======================================================================
//
// M__MATRIX
//
//=======================================================================
viennacl::compressed_matrix<float> Mmatrix(cv::Mat &Dest, float deltaX, float deltaY)
{
    // For timimg Analysis
    cl::Event event_kernel1;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;
    cl::Event event_kernel2;
    cl::Event event_copy1;
    cl::Event event_copy2;
    cl::Event event_copy3;

    // Parameters to compute total buffer size to be allocated to V, C and R buffers
	std::size_t dim_dstvec  = Dest.rows * Dest.cols;
    std::size_t wgSize      = 16;
    std::size_t count       = Dest.rows * Dest.cols * 4;
    std::size_t size        = count *sizeof (float);
    uint buffer_size        = 0;

    // Allocate space for output data from GPU on the host
    std::vector<float>      h_GpuV (count);
    std::vector<cl_uint>    h_GpuC (count);
    std::vector<cl_uint>    h_GpuRp (count);
   
    // Allocate space (buffers) for output data on the device 
    cl::Buffer NZ_Values        (context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Columns       (context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Row_Pointer   (context,CL_MEM_READ_WRITE,size);

    // Initialize host memory to 0 (0 being useful to debug if errors in V)
    memset(h_GpuRp.data(),  0, size);
    memset(h_GpuV.data(),   0, size);
    memset(h_GpuC.data(),   0, size);

    // Copy from host to device memory
    que.enqueueWriteBuffer(NZ_Values,       true, 0, count *sizeof (float), h_GpuV.data(),  NULL, &event_write1);
    que.enqueueWriteBuffer(NZ_Columns,      true, 0, count *sizeof (cl_uint), h_GpuC.data(),  NULL, &event_write2);
	que.enqueueWriteBuffer(NZ_Row_Pointer,  true, 0, (dim_dstvec + 1) *sizeof (cl_uint), h_GpuRp.data(), NULL, &event_write3);

    // Wait until all write events complete
    event_write1.wait();
    event_write2.wait();
    event_write3.wait();

    // New kernel object for computation of each matrix
	cl::Kernel Kernel_M_Row_Kernel(program, "Kernel_M_Row_Pointer");

    // Kernel1 arguments
    Kernel_M_Row_Kernel.setArg<cl::Buffer>    (0, NZ_Row_Pointer);

    // Launch the kernel1
    que.enqueueNDRangeKernel(Kernel_M_Row_Kernel, 0, cl::NDRange((Dest.rows * Dest.cols) + 1 ), cl::NDRange(wgSize * wgSize), NULL, &event_kernel1);

    // Wait until kernel1 is complete
    event_kernel1.wait();

    // ---- To Debug, test and Validate ----e
    // Read into buffer to print
    // que.enqueueReadBuffer(NZ_Row_Pointer, true, 0, (dim_dstvec + 1) *sizeof (cl_uint), h_GpuRp.data(),NULL, NULL);

    // New kernel object for computation of each matrix
	cl::Kernel Kernel_M_kernel(program, "Kernel_M_Matrix");

    //kernel arguments
    Kernel_M_kernel.setArg<cl_float>  (0, deltaX); 
    Kernel_M_kernel.setArg<cl_float>  (1, deltaY);
    Kernel_M_kernel.setArg<cl::Buffer>(2, NZ_Values);    
    Kernel_M_kernel.setArg<cl::Buffer>(3, NZ_Columns);

    //launch the kernel
    que.enqueueNDRangeKernel(Kernel_M_kernel, 0, cl::NDRange(Dest.rows, Dest.cols), cl::NDRange(wgSize, wgSize), NULL, &event_kernel2);

    // Wait until kernel1 is complete
    event_kernel2.wait();

    // ---- To Debug, test and Validate ----
    // que.enqueueReadBuffer(NZ_Values,        true, 0, count *sizeof (float), h_GpuV.data(),   NULL, NULL);
    // que.enqueueReadBuffer(NZ_Columns,       true, 0, count *sizeof (cl_uint), h_GpuC.data(), NULL, NULL);

    // Initialise the compressed D matrix
    viennacl::compressed_matrix<float> _Mmatrix(dim_dstvec, dim_dstvec, count);

    // Get handles of the compressed D matrix
    cl_int error;
    cl_mem cValues = _Mmatrix.handle().opencl_handle().get();
    error = clRetainMemObject (cValues);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Mmatrix.handle().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Mmatrix_Values (cValues);

    cl_mem cCol_indices = _Mmatrix.handle2().opencl_handle().get();
    error = clRetainMemObject (cCol_indices);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Mmatrix.handle2().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Mmatrix_Col_indices (cCol_indices);

    cl_mem cRow_pointer = _Mmatrix.handle1().opencl_handle().get();
    error = clRetainMemObject (cRow_pointer);
    if (error != CL_SUCCESS)
    {
        std::cout << "Error when getting handle for _Mmatrix.handle1().opencl_handle().get(). Error code " << error << "\n";
        cl::errorHandler (error, "clRetainMemObject");
    }
    cl::Buffer _Mmatrix_Row_pointer (cRow_pointer);

    // Read values from host to GPU
    cl_int copyh1  =  que.enqueueCopyBuffer(NZ_Values,        _Mmatrix_Values,      0, 0, count * sizeof(float),              NULL, &event_copy1);
    cl_int copyh2  =  que.enqueueCopyBuffer(NZ_Columns,       _Mmatrix_Col_indices, 0, 0, count * sizeof(cl_uint),            NULL, &event_copy2);
    cl_int copyh3  =  que.enqueueCopyBuffer(NZ_Row_Pointer,   _Mmatrix_Row_pointer, 0, 0, (dim_dstvec + 1) * sizeof(cl_uint), NULL, &event_copy3);

    // Wait until all copy and writes events complete
    event_copy1.wait();
    event_copy2.wait();
    event_copy3.wait();

    /* -------------------------------------- DEBUG INFO -------------------------------------- */  
    /*
    // Timing Analysis

	Core::TimeSpan time1    = OpenCL::getElapsedTime(event_kernel1);
	Core::TimeSpan time2    = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time3    = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time4    = OpenCL::getElapsedTime(event_write3);
	Core::TimeSpan time5    = OpenCL::getElapsedTime(event_kernel2);
	Core::TimeSpan time6    = OpenCL::getElapsedTime(event_copy1);
	Core::TimeSpan time7    = OpenCL::getElapsedTime(event_copy2);
	Core::TimeSpan time8    = OpenCL::getElapsedTime(event_copy3);

    std::cout << std::endl << std::endl;
    std::cout <<"Kernel time for Val, Columns computation               : " << time1 << std::endl;
    std::cout <<"Write and que time for Row Pointer Buffer              : " << time2 << std::endl;
    std::cout <<"Write and que time for Values Buffer                   : " << time3 << std::endl;
    std::cout <<"Write and que time for Columns Buffer                  : " << time4 << std::endl;
    std::cout <<"Kernel time for Row Pointer computation                : " << time5 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Values      : " << time6 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Columns     : " << time7 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix RowPointers : " << time8 << std::endl;

	Core::TimeSpan timeGPUM = time1+time2+time3+time4+time5+time6+time7;

    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of M matrix on the GPU      : "<< timeGPUM << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl << std::endl << std::endl;

    /* DEBUG CODE START */
    /*
    // Debug code to test H Matrix
    // Test code to remove
    
    if(copyh1 == CL_SUCCESS)
        std::cout << "copyh1 value " << copyh1 << "\n";

    if(copyh2 == CL_SUCCESS)
        std::cout << "copyh2 value " << copyh2 << "\n";

    if(copyh3 == CL_SUCCESS)
        std::cout << "copyh3 value " << copyh3 << "\n";

    //Test Vector
    std::vector<float> test_values1 (count);
    memset(test_values1.data(), 0, count * sizeof(float));
    que.enqueueReadBuffer(_Mmatrix_Values,  true, 0, count * sizeof(float), test_values1.data(), NULL, NULL);

    std::vector<unsigned int> test_values2 (count);
    memset(test_values2.data(), 0, count * sizeof(cl_uint));
    que.enqueueReadBuffer(_Mmatrix_Col_indices,  true, 0, count * sizeof(cl_uint), test_values2.data(), NULL, NULL);

    std::vector<unsigned int> test_values3 (count);
    memset(test_values3.data(), 0, (dim_dstvec + 1) * sizeof(cl_uint));
    que.enqueueReadBuffer(_Mmatrix_Row_pointer,  true, 0, (dim_dstvec + 1) * sizeof(cl_uint), test_values3.data(), NULL, NULL); 
    */ 

    /* PRINT MATRIX*/
    /*
    // Print on console
    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_MMat.begin(); it != cpu_MMat.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            std::cout << "( " << it_count << " , " << it1->first << ")" << " --> " << it1->second << "\t"; 
        }
        std::cout << "\n";
        it_count++;
    }
    */

    /*
    // Print in file

    // Setup CPU Result matrix
    std::vector< std::map< unsigned int, float> > cpu_MMat(dim_dstvec);

    //Copy from returned device to host memory
    copy(_Mmatrix, cpu_MMat );

    ofstream myfile1;
    myfile1.open ("MmatrixGPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing M" << "\n";

    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_MMat.begin(); it != cpu_MMat.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            myfile1 << "  (" << it_count << "," << it1->first << ")--" << it1->second; 
        }
        myfile1 << "\n";
        it_count++;
    }

    myfile1.close();
    
    */
	return _Mmatrix;
}


//=======================================================================
//
// COMPOSE SYSTEM MATRIX FUNC
//
//=======================================================================
viennacl::compressed_matrix<float> ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel)
{

	int dim_srcvec = Src.rows * Src.cols;
    int dim_dstvec = Dest.rows * Dest.cols;

    //float maxPsfRadius = 3 * rfactor * psfWidth;

    // CPU computation times
	Core::TimeSpan time1 = Core::getCurrentTime();

	// Do calculation on the host side
    viennacl::compressed_matrix<float> DMatrix = Dmatrix(Src, Dest, rfactor);

    // CPU computation times
	Core::TimeSpan time2 = Core::getCurrentTime();

	// Do calculation on the host side
    viennacl::compressed_matrix<float> HMatrix = Hmatrix(Dest, kernel);

    // CPU computation times
	Core::TimeSpan time3 = Core::getCurrentTime();

	// Do calculation on the host side
    viennacl::compressed_matrix<float> MMatrix = Mmatrix(Dest, delta.x, delta.y);

    // CPU computation times
	Core::TimeSpan time4 = Core::getCurrentTime();

    // ViennaCL sparse matrix- sparse matrix product
    viennacl::compressed_matrix<float> _tempHM =  viennacl::linalg::prod(HMatrix, MMatrix);
    viennacl::compressed_matrix<float> _DHF    =  viennacl::linalg::prod(DMatrix, _tempHM);

	Core::TimeSpan time5 = Core::getCurrentTime();


    /* DEBUG INFO */
    /*
    // Timing Analysis
    
    std::cout << std::endl << std::endl << std::endl;

	Core::TimeSpan timet1 = time2 - time1;
	std::cout << "Time taken to execute _Dmatrix - " << timet1 << std::endl << std::endl;

	Core::TimeSpan timet2 = time3 - time2;
	std::cout << "Time taken to execute _Hmatrix - " << timet2 << std::endl << std::endl;

	Core::TimeSpan timet3 = time4 - time3;
	std::cout << "Time taken to execute _Mmatrix - " << timet3 << std::endl << std::endl;

	Core::TimeSpan timet4 = time5 - time4;
	std::cout << "Time taken for multiplication  - " << timet4 << std::endl << std::endl;

    std::cout << std::endl << std::endl << std::endl;

    std::cout << "-----------------------------------------------------------------" << std::endl << std::endl;
    Core::TimeSpan timet = timet1 + timet2 + timet3 + timet4;
	std::cout << "Total time taken to execute _DHFmatrix - " << timet << " seconds" << std::endl << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl << std::endl;

    std::ofstream log1("Dmatrix_time.txt", std::ios_base::app | std::ios_base::out);
    log1 << timet1 << std::endl;

    std::ofstream log2("Hmatrix_time.txt", std::ios_base::app | std::ios_base::out);
    log2 << timet2 << std::endl;

    std::ofstream log3("Mmatrix_time.txt", std::ios_base::app | std::ios_base::out);
    log3 << timet3 << std::endl;

    std::ofstream log4("Multi_time.txt", std::ios_base::app | std::ios_base::out);
    log4 << timet4 << std::endl;

    std::ofstream log5("DHFmatrix_time.txt", std::ios_base::app | std::ios_base::out);
    log5 << timet << std::endl;

    std::cout << std::endl << std::endl << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"                               DHF Computation complete                          " << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl << std::endl << std::endl;
    
    */

    /*
    // Print on console

    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = _DHF.begin(); it != _DHF.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            std::cout << "( " << it_count << " , " << it1->first << ")" << " --> " << it1->second << "\t"; 
        }
        std::cout << "\n";
        it_count++;
    }
    */

    /*
    // Print in file

    // Setup CPU Result matrix
    std::vector< std::map< unsigned int, float> > cpu_DHF(dim_srcvec);

    //Copy from returned device to host memory
    copy(_DHF, cpu_DHF );

    ofstream myfile1;
    myfile1.open ("DHFmatrixGPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing DHF" << "\n";

    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_DHF.begin(); it != cpu_DHF.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            myfile1 << "  (" << it_count << "," << it1->first << ")--" << it1->second; 
        }
        myfile1 << "\n";
        it_count++;
    }
    myfile1.close();
    
    */

    return _DHF;
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

void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex, std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor)
{

	Gaussiankernel(kernel);

	cv::Point2f Shifts;
	Shifts.x = motionVec[imgindex].at<float>(0,2)*rfactor;
	Shifts.y = motionVec[imgindex].at<float>(1,2)*rfactor;

	viennacl::compressed_matrix<float> A = ComposeSystemMatrix(Src, Dest, Shifts, rfactor, kernel);

    /*
    // Print A in file

    // Number of rows in A
    int dim_srcvec = Src.rows * Src.cols;
    // Number of rows in A
    int dim_dstvec = Dest.rows * Dest.cols;

    // Setup CPU Result matrix
    std::vector< std::map< unsigned int, float> > cpu_A(dim_srcvec);

    //Copy from returned device to host memory
    copy(A, cpu_A );

    ofstream myfile1;
    myfile1.open ("AmatrixGPU.txt");
    
    myfile1 << "-----------------------------------------------------------------------------------------" << std::endl;
    myfile1 << "Printing A" << "\n";

    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_A.begin(); it != cpu_A.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            myfile1 << "  (" << it_count << "," << it1->first << ")--" << it1->second; 
        }
        myfile1 << "\n";
        it_count++;
    }
    myfile1.close();
    */

}

//=======================================================================
//
// MAIN
//
//=======================================================================
int main(int argc, char** argv)
{

	//===========================================================
	// Initialization of the GPU
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
    viennacl::ocl::setup_context(0, context(), device(), que());
    viennacl::ocl::switch_context(0);

	//*******************************************************************

	// Load the source code
	program = OpenCL::loadProgramSource(context, "Kernel.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	//********************************************************************

    size_t image_count = 4;// M
    int rfactor = 2;//magnification factor
    float psfWidth = 3;


    std::vector<cv::Mat> Src(image_count);
    cv::Mat dest;
    cv::Mat kernel = cv::Mat::zeros(cv::Size(psfWidth, psfWidth), CV_32F);

    /***** Generate motion parameters ******/

    std::vector<cv::Mat> motionvec;
    motionMat(motionvec, image_count, rfactor, true);

    /*
    // For performance analysis and plotting average computation times
    for (size_t i = 0;i < 10; i++)
    {
    */
        for (size_t i = 0;i < image_count;i++)
        {
            // 1000 pixel X 1000 pixel image            
            //Src[i] = cv::imread("../Images/Test/LR_000" + boost::lexical_cast<std::string> (i+1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);

            // 128 pixel X 128 pixel image            
            Src[i] = cv::imread("../Images/Cameraman/LR"+ boost::lexical_cast<std::string> (i+1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);

            // 5 pixel X 5 pixel sample test image for testing
            // Src[i] = (Mat_<float>(5,5) << 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1);

            // 3 pixel X 3 pixel sample test image for testing
            // Src[i] = (Mat_<float>(3,3) << 10, 150, 67, 120, 34, 200, 0, 255, 50);
       
	        if(! Src[i].data)
                    std::cerr<<"No files can be found!"<<std::endl;

            Src[i].convertTo(Src[i], CV_32F);


	        dest = cv::Mat(Src[0].rows * rfactor, Src[0].cols * rfactor, CV_16UC1);
	        cv::resize(Src[0], dest, dest.size(), 0, 0, INTER_CUBIC);

            /***** Generate Matrices A = DHF, inverse A = DHFT and B = DHF2, invere B = DHFT2 ******/
            // Changed the corresponding function prototypes accordingly
	        GenerateAT(Src[i], dest, i, motionvec, kernel, rfactor);

	        std::cout<<"Matrices of image "<<(i+1)<<" done."<<std::endl;
        }
    /*
    }
    */

    std::cout<<"CPU calculation is done."<<std::endl;

    return 0;
}
