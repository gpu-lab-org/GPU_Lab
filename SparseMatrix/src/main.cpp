
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
cl::CommandQueue queue;
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
    // TODO Remove Eige and return a viennacl compressed matrix
    Eigen::SparseMatrix<float,Eigen::RowMajor, int> _EDmatrix(Src.rows*Src.cols, Dest.rows*Dest.cols);

    // For Timimg Analysis
    cl::Event event_kernel1;
    cl::Event event_kernel2;
    cl::Event event_copy1;
    cl::Event event_copy2;
    cl::Event event_copy3;
    cl::Event event_write1;
    cl::Event event_write2;
    cl::Event event_write3;

    std::size_t wgSize      = 16;                                       // Work Group size
	std::size_t dim_srcvec  = Src.rows * Src.cols;                      // Number of Rows in D Matrix
    std::size_t dim_dstvec  = Dest.rows * Dest.cols;                    // Number of Columns in D Matrix
    std::size_t count  = Src.rows * Src.cols* rfactor * rfactor;        // Number of non zero elements

    // Create host memory
    std::vector<float>      h_GpuV (count);
    std::vector<cl_uint>    h_GpuC (count);
    std::vector<cl_uint>    h_GpuRp(dim_srcvec + 1);

    // Initilaization of host memory
    memset(h_GpuV.data(), 0, count *sizeof (float));
    memset(h_GpuC.data(), 0, count *sizeof (cl_uint));
    memset(h_GpuRp.data(), 0, (dim_srcvec + 1) * sizeof(cl_uint));

    // TODO Calculate values of Row pointer in CPU. Has to be done in GPU.
	for (int i = 0; i < (dim_srcvec + 1); i++)
	{
		h_GpuRp[i] = i*4;
    }

    // Create device memory 
    cl::Buffer NZ_values        (context,CL_MEM_READ_WRITE,count * sizeof(float));
    cl::Buffer NZ_Columns       (context,CL_MEM_READ_WRITE,count * sizeof(cl_int));
    cl::Buffer NZ_Row_Pointer   (context,CL_MEM_READ_WRITE,(dim_srcvec + 1) * sizeof(cl_int));

    // Copy from host to device memory
    queue.enqueueWriteBuffer(NZ_values,     true, 0, count *sizeof (float),   h_GpuV.data(),  NULL, &event_write1);
    queue.enqueueWriteBuffer(NZ_Columns,    true, 0, count *sizeof (cl_uint), h_GpuC.data(),  NULL, &event_write2);
    queue.enqueueWriteBuffer(NZ_Row_Pointer,true, 0, (dim_srcvec + 1) *sizeof (cl_uint), h_GpuRp.data(),  NULL, &event_write3);

    // Wait until all write events complete
    event_write1.wait();
    event_write2.wait();
    event_write3.wait();

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_D_Row_Kernel(program, "SuperAwesome_D_Row_Pointer");

    // Kernel1 arguments
    SuperAwesome_D_Row_Kernel.setArg<cl_uint>       (0, rfactor); 
    SuperAwesome_D_Row_Kernel.setArg<cl::Buffer>    (1, NZ_Row_Pointer);

    // Launch the kernel1
    queue.enqueueNDRangeKernel(SuperAwesome_D_Row_Kernel, 0, cl::NDRange((Src.rows * Src.cols) + 1 ), cl::NDRange(wgSize * wgSize), NULL, &event_kernel1);

    // TODO remove later during delivery - Read into buffers to print
    // Read into buffer to print
    //queue.enqueueReadBuffer(NZ_Row_Pointer, true, 0, (dim_srcvec + 1) *sizeof (cl_uint), h_GpuRp.data(),NULL, NULL);

    // Wait until kernel1 is complete
    event_kernel1.wait();

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_D_kernel(program, "SuperAwesome_D_Matrix");

    // Kernel2 arguments
    SuperAwesome_D_kernel.setArg<cl_uint>       (0, rfactor); 
    SuperAwesome_D_kernel.setArg<cl::Buffer>    (1, NZ_values);    
    SuperAwesome_D_kernel.setArg<cl::Buffer>    (2, NZ_Columns);

    // Launch the kernel2
    queue.enqueueNDRangeKernel(SuperAwesome_D_kernel, 0,cl::NDRange(Src.rows, Src.cols),cl::NDRange(wgSize, wgSize), NULL, &event_kernel2);

    // Wait until kernel2 is complete
    event_kernel2.wait();

    // TODO remove later during delivery - Read into buffers to print
    //queue.enqueueReadBuffer(NZ_values, true, 0, count *sizeof (float), h_GpuV.data(),NULL, NULL);
    //queue.enqueueReadBuffer(NZ_Columns, true, 0, count *sizeof (cl_uint), h_GpuC.data(),NULL, NULL);

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

    // Copy data to handles TODO remove copyh1, copyh2 and writeh1 after testing
    cl_int copyh1  =  queue.enqueueCopyBuffer(NZ_values,  _Dmatrix_Values,      0, 0, count * sizeof(float),   NULL, &event_copy1);
    cl_int copyh2  =  queue.enqueueCopyBuffer(NZ_Columns, _Dmatrix_Col_indices, 0, 0, count * sizeof(cl_uint), NULL, &event_copy2);
    cl_int copyh3  =  queue.enqueueCopyBuffer(NZ_Row_Pointer, _Dmatrix_Row_pointer, 0, 0, (dim_srcvec + 1) * sizeof(cl_uint), NULL, &event_copy3);

    // Wait until all copy and writes events complete
    event_copy1.wait();
    event_copy2.wait();
    event_copy3.wait();

    /* DEBUG CODE START */
    // TODO Test code to remove
    if(copyh1 == CL_SUCCESS)
        std::cout << "copyh1 value " << copyh1 << "\n";

    if(copyh2 == CL_SUCCESS)
        std::cout << "copyh2 value " << copyh2 << "\n";

    if(copyh3 == CL_SUCCESS)
        std::cout << "writeh1 value " << copyh3 << "\n";

    //Test Vector
    std::vector<float> test_values1 (count);
    memset(test_values1.data(), 0, count * sizeof(float));
    queue.enqueueReadBuffer(_Dmatrix_Values,  true, 0, count * sizeof(float), test_values1.data(), NULL, NULL);

    std::vector<unsigned int> test_values2 (count);
    memset(test_values2.data(), 0, count * sizeof(cl_uint));
    queue.enqueueReadBuffer(_Dmatrix_Col_indices,  true, 0, count * sizeof(cl_uint), test_values2.data(), NULL, NULL);

    std::vector<unsigned int> test_values3 (count);
    memset(test_values3.data(), 0, (dim_srcvec + 1) * sizeof(cl_uint));
    queue.enqueueReadBuffer(_Dmatrix_Row_pointer,  true, 0, (dim_srcvec + 1) * sizeof(cl_uint), test_values3.data(), NULL, NULL);

    /* TEST CODE END */

    /* TESTING MULTIPLICATION START CODE - To be moved to ComposeSystemMatrix() */

    // Setup Identity matrix for testing Multiplication
    // Setup CPU matrix
    std::vector< std::map< unsigned int, float> > cpu_sparse_Imatrix(dim_dstvec);

    // Copy data to CPU matrix - Sequential and time consuming
    for (unsigned int i = 0; i < dim_dstvec; i++)
    {
        cpu_sparse_Imatrix[i][i] = 1;
    }
    cpu_sparse_Imatrix[0][1] = 1;
    cpu_sparse_Imatrix[1][2] = 1;
    cpu_sparse_Imatrix[2][3] = 1;

    // Create ViennaCl compressed Identity matrix for testing
    viennacl::compressed_matrix<float> gpu_sparse_Imatrix(dim_dstvec, dim_dstvec);

    // Copy Identity matrix from CPU to device memory
    copy(cpu_sparse_Imatrix, gpu_sparse_Imatrix );

    // Create ViennaCl compressed Result matrix for testing 
    viennacl::compressed_matrix<float> gpu_sparse_Rmatrix(dim_srcvec, dim_dstvec);

    // CPU computation times
	Core::TimeSpan time10 = Core::getCurrentTime();

    viennacl::tools::timer timer;
    double exec_time;

    timer.start();

    // ViennaCL sparse matrix- sparse matrix product
    gpu_sparse_Rmatrix = viennacl::linalg::prod(_Dmatrix, gpu_sparse_Imatrix);

    exec_time = timer.get();

	Core::TimeSpan time11 = Core::getCurrentTime();

    // Setup CPU Result matrix
    std::vector< std::map< unsigned int, float> > cpu_sparse_Rmatrix(dim_srcvec);

    //Copy from returned device to host memory
    copy(gpu_sparse_Rmatrix, cpu_sparse_Rmatrix );

    // Copy to EDmatrix
    // CPU computation times
	Core::TimeSpan time12 = Core::getCurrentTime();

    viennacl::copy(gpu_sparse_Rmatrix, _EDmatrix);

	Core::TimeSpan time13 = Core::getCurrentTime();

    /* ----------------------------------------- END -------------------------------------------*/

    /* -------------------------------------- DEBUG INFO -------------------------------------- */   

    // Print matrx sizes
    std::cout <<  "_Dmatrix nnz  - " << _Dmatrix.nnz() << std::endl ;
    std::cout <<  "_Dmatrix rows - " << _Dmatrix.size1() << std::endl;
    std::cout <<  "_Dmatrix cols - " << _Dmatrix.size2() << std::endl;
    std::cout <<  "gpu_sparse_Imatrix nnz  - " << gpu_sparse_Imatrix.nnz()   << std::endl ;
    std::cout <<  "gpu_sparse_Imatrix rows - " << gpu_sparse_Imatrix.size1() << std::endl;
    std::cout <<  "gpu_sparse_Imatrix cols - " << gpu_sparse_Imatrix.size2() << std::endl;
    std::cout <<  "gpu_sparse_Rmatrix nnz  - " << gpu_sparse_Rmatrix.nnz()   << std::endl ;
    std::cout <<  "gpu_sparse_Rmatrix rows - " << gpu_sparse_Rmatrix.size1() << std::endl;
    std::cout <<  "gpu_sparse_Rmatrix cols - " << gpu_sparse_Rmatrix.size2() << std::endl; 

    /* TIMING ANALYSIS START */

    // Timing Analysis
	Core::TimeSpan time1 = OpenCL::getElapsedTime(event_kernel1);
	Core::TimeSpan time2 = OpenCL::getElapsedTime(event_kernel2);
	Core::TimeSpan time3 = OpenCL::getElapsedTime(event_copy1);
	Core::TimeSpan time4 = OpenCL::getElapsedTime(event_copy2);
	Core::TimeSpan time5 = OpenCL::getElapsedTime(event_copy2);
	Core::TimeSpan time6 = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time7 = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time8 = OpenCL::getElapsedTime(event_write3);

    std::cout <<"Write and queue time for Values Buffer                 : " << time6 << std::endl;
    std::cout <<"Write and queue time for Columns Buffer                : " << time7 << std::endl;
    std::cout <<"Write and queue time for RowPointers Buffer            : " << time8 << std::endl;
    std::cout <<"Kernel time for Row Pointer computation                : " << time1 << std::endl;
    std::cout <<"Kernel time for Val and Columns computation            : " << time2 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Values      : " << time3 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix Columns     : " << time4 << std::endl;
    std::cout <<"Copy time from kernel to compressed matrix RowPointers : " << time5 << std::endl;

	Core::TimeSpan timeGPU = time1+time2+time3+time4+time5+time6+time7+time8;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;
    std::cout <<"Total Time for computation of D matrix on the GPU      : "<< timeGPU << std::endl;
    std::cout <<"---------------------------------------------------------------------------------" << std::endl;

	Core::TimeSpan timed1 = time11 - time10;
	std::cout << "Time taken to execute to viennacl::linalg::prod computed using CPU timer              : " << timed1    << std::endl;
    std::cout << "Time taken to execute to viennacl::linalg::prod computed using viennacl timer support : " << exec_time << " seconds" <<std::endl;

	Core::TimeSpan timed2 = time13 - time12;
	std::cout << "Time taken to execute copy to convert from compressed matrix to eigen - " << timed2 << std::endl;

    /* TIMING ANALYSIS END */
/*    
    // Print Row pointer
    for (int i = 0; i < (dim_srcvec + 1); i++)
    {
        std::cout << h_GpuRp[i] << "\n" ;
    }

*/

/*
    // Print D Matrix values
    int k = 0;
    for (int i = 0; i < count; i++)
    {
        std::cout << "(" << h_GpuC[i] <<")" << "-->" << h_GpuV[i] << "--> ; ( " << test_values2[i]  <<")" << " ---> " << test_values1[i] << "\n";
        //std::cout << "(" << h_GpuR[i] << ", " << h_GpuC[i] <<")" << "-->" << h_GpuV[i] << "\t";

        k++;
        if (k==Dest.cols)
        {
            std::cout << "\n";
            k=0;
        }
    }
*/

/*
    //Print multiplication result
    int it_count = 0;
    for(std::vector< std::map< unsigned int, float> >::iterator it = cpu_sparse_Rmatrix.begin(); it != cpu_sparse_Rmatrix.end(); ++it)
    {
        for(auto it1=it->begin(); it1!=it->end(); ++it1)
        {
            std::cout << "( " << it_count << " , " << it1->first << ")" << " --> " << it1->second << std::endl; 
        }
        it_count++;
    }
*/

	return _EDmatrix;
}


//=======================================================================
//
// H __ MATRIX
//
//=======================================================================
Eigen::SparseMatrix<float, Eigen::RowMajor,int> Hmatrix(cv::Mat & Dest, const cv::Mat& kernel)
{
	// New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_H_Matrix(program, "SuperAwesome_H_Matrix");

    int dim_dstvec = Dest.rows * Dest.cols;
	std::vector<float> kernelgpu (kernel.rows * kernel.cols);
    std::size_t wgSize = 16;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Hmatrix(dim_dstvec, dim_dstvec);

	for (int m = 0; m < kernel.rows; m++)
            {
                for (int n = 0; n < kernel.cols; n++)
                {
                    kernelgpu[m*kernel.cols +n] = kernel.at<float>(m,n);
                    //std::cout<< kernelgpu[m*kernel.cols +n];            
                }
            }
	std::size_t count  = Dest.rows * 9;
    std::size_t size   = count *sizeof (float);
    std::vector<float> h_GpuV (count);
    std::vector<int> h_GpuC (count);
    std::vector<int> h_GpuR (count);
   
    // Create buffer 
    cl::Buffer NZ_Rows(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_values(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Columns(context,CL_MEM_READ_WRITE,size);
	cl::Buffer GPU_kernel(context,CL_MEM_READ_WRITE,size);

    //initialization
    memset(h_GpuR.data(), 0, size);
    memset(h_GpuV.data(), 0, size);
    memset(h_GpuC.data(), 0, size);
	queue.enqueueWriteBuffer(NZ_Rows, true, 0, size, h_GpuR.data());
    queue.enqueueWriteBuffer(NZ_values, true, 0, size, h_GpuV.data());
    queue.enqueueWriteBuffer(NZ_Columns, true, 0, size, h_GpuC.data());
	queue.enqueueWriteBuffer(GPU_kernel, true, 0, size, kernelgpu.data());

    //kernel arguments
    SuperAwesome_H_Matrix.setArg<cl_int>(0,kernel.rows);
	SuperAwesome_H_Matrix.setArg<cl_int>(1,kernel.cols);
    SuperAwesome_H_Matrix.setArg<cl_float>(2,Dest.cols);
    SuperAwesome_H_Matrix.setArg<cl_float>(3,Dest.rows); 
    SuperAwesome_H_Matrix.setArg<cl_float>(4,dim_dstvec); 
	SuperAwesome_H_Matrix.setArg<cl::Buffer>(5,GPU_kernel);
    SuperAwesome_H_Matrix.setArg<cl::Buffer>(6,NZ_Rows);
    SuperAwesome_H_Matrix.setArg<cl::Buffer>(7,NZ_values);    
    SuperAwesome_H_Matrix.setArg<cl::Buffer>(8,NZ_Columns);

    //launch the kernel
    queue.enqueueNDRangeKernel(SuperAwesome_H_Matrix, 0,cl::NDRange(1, 1),cl::NDRange(wgSize, wgSize), NULL, NULL);
    

    queue.enqueueReadBuffer(NZ_Rows, true, 0, size, h_GpuR.data(),NULL,NULL);
    queue.enqueueReadBuffer(NZ_values, true, 0, size, h_GpuV.data(),NULL,NULL);
    queue.enqueueReadBuffer(NZ_Columns, true, 0, size, h_GpuC.data(),NULL,NULL);

    
//--------------------------------
    int k = 0;
    for (int i = 0; i < count; i++)
    {
        std::cout << "(" << h_GpuR[i] << "," << h_GpuC[i] <<")" << "-->" << h_GpuV[i];
        k++;
        if (k== 9)
        {
            std::cout << "\n";
            k=0;
        }
    } 

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

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_M_Matrix(program, "SuperAwesome_M_Matrix");

	int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Mmatrix(dim_dstvec, dim_dstvec);

    //-------------------------------
    std::size_t wgSize = 16;

    //TODO clean up the code
    // TODO BUffer sizes etc
    std::size_t count  = Dest.rows * (Dest.cols*4);
    std::size_t size   = count *sizeof (float);
    std::vector<float> h_GpuV (count);
    std::vector<int> h_GpuC (count);
    std::vector<int> h_GpuR (count);
   
    // Create buffer 
    cl::Buffer NZ_Rows(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_values(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Columns(context,CL_MEM_READ_WRITE,size);

    //initialization
    memset(h_GpuR.data(), 0, size);
    memset(h_GpuV.data(), 0, size);
    memset(h_GpuC.data(), 0, size);
	queue.enqueueWriteBuffer(NZ_Rows, true, 0, size, h_GpuR.data(),NULL,&event_write1);
    queue.enqueueWriteBuffer(NZ_values, true, 0, size, h_GpuV.data(),NULL,&event_write2);
    queue.enqueueWriteBuffer(NZ_Columns, true, 0, size, h_GpuC.data(),NULL,&event_write3);

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_M_kernel(program, "SuperAwesome_M_Matrix");

    //kernel arguments
    SuperAwesome_M_kernel.setArg<cl_int>(0,Dest.rows);
	SuperAwesome_M_kernel.setArg<cl_int>(1,Dest.cols);
    SuperAwesome_M_kernel.setArg<cl_float>(2,deltaX); 
    SuperAwesome_M_kernel.setArg<cl_float>(3,deltaY);
    SuperAwesome_M_kernel.setArg<cl::Buffer>(4,NZ_Rows);
    SuperAwesome_M_kernel.setArg<cl::Buffer>(5,NZ_values);    
    SuperAwesome_M_kernel.setArg<cl::Buffer>(6,NZ_Columns);
    SuperAwesome_M_kernel.setArg<cl_int>(7,dim_dstvec);

    //launch the kernel
    queue.enqueueNDRangeKernel(SuperAwesome_M_kernel, 0,cl::NDRange(Dest.rows, Dest.cols),cl::NDRange(wgSize, wgSize), NULL, &event_kernel);
    

    queue.enqueueReadBuffer(NZ_Rows, true, 0, size, h_GpuR.data(),NULL,&event_read1);
    queue.enqueueReadBuffer(NZ_values, true, 0, size, h_GpuV.data(),NULL,&event_read2);
    queue.enqueueReadBuffer(NZ_Columns, true, 0, size, h_GpuC.data(),NULL,&event_read3);

	Core::TimeSpan time1 = OpenCL::getElapsedTime(event_kernel);
	Core::TimeSpan time2 = OpenCL::getElapsedTime(event_read1);
	Core::TimeSpan time3 = OpenCL::getElapsedTime(event_read2);
	Core::TimeSpan time4 = OpenCL::getElapsedTime(event_read3);
	Core::TimeSpan time5 = OpenCL::getElapsedTime(event_write1);
	Core::TimeSpan time6 = OpenCL::getElapsedTime(event_write2);
	Core::TimeSpan time7 = OpenCL::getElapsedTime(event_write3);
	Core::TimeSpan timeGPU = time1+time2+time3+time4+time5+time6+time7;

    std::cout <<"Kernel Time "<< time1 << std::endl;
    std::cout <<"Read1 Time "<< time2 << std::endl;
    std::cout <<"Read2 Time "<< time3 << std::endl;
    std::cout <<"Read3 Time "<< time4 << std::endl;
    std::cout <<"Write1 Time "<< time5 << std::endl;
    std::cout <<"Write2 Time "<< time6 << std::endl;
    std::cout <<"Write3 Time "<< time7 << std::endl;
    std::cout <<"Total Time "<< timeGPU << std::endl;

    //--------------------------------
    /*int k = 0;
    for (int i = 0; i < count; i++)
    {
            std::cout << "(" << h_GpuR[i] << "," << h_GpuC[i] <<")" << "-->" << h_GpuV[i];
            k++;
            if (k==Dest.cols)
            {
                std::cout << "\n";
                k=0;
            }
    } */

    // Populate the Sparse MAtrix

    typedef Eigen::Triplet<float> T;
    std::vector<T> tripletList;
    tripletList.reserve(count);
    int k;

    // Timing Analysis
    Core::TimeSpan time10 = Core::getCurrentTime();
    for(k=0; k< count; k++ )
    {
    // Put the values on the stack        
        tripletList.push_back(T(h_GpuR[k],h_GpuC[k],h_GpuV[k])); 
    }
    Core::TimeSpan time11 = Core::getCurrentTime();

	Core::TimeSpan timefor = time11 - time10;
    std::cout << "Time taken to execute for loop - " << timefor << std::endl; 

    // Timing Analysis
    Core::TimeSpan time8 = Core::getCurrentTime();
    // Populate the Sparse MAtrix
    _Mmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    Core::TimeSpan time9 = Core::getCurrentTime();

	Core::TimeSpan timeTriplets = time9 - time8;
    std::cout << "Time taken to execute timeTriplets - " << timeTriplets << std::endl;    


/*
    for (k=0; k<_Mmatrix.outerSize(); ++k)
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
	std::cout << "Time taken to execute DMatrix - " << timed << std::endl;

    //HMatrix = Hmatrix(Dest, kernel);

	// Do calculation on the host side
	Core::TimeSpan time3 = Core::getCurrentTime();
    //MMatrix = Mmatrix(Dest, delta.x, delta.y);
    Core::TimeSpan time4 = Core::getCurrentTime();

	Core::TimeSpan timem = time4 - time3;
	std::cout << "Time taken to execute MMatrix - " << timem << std::endl;

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

	//	Create a command queue
	queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    //*******************************************************************
    // TODO viennacl::init();
    viennacl::ocl::setup_context(0, context(), device(), queue());
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
        Src[i] = cv::imread("../Images/Cameraman/LR"+ boost::lexical_cast<std::string> (i+1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);
     
        
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

