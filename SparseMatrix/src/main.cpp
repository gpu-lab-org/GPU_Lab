
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
Eigen::SparseMatrix<float, Eigen::RowMajor,int> Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor)
{
    Eigen::SparseMatrix<float,Eigen::RowMajor, int> _Dmatrix(Src.rows*Src.cols, Dest.rows*Dest.cols);
    
    unsigned int nnz = Src.rows * Src.cols * 4;

    // Change viennacl context
    viennacl::ocl::setup_context(0, context(), device(), queue());
    viennacl::ocl::switch_context(0);

    std::cout << "Existing context: " << context() << std::endl;
    std::cout << "ViennaCL uses context: " << viennacl::ocl::current_context().handle().get() << std::endl;

/*
    clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with executing clsparseSetup()" << std::endl;
    }

    // Create clSPARSE control object, is required to queue for kernel execution
    clsparseCreateResult createResult = clsparseCreateControl( queue( ) );
    CLSPARSE_V( createResult.status, "Failed to create clsparse control" );

    // Define a SPARSE matrix
    clsparseCsrMatrix _Dmatrix_ClSPARSE;
    clsparseInitCsrMatrix(&_Dmatrix_ClSPARSE);

    // Declaration of CSR matrix parameters
    clsparseIdx_t nnz = Src.rows * Src.cols * 4;
    clsparseIdx_t row = 16384;
    clsparseIdx_t col = 65536;

    // Initialised _Dmatrix parameters
    _Dmatrix_ClSPARSE.num_nonzeros = nnz; 
    _Dmatrix_ClSPARSE.num_rows = row; 
    _Dmatrix_ClSPARSE.num_cols = col;
*/

    // Allocate memory for _Dmatrix CSR matrix
    /*    
    _Dmatrix_ClSPARSE.values = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
                                 _Dmatrix_ClSPARSE.num_nonzeros * sizeof( float ), NULL, NULL );

    _Dmatrix_ClSPARSE.col_indices = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
                                     _Dmatrix_ClSPARSE.num_nonzeros * sizeof( clsparseIdx_t ), NULL, NULL );

    _Dmatrix_ClSPARSE.row_pointer = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
                                     ( _Dmatrix_ClSPARSE.num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, NULL );


    // Create meta data for the ClSPARSE
    clsparseCsrMetaCreate( &_Dmatrix_ClSPARSE, createResult.control );
*/
    std::size_t wgSize = 16;

    //TODO clean 
    std::size_t count  = Src.rows * Src.cols * 4;
    
    std::vector<cl_float>   h_GpuV (count);
    std::vector<cl_uint>    h_GpuC (count);
    std::vector<cl_uint>    h_GpuR (count);
   
    // Create buffer 
    // Using opencl C cl_mem data structures

    cl_mem NZ_values    = ::clCreateBuffer( context(), CL_MEM_READ_WRITE, count * sizeof(cl_float) , NULL, NULL);
    cl_mem NZ_Columns   = ::clCreateBuffer( context(), CL_MEM_READ_WRITE, count * sizeof(cl_uint) , NULL, NULL);
    cl_mem NZ_Rows      = ::clCreateBuffer( context(), CL_MEM_READ_WRITE, count * sizeof(cl_uint) , NULL, NULL);   

    /*
    // Using C++ Wrapper Implementated data structures
    cl::Buffer NZ_values(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Columns(context,CL_MEM_READ_WRITE,size);
    cl::Buffer NZ_Rows(context,CL_MEM_READ_WRITE,size);
    */
    
    //initialization
    memset(h_GpuV.data(), 0, count * sizeof(cl_float));
    memset(h_GpuC.data(), 0, count * sizeof(cl_uint));
    memset(h_GpuR.data(), 0, count * sizeof(cl_uint));

    // Using opencl C based function calls

    ::clEnqueueWriteBuffer(queue(), NZ_values,  true, 0, count * sizeof(cl_float),  h_GpuV.data(), 0, NULL, NULL);
    ::clEnqueueWriteBuffer(queue(), NZ_Columns, true, 0, count * sizeof(cl_uint),   h_GpuC.data(), 0, NULL, NULL);
    ::clEnqueueWriteBuffer(queue(), NZ_Rows,    true, 0, count * sizeof(cl_uint),   h_GpuR.data(), 0, NULL, NULL);

    /*
    // Using C++ Wrapper based function calls
    queue.enqueueWriteBuffer(NZ_values, true, 0, size, h_GpuV.data());
    queue.enqueueWriteBuffer(NZ_Columns, true, 0, size, h_GpuC.data());
	queue.enqueueWriteBuffer(NZ_Rows, true, 0, size, h_GpuR.data());
    */

	//queue.enqueueWriteBuffer(_Dmatrix.values, true, 0, size, h_GpuV.data());
    //queue.enqueueWriteBuffer(_Dmatrix.col_indices, true, 0, size, h_GpuC.data());
    //queue.enqueueWriteBuffer(_Dmatrix.row_pointer, true, 0, size, h_GpuR.data());

    // New kernel object for computation of each matrix
	cl::Kernel SuperAwesome_D_kernel(program, "SuperAwesome_D_Matrix");
    //cl_kernel SuperAwesome_D_kernel = clCreateKernel(program, "SuperAwesome_D_Matrix", NULL);

    //kernel arguments
    /*
    ::clSetKernelArg(SuperAwesome_D_kernel, 0, sizeof(cl_int), (void*)&Src.cols);
    ::clSetKernelArg(SuperAwesome_D_kernel, 1, sizeof(cl_int), (void*)&Dest.cols);
    ::clSetKernelArg(SuperAwesome_D_kernel, 2, sizeof(cl_float), (void*)&rfactor);
    ::clSetKernelArg(SuperAwesome_D_kernel, 3, sizeof(cl_mem), (void*)&NZ_values);
    ::clSetKernelArg(SuperAwesome_D_kernel, 4, sizeof(cl_mem), (void*)&NZ_Columns);
    ::clSetKernelArg(SuperAwesome_D_kernel, 5, sizeof(cl_mem), (void*)&NZ_Rows);
    */

    SuperAwesome_D_kernel.setArg<cl_int>    (0,Src.cols);
	SuperAwesome_D_kernel.setArg<cl_int>    (1,Dest.cols);
    SuperAwesome_D_kernel.setArg<cl_float>  (2,rfactor); 
    SuperAwesome_D_kernel.setArg<cl_mem>    (3,NZ_values);   
    SuperAwesome_D_kernel.setArg<cl_mem>    (4,NZ_Columns);
    SuperAwesome_D_kernel.setArg<cl_mem>    (5,NZ_Rows); 

    //launch the kernel
    queue.enqueueNDRangeKernel(SuperAwesome_D_kernel, 0,cl::NDRange(Src.rows, Src.cols),cl::NDRange(wgSize, wgSize), NULL, NULL);
    
    // Read from Device to Host
    ::clEnqueueReadBuffer(queue(), NZ_values,   true, 0, count * sizeof(cl_float),  h_GpuV.data(), 0, NULL, NULL);
    ::clEnqueueReadBuffer(queue(), NZ_Columns,  true, 0, count * sizeof(cl_uint),   h_GpuC.data(), 0, NULL, NULL);
    ::clEnqueueReadBuffer(queue(), NZ_Rows,     true, 0, count * sizeof(cl_uint),   h_GpuR.data(), 0, NULL, NULL);

    /*
    queue.enqueueReadBuffer(NZ_values, true, 0, size, h_GpuV.data(), NULL, NULL);
    queue.enqueueReadBuffer(NZ_Columns, true, 0, size, h_GpuC.data(), NULL, NULL);
    queue.enqueueReadBuffer(NZ_Rows, true, 0, size, h_GpuR.data(),NULL,NULL);
    */

    // Initialise _Dmatrix_ClSPARSE
    /*
    _Dmatrix_ClSPARSE.values = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
                                     ( _Dmatrix_ClSPARSE.num_nonzeros ) * sizeof( clsparseIdx_t ), NULL, NULL );

    _Dmatrix_ClSPARSE.col_indices = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
                                     ( _Dmatrix_ClSPARSE.num_nonzeros ) * sizeof( clsparseIdx_t ), NULL, NULL );

    _Dmatrix_ClSPARSE.row_pointer = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
                                     ( _Dmatrix_ClSPARSE.num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, NULL );
    
    cl_int copyres = clEnqueueCopyBuffer(queue(), NZ_values, _Dmatrix_ClSPARSE.values, 0, 0, 
                                    _Dmatrix_ClSPARSE.num_nonzeros * sizeof(float), 0, NULL, NULL);
    
    if(copyres == CL_SUCCESS)
        std::cout << "copyres value " << copyres << "\n";

    //Test Vector
    std::vector<float> test_values (_Dmatrix_ClSPARSE.num_nonzeros);
    memset(test_values.data(), 0, _Dmatrix_ClSPARSE.num_nonzeros);
    ::clEnqueueReadBuffer(queue(), _Dmatrix_ClSPARSE.values,  true, 0, size, test_values.data(), 0, NULL, NULL);
*/
/*
    clEnqueueFillBuffer(queue(), _Dmatrix_ClSPARSE.values, &zero, sizeof(float),
                                    0, _Dmatrix.num_nonzeros * sizeof(float), 0, NULL, NULL);
    clEnqueueFillBuffer(queue(), _Dmatrix_ClSPARSE.col_indices, &zero, sizeof(float),
                                    0, _Dmatrix.num_nonzeros * sizeof(float), 0, NULL, NULL);
    clEnqueueFillBuffer(queue(), _Dmatrix_ClSPARSE.row_pointer, &zero, sizeof(float),
                                    0,  (_Dmatrix.num_rows + 1 ) * sizeof(float), 0, NULL, NULL);
*/

    /* ViennaCl*/

    // BUG in library !!!!!!!!!!! Initialise cl_mem for viennacl compressed matrix
    //cl_mem mem_elements =   ::clCreateBuffer(context(), CL_MEM_READ_WRITE, count * sizeof(cl_float),    h_GpuV.data(), NULL);
    //cl_mem mem_col_buffer = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, count * sizeof(cl_uint),     h_GpuC.data(), NULL);
    //cl_mem mem_row_buffer = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, count * sizeof(cl_uint),     h_GpuR.data(), NULL);

    //viennacl::compressed_matrix<float> _vDmatrix(mem_row_buffer, mem_col_buffer, mem_elements, (size_t)(Src.rows*Src.cols), (size_t)(Dest.rows*Dest.cols), (size_t)nnz);

    viennacl::compressed_matrix<float> _vDmatrix ((size_t)(Src.rows*Src.cols), (size_t)(Dest.rows*Dest.cols));

    //_vDmatrix.handle();

    //std::cout << "test is" << test <<  "\n";

    cl_int copyres = ::clEnqueueWriteBuffer(queue(), _vDmatrix.handle().opencl_handle(),  true, 0, count * sizeof(cl_float),  h_GpuV.data(), 0, NULL, NULL);

    std::cout << "copyres value " << copyres << "\n";

    if(copyres == CL_SUCCESS)
        std::cout << "copyres value " << copyres << "\n";


    cl_int copyres2 = clEnqueueCopyBuffer(queue(), NZ_values, _vDmatrix.handle().opencl_handle(), 0, 0, 
                                    count * sizeof(float), 0, NULL, NULL);
    
    std::cout << "copyres2 value " << copyres2 << "\n";

    if(copyres2 == CL_SUCCESS)
        std::cout << "copyres2 value " << copyres2 << "\n";


    //Test Vector
    std::vector<float> test_values (count * sizeof(cl_float));
    memset(test_values.data(), 0, count * sizeof(cl_float));
    ::clEnqueueReadBuffer(queue(), _vDmatrix.handle().opencl_handle(),  true, 0, count * sizeof(cl_float), test_values.data(), 0, NULL, NULL);

    //_vDmatrix.handle1();
    //_vDmatrix.handle2();


    int k = 0;
    for (int i = 0; i < count; i++)
    {
        //std::cout << "(" << h_GpuR[i] << ", " << h_GpuC[i] <<")" << "-->" << h_GpuV[i] << "-->" << test_values[i]  << "\n";
        //std::cout << "(" << h_GpuR[i] << ", " << h_GpuC[i] <<")" << "-->" << h_GpuV[i] << "\t";

        k++;
        if (k==Dest.cols)
        {
            //std::cout << "\n";
            k=0;
        }
    }

	return _Dmatrix;
}


//=======================================================================
//
// H __ MATRIX
//
//==========================433=============================================
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



    for (k=0; k<_Mmatrix.outerSize(); ++k)
    {    
        for (Eigen::SparseMatrix<float,Eigen::RowMajor, int>::InnerIterator it(_Mmatrix,k); it; ++it)
        {
            std::cout <<   "  (" << it.row() << "," << it.col() << ")--" << it.value();

        }
        std::cout << "\n";
    }


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

