#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif



__kernel void SuperAwesome_D_Matrix (uint rfactor,__global float* NZ_values,__global int* NZ_Columns, __global int* NZ_Rows) {

    size_t i = get_global_id(0);                                                                                    // the Ith loop we removed
	size_t j = get_global_id(1);                                                                                    // the Jth loop we removed
    size_t countX = get_global_size(0);                                                                             // The size of Ith loop

    uint k = 0;

	uint LRindex = i*countX + j;                                                                                    // The row index
	for (int m = rfactor*i; m < (i+1)*rfactor; m++)
	{ 
		for (int n = rfactor*j; n < (j+1)*rfactor; n++)
		{
            // i*countX*rfactor*rfactor are the TOTAL number of memory locations to loop over
            // k is the small innermost loop that puts 4 values in each row
            // j*4 is the Jth loop that was there in the CPU implementation

			uint HRindex = m*countX*rfactor + n;                                                                    // Previously m*dst_cols+ n; // The column index
            NZ_values [j*rfactor*rfactor + i*countX*rfactor*rfactor + k] = 1.0/rfactor/rfactor; 
            NZ_Columns[j*rfactor*rfactor + i*countX*rfactor*rfactor + k] = HRindex;
            NZ_Rows   [j*rfactor*rfactor + i*countX*rfactor*rfactor + k] = LRindex;
            k++;
		}
	}
}

__kernel void SuperAwesome_D_Row_Pointer (uint rfactor, __global int* NZ_Row_Pointer) {

    size_t i = get_global_id(0);                                                                                    // The row order

    NZ_Row_Pointer[i] = i * rfactor * rfactor;
}


__kernel void SuperAwesome_H_Matrix (__global float* gauss_kernel, __global float* NZ_values, __global int* NZ_Rows, __global int* NZ_Columns) {


    size_t i = get_global_id(0);                                    // the i'th row
	size_t j = get_global_id(1);                                    // the j'th column
    size_t Dest_rows = get_global_size(0);                          // total number of rows in Dest
    size_t Dest_cols = get_global_size(1);                          // total number of columns in Dest

    // Total buffer size for checking bounds # no
    int dim_dstvec = Dest_rows * Dest_cols;

	int k = 0;
    // Corresponding row of H-Matrix 
    int index = i*Dest_cols + j;

    // Row indexing Offset to handle matrix to buffer representation
    int row_offset =  index*9;

    // Set the corresponding row elements
	unsigned int UL = (i-1)*Dest_cols + (j-1);

	if (i-1 >= 0 && j-1 >= 0 && UL < dim_dstvec)
	{        
		NZ_values[row_offset + k] = gauss_kernel[0];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = UL;
		k++; 			
	}

	unsigned int UM = (i-1)*Dest_cols + j;
	if (i-1 >= 0 && UM < dim_dstvec)
	{	
		NZ_values[row_offset + k] = gauss_kernel[1];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = UM;
		k++;
	}

	unsigned int UR = (i-1)*Dest_cols + (j+1);
	if (i-1 >= 0 && j+1 < Dest_cols && UR < dim_dstvec)
	{	
		NZ_values[row_offset + k] = gauss_kernel[2];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = UR;
		k++;
	}

	unsigned int ML = i*Dest_cols + (j-1);
	if (j-1 >= 0 && ML < dim_dstvec)
	{
		NZ_values[row_offset + k] = gauss_kernel[3];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = ML;
		k++;
	}

	unsigned int MR = i*Dest_cols + (j+1);
	if (j+1 < Dest_cols && MR < dim_dstvec)
	{	
		NZ_values[row_offset + k] = gauss_kernel[4];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = MR;
		k++;
	}

	unsigned int BL = (i+1)*Dest_cols + (j-1);
	if (j-1 >= 0 && i+1 < Dest_rows && BL < dim_dstvec)
	{
		NZ_values[row_offset + k] = gauss_kernel[5];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = BL;
		k++;
	}

	unsigned int BM = (i+1)*Dest_cols + j;
	if (i+1 < Dest_rows && BM < dim_dstvec)
	{
		NZ_values[row_offset + k] = gauss_kernel[6];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = BM;
		k++;
	}

	unsigned int BR = (i+1)*Dest_cols + (j+1);
	if (i+1 < Dest_rows && j+1 < Dest_cols && BR < dim_dstvec)
	{	
		NZ_values[row_offset + k] = gauss_kernel[7];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = BR;
		k++;
	}

		NZ_values[row_offset + k] = gauss_kernel[8];
        NZ_Rows[row_offset + k] = index;
		NZ_Columns[row_offset + k] = index;

}



__kernel void SuperAwesome_M_Matrix (int dst_rows, int dst_cols,float deltaX,float deltaY,__global int* NZ_Rows,__global float* NZ_values,__global int* NZ_Columns,int dim_dstvec) {
       
    size_t i = get_global_id(0); // the Ith loop we removed
	size_t j = get_global_id(1);  // the Jth loop we removed
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);

    //int dim_dstvec = dst_rows*dst_cols;
    //int k = 0;
    if(i < (dst_rows-floor(deltaY)) && j< (dst_cols-floor(deltaX)) && (i+floor(deltaY) >= 0) && (j+floor(deltaX) >= 0))
			{
				int index = i*dst_cols + j;
				int neighborUL = (i+(deltaY))*dst_cols + (j+(deltaX));
				int neighborUR = (i+(deltaY))*dst_cols + (j+(deltaX)+1);
				int neighborBR = (i+(deltaY)+1)*dst_cols + (j+(deltaX)+1);
				int neighborBL = (i+(deltaY)+1)*dst_cols + (j+(deltaX));

				if(neighborUL >= 0 && neighborUL < dim_dstvec)
                {
					//_Mmatrix.coeffRef(index, neighborUL) = (i+std::floor(deltaY)+1-(i+deltaY))*(j+std::floor(deltaX)+1-(j+deltaX));
                    NZ_values[i*countY*4 + j*4] = (i+(deltaY)+1-(i+deltaY))*(j+(deltaX)+1-(j+deltaX)); 
                    NZ_Rows[i*countY*4 + j*4] = index;
                    NZ_Columns[i*countY*4 + j*4] = neighborUL;
                    //k++;
                }   
				if(neighborUR >= 0 && neighborUR < dim_dstvec)
                {
					//_Mmatrix.coeffRef(index, neighborUR) = (i+std::floor(deltaY)+1-(i+deltaY))*(j+deltaX-(j+std::floor(deltaX)));
                    NZ_values[i*countY*4 + j*4+1] = (i+(deltaY)+1-(i+deltaY))*(j+deltaX-(j+(deltaX))); 
                    NZ_Rows[i*countY*4 + j*4+1] = index;
                    NZ_Columns[i*countY*4 + j*4+1] = neighborUR;
                    //k++;
                }
				if(neighborBR >= 0 && neighborBR < dim_dstvec)
                {
					//_Mmatrix.coeffRef(index, neighborBR) = (i+deltaY-(i+std::floor(deltaY)))*(j+deltaX-(j+std::floor(deltaX)));
                    NZ_values[i*countY*4 + j*4+2] = (i+deltaY-(i+(deltaY)))*(j+deltaX-(j+(deltaX)));
                    NZ_Rows[i*countY*4 + j*4+2] = index;
                    NZ_Columns[i*countY*4 + j*4+2] = neighborBR;
                    //k++;
                }
				if(neighborBL >= 0 && neighborBL < dim_dstvec)
                {
					//_Mmatrix.coeffRef(index, neighborBL) = (i+deltaY-(i+std::floor(deltaY)))*(j+std::floor(deltaX)+1-(j+deltaX));
                    NZ_values[i*countY*4 + j*4+3] = (i+deltaY-(i+(deltaY)))*(j+(deltaX)+1-(j+deltaX));
                    NZ_Rows[i*countY*4 + j*4+3] = index;
                    NZ_Columns[i*countY*4 + j*4+3] = neighborBL;
                    //k++;
                }
			}
}
