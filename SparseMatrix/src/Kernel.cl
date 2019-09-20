#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

/*
 * SuperAwesome_D_Matrix :
 * Description - Compute the Values, Columns and Rows vectors of the D-Matrix as required for the Sparse CRS or CLS representation.
*/
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

/*
 * SuperAwesome_D_Row_Pointer :
 * Description - Compute the Row pointer vector of the Sparse D-Matrix for the Sparse CRS representation.
*/
__kernel void SuperAwesome_D_Row_Pointer (uint rfactor, __global int* NZ_Row_Pointer) {

    size_t i = get_global_id(0);                                                                                    // The row order

    NZ_Row_Pointer[i] = i * rfactor * rfactor;
}

/*
 * SuperAwesome_H_Matrix :
 * Description - Compute the Values, Columns and Rows vectors of the H-Matrix as required for the Sparse CRS or CLS representation.
*/
__kernel void SuperAwesome_H_Matrix (__global float* gauss_kernel, __global float* NZ_values, __global int* NZ_Rows, __global int* NZ_Columns, uint kernel_rows, uint kernel_cols) {

    size_t i = get_global_id(0);                                    // the i'th row
	size_t j = get_global_id(1);                                    // the j'th column
    size_t Dest_rows = get_global_size(0);                          // total number of rows in Dest
    size_t Dest_cols = get_global_size(1);                          // total number of columns in Dest

    // Total buffer size for checking bounds # no
    uint dim_dstvec = Dest_rows * Dest_cols;

	int k = 0;
    // Corresponding row of H-Matrix 
    uint index = i*Dest_cols + j;

	int radius_y = ((kernel_rows-1)/2);
    int radius_x = ((kernel_cols-1)/2);
	uint kernelSize = kernel_rows*kernel_cols;

    // Row indexing Offset to handle matrix to buffer representation
    int row_offset =  index*kernelSize;

    for (int m = 0; m < kernel_rows; m++)
    {
        for (int n = 0; n < kernel_cols; n++)
        {
            int loc = (i-radius_y+m)*Dest_cols + (j-radius_x+n);
            if ( ((int)i)-radius_y+m >= 0 && ((int)i)-radius_y+m < Dest_rows && ((int)j)-radius_x+n >= 0 && ((int)j)-radius_x+n < Dest_cols && loc < dim_dstvec)
            {
			    //_Hmatrix.coeffRef(index,loc) = kernel.at<float>(m,n);
				NZ_values[row_offset + k] = gauss_kernel[m*kernel_rows +n];
				NZ_Rows[row_offset + k] = index;
				NZ_Columns[row_offset + k] = loc;
				k++;
			}
        }
    }
}


/*
 * SuperAwesome_H_Row_Pointer :
 * Description - Compute the Row pointer vector of the Sparse H-Matrix for the Sparse CRS representation.
*/
__kernel void SuperAwesome_H_Row_Pointer (float psfW, uint Dest_rows, uint Dest_cols,  __global int* NZ_Row_Pointer) {

    size_t i    = get_global_id(0);                                           // Get the index of Row Pointer
    size_t xact = (i / Dest_cols);                             // X value of Dest matrix
	size_t yact = (i % Dest_cols);                             // Y value of Dest matrix

    int x           = 0;
    int y           = 0;
    int psfWidth    = (int) psfW;

    // Taking care for the first element of the Row Pointer vector which is zero.
    if ( (xact == 0) && (yact == 0) )
    {
        NZ_Row_Pointer[i] = 0;  
        return;
    }  

    // For handling every (*, 0) element of the matrix
    if ( (xact > 0) && (yact == 0) )
    {
        x = xact - 1;
        y = Dest_cols;    
    }
    // For moving one to one element left for the rest of the elements    
    else
    {
        x = xact;
        y = yact - 1;
    }

    // Handle corner elements
    if(((x - (psfWidth / 2)) < 0) && ((y - (psfWidth / 2)) < 0))
        NZ_Row_Pointer[i] = ( (psfWidth + (x - (psfWidth / 2))) * (psfWidth + (y - (psfWidth / 2))) );

    // Handle edge elements
    else if(((x - (psfWidth / 2)) < 0) && ((y + (psfWidth / 2)) > Dest_rows))
        NZ_Row_Pointer[i] = ( (psfWidth + (x - (psfWidth / 2))) * (psfWidth - (y + (psfWidth / 2) - Dest_rows)) );

    // Handle edge elements
    else if(((x + (psfWidth / 2)) > 0) && ((y - (psfWidth / 2)) < 0))
        NZ_Row_Pointer[i] = ( (psfWidth - (x + (psfWidth / 2) - Dest_cols)) * (psfWidth + (y - (psfWidth / 2))) );

    // Handle corner elements
    else if(((x + (psfWidth / 2)) > 0) && ((y + (psfWidth / 2)) > 0))
        NZ_Row_Pointer[i] = ( (psfWidth - (x + (psfWidth / 2) - Dest_cols)) * (psfWidth - (y + (psfWidth / 2) - Dest_rows)) );

    // Rest of the elements
    else
        NZ_Row_Pointer[i] = (psfWidth * psfWidth);

}


/*
 * SuperAwesome_M_Matrix :
 * Description - Compute the Values, Columns and Rows vectors of the M-Matrix as required for the Sparse CRS or CLS representation.
*/
__kernel void SuperAwesome_M_Matrix (float deltaX, float deltaY, __global int* NZ_Rows, __global float* NZ_values, __global int* NZ_Columns) {
       
    size_t i = get_global_id(0);                                // the Ith loop we removed
	size_t j = get_global_id(1);                                // the Jth loop we removed
    size_t countX = get_global_size(0);                         // Dest.rows
    size_t countY = get_global_size(1);                         // Dest.cols

    uint dim_dstvec = countX * countY;                          // CPU - int dim_dstvec = Dest.rows * Dest.cols;

    if(i < (countX-floor(deltaY)) && j< (countY-floor(deltaX)) && (i+floor(deltaY) >= 0) && (j+floor(deltaX) >= 0))
	{
		int index = i*countY + j;
		int neighborUL = (i+(deltaY))*countY + (j+(deltaX));
		int neighborUR = (i+(deltaY))*countY + (j+(deltaX)+1);
		int neighborBR = (i+(deltaY)+1)*countY + (j+(deltaX)+1);
		int neighborBL = (i+(deltaY)+1)*countY + (j+(deltaX));

		if(neighborUL >= 0 && neighborUL < dim_dstvec)
        {
			//CPU - _Mmatrix.coeffRef(index, neighborUL) = (i+std::floor(deltaY)+1-(i+deltaY))*(j+std::floor(deltaX)+1-(j+deltaX));
            NZ_values   [i*countY*4 + j*4]    = (i+(deltaY)+1-(i+deltaY))*(j+(deltaX)+1-(j+deltaX)); 
            NZ_Rows     [i*countY*4 + j*4]    = index;
            NZ_Columns  [i*countY*4 + j*4]    = neighborUL;
        }   
		if(neighborUR >= 0 && neighborUR < dim_dstvec)
        {
			//CPU - _Mmatrix.coeffRef(index, neighborUR) = (i+std::floor(deltaY)+1-(i+deltaY))*(j+deltaX-(j+std::floor(deltaX)));
            NZ_values   [i*countY*4 + j*4+1]  = (i+(deltaY)+1-(i+deltaY))*(j+deltaX-(j+(deltaX))); 
            NZ_Rows     [i*countY*4 + j*4+1]  = index;
            NZ_Columns  [i*countY*4 + j*4+1]  = neighborUR;
        }
		if(neighborBR >= 0 && neighborBR < dim_dstvec)
        {
			//CPU - _Mmatrix.coeffRef(index, neighborBR) = (i+deltaY-(i+std::floor(deltaY)))*(j+deltaX-(j+std::floor(deltaX)));
            NZ_values   [i*countY*4 + j*4+2]  = (i+deltaY-(i+(deltaY)))*(j+deltaX-(j+(deltaX)));
            NZ_Rows     [i*countY*4 + j*4+2]  = index;
            NZ_Columns  [i*countY*4 + j*4+2]  = neighborBR;
        }
		if(neighborBL >= 0 && neighborBL < dim_dstvec)
        {
			//CPU - _Mmatrix.coeffRef(index, neighborBL) = (i+deltaY-(i+std::floor(deltaY)))*(j+std::floor(deltaX)+1-(j+deltaX));
            NZ_values   [i*countY*4 + j*4+3]  = (i+deltaY-(i+(deltaY)))*(j+(deltaX)+1-(j+deltaX));
            NZ_Rows     [i*countY*4 + j*4+3]  = index;
            NZ_Columns  [i*countY*4 + j*4+3]  = neighborBL;
        }
	}
}

/*
 * SuperAwesome_M_Row_Pointer :
 * Description - Compute the Row pointer vector of the Sparse M-Matrix for the Sparse CRS representation.
*/
__kernel void SuperAwesome_M_Row_Pointer (__global int* NZ_Row_Pointer) {

    size_t i = get_global_id(0);                                                                                    // The row order

    NZ_Row_Pointer[i] = 4;
}
