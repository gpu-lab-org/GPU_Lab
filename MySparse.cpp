
#include "MySparse.hpp"


MySparse::MySparse() {}

MySparse::MySparse(short Rows,short Cols) 
{
        this->mRows = Rows;
        this->mCols = Cols;
    //    mNNZ = NNZ;
    //    double test[NNZ];
    //    mVals[NNZ] ;//= test;
    //    mVals.resize(NNZ);
    //    mColIndex.resize(NNZ);
        mNNZ = 0;
        this->PrevRowAdded = 0;
        mOuterStarts.resize(Rows + 1);
        mOuterStarts[0] = 0;
}

void MySparse::InsertNewCoeff(int row,int col, float val)
{
    // THIS FUNCTION CANNOT JUST ADD THINGS IN THE MIDDLE.
    // Always.ALWAYS. the last coefficient according to Row Major
    // MAYBE MINUTE MINUTE BUG THAT IT DOESN'T CATCH OR PRINT THE VERY LAST ROW.
    // Otherwise I have tested it well and it works fine!!
    this->mVals.push_back(val);
    this->mColIndex.push_back(col);
    if (row > this->PrevRowAdded)
    {
        mOuterStarts[row] = (this->mColIndex.size()) - 1;
        this->PrevRowAdded++;
    }
    this->ResizeByOne();
}

void MySparse::ResizeByOne()
{
    this->mNNZ++;
    this->mVals.resize(this->mNNZ);
    this->mColIndex.resize(this->mNNZ);
}