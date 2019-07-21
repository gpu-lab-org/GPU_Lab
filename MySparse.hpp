

#include <vector>

class MySparse {
public:

    short mRows;
    short mCols;
    short PrevRowAdded;
    int mNNZ;

    // Default Constructor
    MySparse();

    MySparse(short Rows,short Cols);

    void InsertNewCoeff(int row,int col, float val);
    void ResizeByOne();

    std::vector<float> mVals;
    std::vector<int> mOuterStarts;
    std::vector<int> mColIndex;
};


