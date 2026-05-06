#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iomanip> 
#include <algorithm> 

#include "hl_Core.h"
#include "Block_Map.h" 
#include "GenericFieldStruct.h" 

int main(int argc, char** argv) 
{
    using std::vector;
    using namespace hiperlife; 

    Init(argc, argv);
    int myRank = MyRank();
    int nProcs = NumProcs();

    if (myRank == 0) {
        std::cout << "=== TEST 1: CAMPOS Y MULTI-FIELDS ===" << std::endl;
        std::cout << "Running on " << nProcs << " processors.\n" << std::endl;
    }

    vector<int> distributedData;
    if (myRank == 0) {
        distributedData = {10, 2, 3, 4}; 
    } else if (myRank == 1) {
        distributedData = {5, 6};       
    } else {
        distributedData = {myRank};     
    }

    BlockMap myBlockMap("MyMap", MpiComm());
    myBlockMap.setLocNItem(distributedData.size());
    myBlockMap.Update(); 

    // --- Case 1: Explicit Map & Vector Initialization ---
    if (myRank == 0) std::cout << "\n--- Case 1: Explicit Map & Vector ---" << std::endl;

    GenericFieldStruct<int> fieldExplicit("FieldExplicit", MpiComm());
    fieldExplicit.setBlockMap(myBlockMap);
    fieldExplicit.setDistFields(distributedData);
    fieldExplicit.setNFlds(1);
    fieldExplicit.Update();

    int val0 = fieldExplicit.getValue(0, 0, IndexType::Local);
    std::cout << "Rank " << myRank << " [Explicit]: Local[0] = " << val0 << std::endl;
    MPI_Barrier(MpiComm());

    // --- Case 2: AutoMap from Vector ---
    if (myRank == 0) std::cout << "\n--- Case 2: AutoMap from Vector ---" << std::endl;

    GenericFieldStruct<int> fieldAuto("FieldAuto", MpiComm());
    fieldAuto.setDistFields(distributedData);
    fieldAuto.setNFlds(1);
    fieldAuto.Update(); 

    if (nProcs > 1 && myRank == 1) {
        int globIdx = 4; 
        int val = fieldAuto.getValue(0, globIdx, IndexType::Global);
        std::cout << "Rank 1 [Auto]: Global[" << globIdx << "] = " << val << " (Expected 5)" << std::endl;
    }
    MPI_Barrier(MpiComm());

    // --- Case 3: Global Scatter ---
    if (myRank == 0) std::cout << "\n--- Case 3: Global Scatter ---" << std::endl;

    std::vector<int> globalData;
    if (myRank == 0) {
        int totalItems = myBlockMap.nItem();
        for (int i = 0; i < totalItems; ++i) {
            globalData.push_back((i + 1) * 100);
        }
    }

    GenericFieldStruct<int> fieldGlobal("FieldGlobal", MpiComm());
    fieldGlobal.setBlockMap(myBlockMap);
    fieldGlobal.setGlobFields(globalData.data()); 
    fieldGlobal.setNFlds(1);
    fieldGlobal.Update(); 

    std::cout << "Rank " << myRank << " [Global]: Received ";
    for (int i = 0; i < fieldGlobal.loc_nItem(); ++i) {
        std::cout << fieldGlobal.getValue(0, i, IndexType::Local) << " ";
    }
    std::cout << std::endl;
    MPI_Barrier(MpiComm());

    // --- Case 4: Raw Pointers & Multi-Field ---
    if (myRank == 0) std::cout << "\n--- Case 4: Raw Pointer & Multi-Fields ---" << std::endl;

    int numFlds = 3;
    int locItems = myBlockMap.loc_nItem(); 
    
    vector<double> vec_data(numFlds * locItems);
    for (int i = 0; i < (int)vec_data.size(); ++i) {
        vec_data[i] = (myRank + 1) * 0.1 + i; 
    }

    GenericFieldStruct<double> fieldDouble("FieldDouble", MpiComm());
    fieldDouble.setBlockMap(myBlockMap);
    fieldDouble.setNFlds(numFlds);
    fieldDouble.setDistFields(vec_data.data(), locItems); 
    fieldDouble.Update();

    if (locItems > 1) {
        double val = fieldDouble.getValue(0, 1, IndexType::Local);
        std::cout << "Rank " << myRank << " [Double]: Field0, Node1 = " << val << std::endl;
    }
    MPI_Barrier(MpiComm());

    // --- Case 5: Modifiers ---
    if (myRank == 0) std::cout << "\n--- Case 5: Modifiers ---" << std::endl;

    if (nProcs > 1) {
        int targetGlobalIdx = 4; 
        
        if (fieldGlobal.isItemInPart(targetGlobalIdx) || fieldGlobal.isItemInPartGhosts(targetGlobalIdx)) {
            fieldGlobal.setValue(0, targetGlobalIdx, IndexType::Global, 9999);
            
            std::cout << "Rank " << myRank << " [Mod]: Value at Global " << targetGlobalIdx << " is now " 
                      << fieldGlobal.getValue(0, targetGlobalIdx, IndexType::Global) << " (Expected 9999)" << std::endl;
        }
    }
    MPI_Barrier(MpiComm());

    Finalize();
    return 0;
}

