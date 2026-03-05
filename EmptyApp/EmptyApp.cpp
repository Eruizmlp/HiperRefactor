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
#include "GhostManager.h" 

int main(int argc, char** argv) 
{
    using std::vector;
    using namespace hiperlife; 

    Init(argc, argv);
    int myRank = MyRank();
    int nProcs = NumProcs();

    if (myRank == 0) {
        std::cout << "=== STARTING HIPERLIFE DEMO ===" << std::endl;
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

    // --- Explicit Map & Vector Initialization ---
    if (myRank == 0) std::cout << "--- Case 1: Explicit Map & Vector ---" << std::endl;

    BlockMap myBlockMap("MyMap", MpiComm());
    myBlockMap.setLocNItem(distributedData.size());
    myBlockMap.Update(); 

    GenericFieldStruct<int> fieldExplicit("FieldExplicit", MpiComm());
    fieldExplicit.setBlockMap(myBlockMap);
    fieldExplicit.setDistFields(distributedData);
    fieldExplicit.setNFlds(1);
    fieldExplicit.Update();

    int val0 = fieldExplicit.getValue(0, 0, IndexType::Local);
    std::cout << "Rank " << myRank << " [Explicit]: Local[0] = " << val0 << std::endl;
    MPI_Barrier(MpiComm());

    // --- AutoMap from Vector ---
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

    // --- Global Scatter ---
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

    // --- Raw Pointers & Multi-Field ---
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

    // --- Modifiers ---
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
    
    // --- GhostManager Topology Evaluation ---
    if (myRank == 0) std::cout << "\n--- Case 6: GhostManager Topology Test ---" << std::endl;

    std::vector<int> candidates;
    if (myRank == 0) {
        if (nProcs > 1) candidates = {4, 1, 5, 4}; 
    } else if (myRank == 1) {
        candidates = {3, 0};
    } else if (myRank == 2) {
        candidates = {5};
    }

    GhostManager ghostMgr("TestGhostMgr", MpiComm());
    ghostMgr.setBlockMap(myBlockMap);
    ghostMgr.setGhostLists(candidates);
    ghostMgr.Update();

    const auto& cleanGhosts = ghostMgr.getGhosts();
    std::cout << "Rank " << myRank << " [GhostMgr]: Clean Ghosts List = { ";
    for (int g : cleanGhosts) std::cout << g << " ";
    std::cout << "}" << std::endl;

    const auto& sendMap = ghostMgr.getSendMap();
    for (const auto& pair : sendMap) {
        std::cout << "Rank " << myRank << " [GhostMgr]: Will send my LOCAL nodes { ";
        for (int lid : pair.second) std::cout << lid << " ";
        std::cout << "} to Rank " << pair.first << std::endl;
    }
    MPI_Barrier(MpiComm());

    // --- Ghost Data Communication ---
    if (myRank == 0) std::cout << "\n--- Case 7: Topology + Communication Test ---" << std::endl;

    std::vector<double> values(MyRank()*2);
    for (int i = 0; i < values.size(); ++i) {
        //int globalID = myBlockMap.myOffset() + i;
        values[i] = i%10 + 100*MyRank();
    }


    GenericFieldStruct<double> commField("CommField", MpiComm());
    commField.setDistFields(values);
    commField.setLocNItem(myBlockMap.loc_nItem());
    commField.setNFlds(1);
    commField.Update();

    commField.UpdateGhosts(candidates); 

    if (nProcs > 1) {
        if (myRank == 0) {
            double ghostVal4 = commField.getValue(0, 4, IndexType::Global);
            std::cout << "Rank 0 [Initial Sync]: Read Global 4 = " << ghostVal4 << " (Expected 104)" << std::endl;
        } else if (myRank == 1) {
            double ghostVal0 = commField.getValue(0, 0, IndexType::Global);
            std::cout << "Rank 1 [Initial Sync]: Read Global 0 = " << ghostVal0 << " (Expected 100)" << std::endl;
        }
    }
    MPI_Barrier(MpiComm());

    if (myRank == 0) {
        std::cout << "\nVisualizing Distributed Field State (After Topology + Comm):" << std::endl;
    }
    
    std::cout << commField; 
    MPI_Barrier(MpiComm());

    Finalize();
    return 0;
}