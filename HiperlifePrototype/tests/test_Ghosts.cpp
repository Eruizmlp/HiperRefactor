#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>
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
        std::cout << "=== TEST 2: GHOSTS Y COMUNICACIONES ===\n" << std::endl;
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

    // --- Case 6: GhostManager Topology Evaluation ---
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

    // --- Case 7: Topology + Communication Test ---
    if (myRank == 0) std::cout << "\n--- Case 7: Topology + Communication Test ---" << std::endl;

    std::vector<double> values(myBlockMap.loc_nItem());
    for (int i = 0; i < (int)values.size(); ++i) {
        values[i] = i%10 + 100*myRank;
    }

    GenericFieldStruct<double> commField("CommField", MpiComm());
    commField.setBlockMap(myBlockMap); //TODO 
    commField.setDistFields(values);
    commField.setNFlds(1);
    commField.Update();

    commField.UpdateGhosts(candidates); 

    if (nProcs > 1) {
        if (myRank == 0) {
            double ghostVal4 = commField.getValue(0, 4, IndexType::Global);
            std::cout << "Rank 0 [Initial Sync]: Read Global 4 = " << ghostVal4 << " (Expected 100)" << std::endl;
        } else if (myRank == 1) {
            double ghostVal0 = commField.getValue(0, 0, IndexType::Global);
            std::cout << "Rank 1 [Initial Sync]: Read Global 0 = " << ghostVal0 << " (Expected 0)" << std::endl;
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
