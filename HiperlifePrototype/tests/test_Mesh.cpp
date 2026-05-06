#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iomanip> 
#include <algorithm> 
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
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

    SmartPtr<StructMeshGenerator> structMesh = Create<StructMeshGenerator>();
    structMesh->setMesh(ElemType::Triang , BasisFuncType::Lagrangian, 1);
    structMesh->genStructMesh(25);

    SmartPtr<DistributedMesh> disMesh = Create<DistributedMesh>();
    disMesh->setMesh(structMesh);
    disMesh->setElementLocatorEngine(ElementLocatorEngine::BoundingVolumeHierarchy);
    disMesh->setBalanceMesh(true);
    disMesh->Update();

    std::cout << *disMesh->_nodeData << std::endl;
    disMesh->printFileLegacyVtk("Mesh");

    int loc_nNodes = disMesh->loc_nPts();
    int numDim = disMesh->nDim();

    vector<double> hl_local_coords(loc_nNodes * numDim);
    for (int i = 0; i < loc_nNodes; ++i) {
        hl_local_coords[i * numDim + 0] = disMesh->nodeCoord(i, 0, IndexType::Local);
        hl_local_coords[i * numDim + 1] = disMesh->nodeCoord(i, 1, IndexType::Local);
        hl_local_coords[i * numDim + 2] = disMesh->nodeCoord(i, 2, IndexType::Local);
    }

    vector<int> hl_ghost_nodes = disMesh->listGhosts();

    BlockMap nodeMap("MeshMap", MpiComm());
    nodeMap.setLocNItem(loc_nNodes);
    nodeMap.Update();

    GenericFieldStruct<double> coordField("Coordinates", MpiComm());
    coordField.setBlockMap(nodeMap);
    coordField.setNFlds(numDim);
    coordField.setDistFields(hl_local_coords);
    coordField.Update();

    coordField.UpdateGhosts(hl_ghost_nodes);

    MPI_Barrier(MpiComm());

    const auto& fieldGhosts = coordField.ghostManager().getGhosts();
    if (fieldGhosts.size() != hl_ghost_nodes.size()) {
        hiperlife::Abort("Ghost count mismatch between GenericFieldStruct and DistributedMesh.");
    }

    if (myRank == 0) {
        std::cout << "Test passed successfully with local nodes processed: " << loc_nNodes << std::endl;
    }

    Finalize();
    return 0;
}