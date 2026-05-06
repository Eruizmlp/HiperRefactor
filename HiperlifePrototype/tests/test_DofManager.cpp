#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>
#include "hl_Core.h"
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
#include "hl_DOFsHandler.h"
#include "DofManager.h"

using namespace hiperlife;

// Helper to strictly compare field values
void assertEquivalence(double valNew, double valOld, const std::string& context) {
    if (std::abs(valNew - valOld) > 1e-14) {
        std::cerr << "Rank " << MyRank() << " FAILED: " << context 
                  << " | New: " << valNew << ", Old: " << valOld << std::endl;
        hiperlife::Abort("Mathematical equivalence assertion failed.");
    }
}

int main(int argc, char** argv) {
    Init(argc, argv);
    int myRank = MyRank();

    if (VerboseRank()) {
        std::cout << "=== DOFsHandler vs DofManager Correctness Suite ===\n" << std::endl;
        std::cout << "Validating state equivalence (ICs, BCs, Ghosts, Interpolation)...\n" << std::endl;
    }

    SmartPtr<StructMeshGenerator> structMesh = Create<StructMeshGenerator>();
    structMesh->setMesh(ElemType::Triang, BasisFuncType::Lagrangian, 1);
    structMesh->genStructMesh(50); 

    SmartPtr<DistributedMesh> disMesh = Create<DistributedMesh>();
    disMesh->setMesh(structMesh);
    disMesh->setElementLocatorEngine(ElementLocatorEngine::BoundingVolumeHierarchy);
    disMesh->setBalanceMesh(true);
    disMesh->Update();

    int numDOFs = 2;
    auto icFunc = [](double x, double y) { return std::sin(x * 3.0) + std::cos(y * 7.0); };

    SmartPtr<DofManager> newDofs = Create<DofManager>(disMesh);
    newDofs->setNumDOFs(numDOFs);
    newDofs->Update();

    SmartPtr<DOFsHandler> oldDofs = Create<DOFsHandler>(disMesh);
    oldDofs->setNumDOFs(numDOFs);
    oldDofs->Update();

    newDofs->setInitialCondition(0, icFunc);
    newDofs->setInitialCondition(1, 2.5);
    
    oldDofs->setInitialCondition(0, icFunc);
    oldDofs->setInitialCondition(1, 2.5);

    newDofs->setBoundaryCondition(0, MAxis::Xmin, 0.0);
    newDofs->setBoundaryCondition(1, MAxis::Ymax, 1.0);

    oldDofs->setBoundaryCondition(0, MAxis::Xmin, 0.0);
    oldDofs->setBoundaryCondition(1, MAxis::Ymax, 1.0);

    newDofs->UpdateGhosts(true);
    oldDofs->UpdateGhosts(true);

    for (int i = 0; i < disMesh->loc_nPts(); ++i) {
        for (int d = 0; d < numDOFs; ++d) {
            double vNew = newDofs->nodeDOFs->getValue(d, i, IndexType::Local);
            double vOld = oldDofs->nodeDOFs->getValue(d, i, IndexType::Local);
            assertEquivalence(vNew, vOld, "Local DOF mismatch at index " + std::to_string(i));
        }
    }

    const std::vector<int>& listGhosts = disMesh->listGhosts();
    for (int ghost : listGhosts) {
        for (int d = 0; d < numDOFs; ++d) {
            double vNew = newDofs->nodeDOFs->getValue(d, ghost, IndexType::Global);
            double vOld = oldDofs->nodeDOFs->getValue(d, ghost, IndexType::Global);
            assertEquivalence(vNew, vOld, "Ghost DOF mismatch at global index " + std::to_string(ghost));
        }
    }

    for (int i = 0; i < disMesh->loc_nPts(); ++i) {
        for (int d = 0; d < numDOFs; ++d) {
           
            bool flagNew = newDofs->constrFlag(d, i, IndexType::Local);
            bool flagOld = oldDofs->constrFlag(d, i, IndexType::Local);
            if (flagNew != flagOld) hiperlife::Abort("Constraint flag mismatch.");

          
            if (flagNew) {
                double valNew = newDofs->constrValue(d, i, IndexType::Local);
                double valOld = oldDofs->constrValue(d, i, IndexType::Local);
                assertEquivalence(valNew, valOld, "Constraint value mismatch at index " + std::to_string(i));
            }
        }
    }

    int nLocElem = disMesh->loc_nElem();
    std::vector<int> elems(nLocElem);
    std::iota(elems.begin(), elems.end(), 0);

    std::vector<std::vector<double>> rCoords(nLocElem, {0.333333333, 0.333333333}); 
    std::vector<int> fields = {0, 1};

    std::vector<std::vector<double>> interpNew = newDofs->interpolateDOFs(fields, elems, rCoords, IndexType::Local);
    std::vector<std::vector<double>> interpOld = oldDofs->interpolateDOFs(fields, elems, rCoords, IndexType::Local);

    if (interpNew.size() != interpOld.size()) hiperlife::Abort("Interpolation array size mismatch.");
    
    for (size_t i = 0; i < interpNew.size(); ++i) {
        for (size_t d = 0; d < interpNew[i].size(); ++d) {
            assertEquivalence(interpNew[i][d], interpOld[i][d], "Interpolation mismatch at element " + std::to_string(i));
        }
    }

    GlobalBarrier();

    if (VerboseRank()) {
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "SUCCESS: DofManager passed all state equivalence checks." << std::endl;
        std::cout << "The refactor maps 1:1 with legacy DOFsHandler physics." << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    Finalize();
    return 0;
}