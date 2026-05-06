#include "DofManager.h"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <cstddef>
#include <numeric>

namespace hiperlife {

DofManager::DofManager(SmartPtr<DistributedMesh> inputMesh, std::string tag)
    : DistributedClass(tag, inputMesh->comm()), mesh(inputMesh) {}

void DofManager::setNumDOFs(int dofs) {
    nodeDOFs = Create<GenericFieldStruct<double>>("nodeDOFs", _comm);
    nodeDOFs0 = Create<GenericFieldStruct<double>>("nodeDOFs0", _comm);
    nodeDOFs->setNFlds(dofs);
    nodeDOFs0->setNFlds(dofs);
}

void DofManager::setDOFs(const std::vector<std::string>& tags) {
    nodeDOFs = Create<GenericFieldStruct<double>>("nodeDOFs", _comm);
    nodeDOFs0 = Create<GenericFieldStruct<double>>("nodeDOFs0", _comm);
    nodeDOFs->setNFlds(tags.size());
    nodeDOFs0->setNFlds(tags.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        nodeDOFs->setFieldName(i, tags[i]);
        nodeDOFs0->setFieldName(i, tags[i]);
    }
}

int DofManager::numDOFs() const {
    return (nodeDOFs == Teuchos::null) ? 0 : nodeDOFs->numFlds();
}

void DofManager::setNumNodeAuxF(int auxflds) {
    if (auxflds <= 0) return;
    nodeAuxF = Create<GenericFieldStruct<double>>("nodeAuxF", _comm);
    nodeAuxF->setNFlds(auxflds);
}

void DofManager::setNodeAuxF(const std::vector<std::string>& tags) {
    if (tags.empty()) return;
    nodeAuxF = Create<GenericFieldStruct<double>>("nodeAuxF", _comm);
    nodeAuxF->setNFlds(tags.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        nodeAuxF->setFieldName(i, tags[i]);
    }
}

int DofManager::numNodeAuxF() const {
    return (nodeAuxF == Teuchos::null) ? 0 : nodeAuxF->numFlds();
}

void DofManager::setNumElemAuxF(int auxflds) {
    if (auxflds <= 0) return;
    elemAuxF = Create<GenericFieldStruct<double>>("elemAuxF", _comm);
    elemAuxF->setNFlds(auxflds);
}

void DofManager::setElemAuxF(const std::vector<std::string>& tags) {
    if (tags.empty()) return;
    elemAuxF = Create<GenericFieldStruct<double>>("elemAuxF", _comm);
    elemAuxF->setNFlds(tags.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        elemAuxF->setFieldName(i, tags[i]);
    }
}

int DofManager::numElemAuxF() const {
    return (elemAuxF == Teuchos::null) ? 0 : elemAuxF->numFlds();
}

void DofManager::Update() {
    if (_nameTag == "DistributedClass") _nameTag = "DofManager";
    if (mesh != Teuchos::null) {
        _update_new();
    } else {
        hiperlife::Abort("DofManager::Update: Mesh is not set. Legacy load methods have been deprecated.");
    }
    UpdateGhosts(true);
}

void DofManager::_update_new() {
    mesh->setNameTag("DisMesh");
    
    BlockMap nodeMap("nodeMap", _comm);
    nodeMap.setLocNItem(mesh->loc_nPts());
    nodeMap.Update();

    nodeDOFs->setBlockMap(nodeMap);
    nodeDOFs->Update();
    nodeDOFs0->setBlockMap(nodeMap);
    nodeDOFs0->Update();

    flagConstr = Create<GenericFieldStruct<double>>("flagConstr", _comm);
    flagConstr->setBlockMap(nodeMap);
    flagConstr->setNFlds(numDOFs());
    flagConstr->Update();
    flagConstr->setValue(-1.0); 

    valuConstr = Create<GenericFieldStruct<double>>("valuConstr", _comm);
    valuConstr->setBlockMap(nodeMap);
    valuConstr->setNFlds(numDOFs());
    valuConstr->Update();
    valuConstr->setValue(0.0);

    if (nodeAuxF != Teuchos::null && nodeAuxF->numFlds() > 0) {
        nodeAuxF->setBlockMap(nodeMap);
        nodeAuxF->Update();
    }
    if (elemAuxF != Teuchos::null && elemAuxF->numFlds() > 0) {
        BlockMap elemMap("elemMap", _comm);
        elemMap.setLocNItem(mesh->loc_nElem());
        elemMap.Update();
        elemAuxF->setBlockMap(elemMap);
        elemAuxF->Update();
    }
}

void DofManager::UpdateConstrainGhosts(bool genImporters) {
    if (genImporters) {
        const std::vector<int>& listGhosts = mesh->listGhosts();
        flagConstr->UpdateGhosts(listGhosts);
        valuConstr->UpdateGhosts(listGhosts);
    } else {
        flagConstr->UpdateGhosts();
        valuConstr->UpdateGhosts();
    }
}

void DofManager::UpdateDOFsGhosts(bool genImporters) {
    if (genImporters) {
        const std::vector<int>& listGhosts = mesh->listGhosts();
        nodeDOFs->UpdateGhosts(listGhosts);
    } else {
        nodeDOFs->UpdateGhosts();
    }
}

void DofManager::UpdateDOFs0Ghosts(bool genImporters) {
    if (genImporters) {
        const std::vector<int>& listGhosts = mesh->listGhosts();
        nodeDOFs0->UpdateGhosts(listGhosts);
    } else {
        nodeDOFs0->UpdateGhosts();
    }
}

void DofManager::UpdateAuxFGhosts(bool genImporters) {
    if (nodeAuxF != Teuchos::null) {
        if (genImporters) {
            const std::vector<int>& listGhosts = mesh->listGhosts();
            nodeAuxF->UpdateGhosts(listGhosts);
        } else {
            nodeAuxF->UpdateGhosts();
        }
    }
    if (elemAuxF != Teuchos::null) {
        if (genImporters) {
            elemAuxF->UpdateGhosts(std::vector<int>{});
        } else {
            elemAuxF->UpdateGhosts();
        }
    }
}

void DofManager::UpdateGhosts(bool genImporters) {
    UpdateConstrainGhosts(genImporters);
    UpdateDOFsGhosts(genImporters);
    UpdateDOFs0Ghosts(genImporters);
    UpdateAuxFGhosts(genImporters);
}

// --- Constraints & Boundary Conditions ---
void DofManager::setConstraint(int dof, int idx, IndexType idxtype, double value) {
    flagConstr->setValue(dof, idx, idxtype, 1.0);
    valuConstr->setValue(dof, idx, idxtype, value);
}

void DofManager::setConstraint(const std::string& tag, int idx, IndexType idxtype, double value) {
    setConstraint(nodeDOFs->getFieldIndex(tag), idx, idxtype, value);
}

void DofManager::setConstraint(int dof, double value) {
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        setConstraint(dof, i, IndexType::Local, value);
    }
}

void DofManager::setBoundaryCondition(int dof, int idx, IndexType idxtype, double value) {
    if (idxtype == IndexType::Local) {
        if (mesh->nodeCrease(idx, idxtype) > 0)
            setConstraint(dof, idx, idxtype, value);
    } else {
        if (mesh->isNodeInPart(idx)) {
            int loc_idx = mesh->_graph->locIdx(idx);
            if (mesh->nodeCrease(loc_idx, idxtype) > 0)
                setConstraint(dof, loc_idx, idxtype, value);
        }
    }
}

void DofManager::setBoundaryCondition(const std::string& tag, int idx, IndexType idxtype, double value) {
    setBoundaryCondition(nodeDOFs->getFieldIndex(tag), idx, idxtype, value);
}

void DofManager::setBoundaryCondition(int dof, MAxis ax, double value) {
    int myMAxis = getMAxis(ax);
    int myPDim = getAxis(ax) + 1;
    
    if (myPDim > mesh->_pDim)
        hiperlife::Abort("setBoundaryCondition: MAxis is greater than parametric dimensions (pDim).");
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        if (mesh->nodeMAxis(i, IndexType::Local) % myMAxis == 0)
            setConstraint(dof, i, IndexType::Local, value);
    }
}

void DofManager::setBoundaryCondition(const std::string& tag, MAxis ax, double value) {
    setBoundaryCondition(nodeDOFs->getFieldIndex(tag), ax, value);
}

void DofManager::setBoundaryCondition(int dof, MAxis ax, std::function<double(double)> f) {
    int myMAxis = getMAxis(ax);
    int myAxis = getAxis(ax);
    if (myAxis + 1 > mesh->_pDim) hiperlife::Abort("setBoundaryCondition: MAxis is greater than pDim.");
    
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        if (mesh->nodeMAxis(i, IndexType::Local) % myMAxis == 0) {
            double chi = (myAxis == 0) ? mesh->nodeCoord(i, 1, IndexType::Local) : mesh->nodeCoord(i, 0, IndexType::Local);
            setConstraint(dof, i, IndexType::Local, f(chi));
        }
    }
}

void DofManager::setBoundaryCondition(int dof, MAxis ax, std::function<double(double, double)> f) {
    int myMAxis = getMAxis(ax);
    int myAxis = getAxis(ax);
    if (myAxis + 1 > mesh->_pDim) hiperlife::Abort("setBoundaryCondition: MAxis is greater than pDim.");
    
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        if (mesh->nodeMAxis(i, IndexType::Local) % myMAxis == 0) {
            double chi{}, eta{};
            if (myAxis == 0) { chi = mesh->nodeCoord(i, 1, IndexType::Local); eta = mesh->nodeCoord(i, 2, IndexType::Local); }
            else if (myAxis == 1) { chi = mesh->nodeCoord(i, 0, IndexType::Local); eta = mesh->nodeCoord(i, 2, IndexType::Local); }
            else if (myAxis == 2) { chi = mesh->nodeCoord(i, 0, IndexType::Local); eta = mesh->nodeCoord(i, 1, IndexType::Local); }
            setConstraint(dof, i, IndexType::Local, f(chi, eta));
        }
    }
}

void DofManager::setBoundaryCondition(int dof, double value) {
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        if (mesh->nodeCrease(i, IndexType::Local) > 0)
            setConstraint(dof, i, IndexType::Local, value);
    }
}

void DofManager::setBoundaryCondition(double value) {
    for (int d = 0; d < numDOFs(); d++) {
        setBoundaryCondition(d, value);
    }
}

void DofManager::removeConstraint(int dof, int idx, IndexType idxtype) {
    flagConstr->setValue(dof, idx, idxtype, -1.0);
}

void DofManager::removeConstraint(int dof) {
    flagConstr->setValue(dof, -1.0);
}

void DofManager::removeBoundaryCondition(int dof, MAxis ax) {
    int myMAxis = getMAxis(ax);
    if (getAxis(ax) + 1 > mesh->_pDim) hiperlife::Abort("removeBoundaryCondition: MAxis is greater than pDim.");
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        if (mesh->nodeMAxis(i, IndexType::Local) % myMAxis == 0)
            removeConstraint(dof, i, IndexType::Local);
    }
}

bool DofManager::constrFlag(int dof, int idx, IndexType idxtype) const {
    return flagConstr->getValue(dof, idx, idxtype) > 0.0;
}

bool DofManager::constrFlag(const std::string& tag, int idx, IndexType idxtype) const {
    return constrFlag(nodeDOFs->getFieldIndex(tag), idx, idxtype);
}

double DofManager::constrValue(int dof, int idx, IndexType idxtype) const {
    return valuConstr->getValue(dof, idx, idxtype);
}

double DofManager::constrValue(const std::string& tag, int idx, IndexType idxtype) const {
    return constrValue(nodeDOFs->getFieldIndex(tag), idx, idxtype);
}

// --- Initial Conditions ---
void DofManager::setInitialCondition(int dof, int idx, IndexType idxtype, double value) {
    nodeDOFs->setValue(dof, idx, idxtype, value);
}

void DofManager::setInitialCondition(const std::string& tag, int idx, IndexType idxtype, double value) {
    setInitialCondition(nodeDOFs->getFieldIndex(tag), idx, idxtype, value);
}

void DofManager::setInitialCondition(int dof, double value) {
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        setInitialCondition(dof, i, IndexType::Local, value);
    }
}

void DofManager::setInitialCondition(const std::string& tag, double value) {
    setInitialCondition(nodeDOFs->getFieldIndex(tag), value);
}

void DofManager::setInitialCondition(int dof, std::function<double(double)> f) {
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        double x = mesh->nodeCoord(i, 0, IndexType::Local);
        setInitialCondition(dof, i, IndexType::Local, f(x));
    }
}

void DofManager::setInitialCondition(int dof, std::function<double(double, double)> f) {
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        double x = mesh->nodeCoord(i, 0, IndexType::Local);
        double y = mesh->nodeCoord(i, 1, IndexType::Local);
        setInitialCondition(dof, i, IndexType::Local, f(x, y));
    }
}

void DofManager::setInitialCondition(int dof, std::function<double(double, double, double)> f) {
    for (int i = 0; i < mesh->loc_nPts(); i++) {
        double x = mesh->nodeCoord(i, 0, IndexType::Local);
        double y = mesh->nodeCoord(i, 1, IndexType::Local);
        double z = mesh->nodeCoord(i, 2, IndexType::Local);
        setInitialCondition(dof, i, IndexType::Local, f(x, y, z));
    }
}

// --- Field Interpolations ---
std::vector<std::vector<double>> DofManager::interpolateField(
    const std::vector<SmartPtr<GenericFieldStruct<double>>>& fieldStrs, 
    std::vector<std::vector<int>> fields, 
    const std::vector<int>& elems, 
    const std::vector<std::vector<double>>& rCoords, 
    IndexType type) 
{
    if (fieldStrs.size() != fields.size()) hiperlife::Abort("interpolateField: Size mismatch.");

    for (size_t s = 0; s < fieldStrs.size(); s++) {
        if (fields[s].empty()) {
            fields[s].resize(fieldStrs[s]->numFlds());
            std::iota(fields[s].begin(), fields[s].end(), 0);
        }
    }

    int tot_nFields = 0;
    for (const auto& f : fields) tot_nFields += f.size();

    std::vector<std::vector<double>> values(elems.size(), std::vector<double>(tot_nFields, 0.0));
    if (mesh->loc_nElem() == 0) return values;

    for (size_t n = 0; n < elems.size(); n++) {
        int elem = elems[n];
        int tag = mesh->_basisFunctions->getTag(elem, type);
        
        std::vector<double> mutableRCoords = rCoords[n];
        std::vector<std::vector<double>> BF = mesh->_basisFunctions->computeBasisFunctionsReference(mutableRCoords, tag, 0);

        int eNN = mesh->_adjcyEN->getNumNbors(elem, type);
        
        std::vector<int> nbors = mesh->_adjcyEN->getNborList(elem, type);

        int k = 0;
        for (size_t s = 0; s < fieldStrs.size(); s++) {
            for (size_t f = 0; f < fields[s].size(); f++) {
                for (int i = 0; i < eNN; i++) {
                    values[n][k] += fieldStrs[s]->getValue(fields[s][f], nbors[i], IndexType::Global) * BF[0][i];
                }
                k++;
            }
        }
    }
    
    return values;
}

std::vector<double> DofManager::interpolateField(
    const std::vector<SmartPtr<GenericFieldStruct<double>>>& fieldStrs, 
    const std::vector<std::vector<int>>& fields, 
    int elem, 
    const std::vector<double>& rCoords, 
    IndexType type) 
{
    std::vector<int> elems = {elem};
    std::vector<std::vector<double>> rCoords_v = {rCoords};
    return interpolateField(fieldStrs, fields, elems, rCoords_v, type)[0];
}

std::vector<std::vector<double>> DofManager::interpolateDOFs(std::vector<int> fields, const std::vector<int>& elems, const std::vector<std::vector<double>>& rCoords, IndexType type) {
    return interpolateField({nodeDOFs}, {fields}, elems, rCoords, type);
}

std::vector<double> DofManager::interpolateDOFs(std::vector<int> fields, int elem, const std::vector<double>& rCoords, IndexType type) {
    return interpolateField({nodeDOFs}, {fields}, elem, rCoords, type);
}

} // namespace hiperlife