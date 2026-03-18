#ifndef GENERICFIELDSTRUCT_IMPL_H
#define GENERICFIELDSTRUCT_IMPL_H

namespace hiperlife {

template <typename T>
GenericFieldStruct<T>::GenericFieldStruct(std::string tag, MPI_Comm comm) 
    : DistributedClass(tag, comm), 
      _map(tag + "_map", comm), 
      _ghostManager(tag + "_ghostManager", comm) ,
      _ghostIE(tag + "_ghostIE", comm)
{
    _fieldTags.resize(_numFlds, "Var");
}

template <typename T>
int GenericFieldStruct<T>::nItem() const { return _map.nItem(); }

template <typename T>
int GenericFieldStruct<T>::loc_nItem() const { return _map.loc_nItem(); }

template <typename T>
int GenericFieldStruct<T>::myOffset() const { return _map.myOffset(); }

template <typename T>
int GenericFieldStruct<T>::myRank() const { return this->_myRank; }

template <typename T>
int GenericFieldStruct<T>::numProcs() const { return this->_numProcs; }

template <typename T>
int GenericFieldStruct<T>::numFlds() const { return _numFlds; }

template <typename T>
GhostManager& GenericFieldStruct<T>::ghostManager() { return _ghostManager; }

template <typename T>
void GenericFieldStruct<T>::UpdateGhosts(const std::vector<int>& candidates) {
    _ghostManager.setBlockMap(_map);
    _ghostManager.setGhostLists(candidates);
    _ghostManager.Update();

    _ghostData.resize(_ghostManager.getGhosts().size() * _numFlds);
    UpdateGhosts();
}

template <typename T>
void GenericFieldStruct<T>::UpdateGhosts() {
    _ghostIE.communicate(_ghostManager, _data, _ghostData, _numFlds);
}

template <typename T>
void GenericFieldStruct<T>::setBlockMap(const BlockMap map) {
    if(!_data.empty()) {
        hiperlife::Abort(" Cannot set map after data is allocated.");
    }
    _map = map; 
    _mapIsSet = true;
    _ghostManager.setBlockMap(_map);
}

template <typename T>
void GenericFieldStruct<T>::setNFlds(int nflds) {
    if (nflds <= 0) throw std::runtime_error("Number of fields must be positive.");
    _numFlds = nflds;
    _fieldTags.resize(_numFlds, "Var"); 
}

template <typename T>
void GenericFieldStruct<T>::setLocNItem(int n) {
    _map.setLocNItem(n);
}

template <typename T>
void GenericFieldStruct<T>::setValue(T value) {
    if (!_mapIsSet) hiperlife::Abort("Map not set");
    std::fill(_data.begin(), _data.end(), value);
}

template <typename T>
void GenericFieldStruct<T>::setValue(int fieldIdx, T value) {
    if (!_mapIsSet) hiperlife::Abort("Map not set");
    if (fieldIdx < 0 || fieldIdx >= _numFlds) hiperlife::Abort("Invalid field index");
    
    int nTotal = _data.size() / _numFlds;
    for(int i = 0; i < nTotal; ++i) {
        int flatIdx = i * _numFlds + fieldIdx; 
        _data[flatIdx] = value;
    }
}

template <typename T>
void GenericFieldStruct<T>::setFieldName(int idx, std::string name) {
    if (idx < 0 || idx >= _numFlds) hiperlife::Abort("Index out of bounds for tag.");
    _tagToIndex[name] = idx;
    if ((int)_fieldTags.size() <= idx) _fieldTags.resize(idx+1);
    _fieldTags[idx] = name;
}

template <typename T>
int GenericFieldStruct<T>::getFieldIndex(const std::string& name) const {
    auto it = _tagToIndex.find(name);
    if (it == _tagToIndex.end()) hiperlife::Abort("Unknown field tag: " + name);
    return it->second;
}

template <typename T>
void GenericFieldStruct<T>::setDistFields(T* fields, int locNItem) {
    _tmp_distFields = fields;   
    _map.setLocNItem(locNItem); 
    _tmp_distVector = nullptr;
    _tmp_globFields = nullptr;
}

template <typename T>
void GenericFieldStruct<T>::setDistFields(const std::vector<T>& fields) {
    _tmp_distVector = &fields;
    _tmp_distFields = nullptr;
    _tmp_globFields = nullptr;
}

template <typename T>
void GenericFieldStruct<T>::setGlobFields(T* fields) {
    _tmp_globFields = fields;
    _initFromGlobal = true;
    _tmp_distFields = nullptr;
    _tmp_distVector = nullptr;
}

template <typename T>
void GenericFieldStruct<T>::Update() {
    if (!_mapIsSet) { 
        hiperlife::Abort("BlockMap has not been set. Use setBlockMap() before calling Update().");
    }

    const int nLoc = _map.loc_nItem(); 
    const int totalSize = _numFlds * nLoc;
    
    if ((int)_data.size() < totalSize) {
        _data.assign(totalSize, T());
    }

    if (_tmp_distFields != nullptr) {
        std::copy(_tmp_distFields, _tmp_distFields + totalSize, _data.begin());
    }
    else if (_tmp_distVector != nullptr) {
         if ((int)_tmp_distVector->size() != totalSize) {
             hiperlife::Abort("Dimension mismatch. Input vector size is " + 
                              std::to_string(_tmp_distVector->size()) + 
                              ", but expected locNItem (" + std::to_string(nLoc) + 
                              ") * numFlds (" + std::to_string(_numFlds) + ") = " + std::to_string(totalSize) + ".");
         }
         _data = *_tmp_distVector; 
    }
    else if (_initFromGlobal) {
        _performScatter();
    }
}

template <typename T>
bool GenericFieldStruct<T>::isItemInPart(int globIdx) const {
    return _map.getLocalIndex(globIdx) != -1;
}

template <typename T>
bool GenericFieldStruct<T>::isItemInPartGhosts(int globIdx) const {
    return _ghostManager.isGhost(globIdx);
}

template <typename T>
std::string GenericFieldStruct<T>::tag(int i) const {
    if (i < 0 || i >= (int)_fieldTags.size()) return "Unknown";
    return _fieldTags[i];
}

template <typename T>
void GenericFieldStruct<T>::setValue(int fieldIdx, int idx, IndexType type, T value) {
    if (!_mapIsSet) hiperlife::Abort("Map not initialized.");
    if (fieldIdx < 0 || fieldIdx >= _numFlds) hiperlife::Abort("Invalid Field Index.");

    if (type == IndexType::Local) {
        int flatIdx = (idx * _numFlds) + fieldIdx;
        if (flatIdx >= (int)_data.size()) hiperlife::Abort("Local index out of bounds.");
        _data[flatIdx] = value;
        return;
    }

    if (type == IndexType::Global) {
        int localIdx = _map.getLocalIndex(idx);
        if (localIdx != -1) {
            int flatIdx = (localIdx * _numFlds) + fieldIdx;
            _data[flatIdx] = value;
            return;
        }
        
        const auto& ghosts = _ghostManager.getGhosts();
        auto it = std::lower_bound(ghosts.begin(), ghosts.end(), idx);
        
        if (it != ghosts.end() && *it == idx) {
            int ghostIdx = std::distance(ghosts.begin(), it);
            int flatIdx = (ghostIdx * _numFlds) + fieldIdx;
            
            if (flatIdx >= (int)_ghostData.size()) {
                 hiperlife::Abort("Writing to unallocated ghost. Call UpdateGhosts first.");
            }
            
            _ghostData[flatIdx] = value; 
            return;
        }
    }
    hiperlife::Abort("setValue - Global Index " + std::to_string(idx) + " not found locally or in ghosts.");
}

template <typename T>
T GenericFieldStruct<T>::getValue(int fieldIdx, int idx, IndexType type) const {
    if (!_mapIsSet) hiperlife::Abort("Map not initialized.");
    if (fieldIdx < 0 || fieldIdx >= _numFlds) hiperlife::Abort("Invalid Field Index.");
    
    if (type == IndexType::Local) {
        int flatIdx = (idx * _numFlds) + fieldIdx;
        if (flatIdx >= (int)_data.size()) hiperlife::Abort("Local index out of bounds.");
        return _data[flatIdx];
    }

    if (type == IndexType::Global) {
        int localIdx = _map.getLocalIndex(idx);
        if (localIdx != -1) {
            int flatIdx = (localIdx * _numFlds) + fieldIdx;
            return _data[flatIdx];
        }
        
        const auto& ghosts = _ghostManager.getGhosts();
        auto it = std::lower_bound(ghosts.begin(), ghosts.end(), idx);
        
        if (it != ghosts.end() && *it == idx) {
            int ghostIdx = std::distance(ghosts.begin(), it);
            int flatIdx = (ghostIdx * _numFlds) + fieldIdx;
            
            if (flatIdx >= (int)_ghostData.size()) {
                hiperlife::Abort("Ghost data not allocated. Call UpdateGhosts first.");
            }
            return _ghostData[flatIdx]; 
        }
    }
    hiperlife::Abort("getValue - Global Index " + std::to_string(idx) + " not found locally or in ghosts.");
    return T(); 
}

template <typename T>
void GenericFieldStruct<T>::_performScatter() {
    std::vector<int> counts = _map.getCounts(); 
    std::vector<int> offsets = _map.getOffsets(); 
    
    int nodeBlockSize = _numFlds * sizeof(T);
    
    for(auto& c : counts) c *= nodeBlockSize;
    for(auto& o : offsets) o *= nodeBlockSize;

    int nLocBytes = _map.loc_nItem() * nodeBlockSize; 

    const void* sendbuf = nullptr;
    if (this->_myRank == 0) {
        if (_tmp_globFields == nullptr) {
            hiperlife::Abort("Rank 0: Global data pointer is null in _performScatter.");
        }
        sendbuf = reinterpret_cast<const void*>(_tmp_globFields);
    }
    
    void* recvbuf = reinterpret_cast<void*>(_data.data());

    MPI_Scatterv(
        sendbuf,            
        counts.data(),      
        offsets.data(),     
        MPI_BYTE,           
        recvbuf,            
        nLocBytes,          
        MPI_BYTE,           
        0,                  
        this->_comm               
    );
}

} 

#endif 