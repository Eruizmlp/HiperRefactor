#ifndef GENERICFIELDSTRUCT_H
#define GENERICFIELDSTRUCT_H

#include "Block_Map.h" 
#include <algorithm> 
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <map> 
#include "hl_TypeDefs.h"
#include "GhostManager.h" 
#include "Ghost_IE.h"

namespace hiperlife {

template <typename T>
class GenericFieldStruct : public hiperlife::DistributedClass {
protected:

    bool _mapIsSet{false}; 
    bool _initFromGlobal{false};
    
    BlockMap _map; 
    GhostManager _ghostManager; 
    GhostIE<T> _ghostIE;
    
    std::vector<T> _data; 
    std::vector<T> _ghostData;

    int _numFlds{1};

    std::map<std::string, int> _tagToIndex;
    std::vector<std::string> _fieldTags; 

    const std::vector<T>* _tmp_distVector{nullptr};
    T* _tmp_distFields{nullptr};
    T* _tmp_globFields{nullptr};

public:
    
    GenericFieldStruct(std::string tag, MPI_Comm comm) 
        : DistributedClass(tag, comm), 
          _map(tag + "_map", comm), 
          _ghostManager(tag + "_ghostManager", comm) ,
          _ghostIE(tag + "_ghostIE", comm)
    {
        _fieldTags.resize(_numFlds, "Var");
    }
    
    // GETTERS 
    int nItem() const { return _map.nItem(); }
    int loc_nItem() const { return _map.loc_nItem(); }
    int myOffset() const { return _map.myOffset(); }

    int myRank() const { return _myRank; }          
    int numProcs() const {return _numProcs;}        
    int numFlds() const { return _numFlds; }

    GhostManager& ghostManager() { return _ghostManager; }

    void UpdateGhosts(const std::vector<int>& candidates)
    {
        _ghostManager.setBlockMap(_map);
        _ghostManager.setGhostLists(candidates);
        _ghostManager.Update();

        _ghostData.resize(_ghostManager.getGhosts().size() * _numFlds);

        UpdateGhosts();
    }

    void UpdateGhosts()
    {
        _ghostIE.communicate(_ghostManager, _data, _ghostData, _numFlds);
    }

    // SETTERS
    void setBlockMap(const BlockMap map){
       if(!_data.empty()) {
            hiperlife::Abort("GenericFieldStruct: Cannot set map after data is allocated.");
        }
        _map = map; 
        _mapIsSet = true;
        
        _ghostManager.setBlockMap(_map);
    }

    void setNFlds(int nflds) {
        if (nflds <= 0) throw std::runtime_error("Number of fields must be positive.");
        _numFlds = nflds;
        _fieldTags.resize(_numFlds, "Var"); 
    }

    void setLocNItem(int n) {
        _map.setLocNItem(n);
    }
    void setValue(T value) {
        if (!_mapIsSet) hiperlife::Abort("Map not set");
        std::fill(_data.begin(), _data.end(), value);
    }

    void setValue(int fieldIdx, T value) {
        if (!_mapIsSet) hiperlife::Abort("Map not set");
        if (fieldIdx < 0 || fieldIdx >= _numFlds) hiperlife::Abort("Invalid field index");
        
        int nTotal = _data.size() / _numFlds;
        for(int i = 0; i < nTotal; ++i) {
            int flatIdx = i * _numFlds + fieldIdx; 
            _data[flatIdx] = value;
        }
    }

    void setFieldName(int idx, std::string name) {
        if (idx < 0 || idx >= _numFlds) hiperlife::Abort("GenericFieldStruct: Index out of bounds for tag.");
        _tagToIndex[name] = idx;
        if ((int)_fieldTags.size() <= idx) _fieldTags.resize(idx+1);
        _fieldTags[idx] = name;
    }

    int getFieldIndex(const std::string& name) const {
        auto it = _tagToIndex.find(name);
        if (it == _tagToIndex.end()) hiperlife::Abort("Unknown field tag: " + name);
        return it->second;
    }

    //DATA SETTERS 
    void setDistFields(T* fields, int locNItem) {
        _tmp_distFields = fields;   
        _map.setLocNItem(locNItem); 
        _tmp_distVector = nullptr;
        _tmp_globFields = nullptr;
    }

    void setDistFields(const std::vector<T>& fields) {
        _tmp_distVector = &fields;
        _tmp_distFields = nullptr;
        _tmp_globFields = nullptr;
    }

    void setGlobFields(T* fields) {
        _tmp_globFields = fields;
        _initFromGlobal = true;
        _tmp_distFields = nullptr;
        _tmp_distVector = nullptr;
    }

    void Update() {

        if (!_mapIsSet) { 
            if (_tmp_distVector != nullptr) {
                int inferredSize = _tmp_distVector->size();
                if (_numFlds > 0) inferredSize /= _numFlds;
                _map.setLocNItem(inferredSize);
            }
            _map.Update(); 
            _mapIsSet = true;
            
            _ghostManager.setBlockMap(_map);
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
                 hiperlife::Abort("GenericFieldStruct: Vector size mismatch in Update().");
             }
             _data = *_tmp_distVector; 
        }

        else if (_initFromGlobal) {
            _performScatter();
        }
    }
    
    // --- HELPERS PARA EL PRINT ---
    bool isItemInPart(int globIdx) const {
        return _map.getLocalIndex(globIdx) != -1;
    }

    bool isItemInPartGhosts(int globIdx) const {
        return _ghostManager.isGhost(globIdx);
    }
    
    std::string tag(int i) const {
        if (i < 0 || i >= (int)_fieldTags.size()) return "Unknown";
        return _fieldTags[i];
    }

    

    void setValue(int fieldIdx, int idx, IndexType type, T value) {
        if (!_mapIsSet) hiperlife::Abort("Map not initialized.");
        if (fieldIdx < 0 || fieldIdx >= _numFlds) hiperlife::Abort("Invalid Field Index.");

        // Case 1: Local Index (Directly in _data)
        if (type == IndexType::Local) {
            int flatIdx = (idx * _numFlds) + fieldIdx;
            if (flatIdx >= (int)_data.size()) hiperlife::Abort("GenericFieldStruct: Local index out of bounds.");
            _data[flatIdx] = value;
            return;
        }

        // Case 2: Global Index
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
                     hiperlife::Abort("GenericFieldStruct: Writing to unallocated ghost. Call UpdateGhosts first.");
                }
                
                _ghostData[flatIdx] = value; 
                return;
            }
        }

        // Neither local nor ghost
        hiperlife::Abort("GenericFieldStruct::setValue - Global Index " + std::to_string(idx) + " not found locally or in ghosts.");
    }

   T getValue(int fieldIdx, int idx, IndexType type = IndexType::Global) const {
        if (!_mapIsSet) hiperlife::Abort("Map not initialized.");
        if (fieldIdx < 0 || fieldIdx >= _numFlds) hiperlife::Abort("Invalid Field Index.");
        
        // Case 1: Local Index
        if (type == IndexType::Local) {
            int flatIdx = (idx * _numFlds) + fieldIdx;
            if (flatIdx >= (int)_data.size()) hiperlife::Abort("GenericFieldStruct: Local index out of bounds.");
            return _data[flatIdx];
        }

        // Case 2: Global Index
        if (type == IndexType::Global) {
            // Subcase A: Local Item
            int localIdx = _map.getLocalIndex(idx);
            if (localIdx != -1) {
                int flatIdx = (localIdx * _numFlds) + fieldIdx;
                return _data[flatIdx];
            }
            
            // Subcase B: Ghost Item
            const auto& ghosts = _ghostManager.getGhosts();
            auto it = std::lower_bound(ghosts.begin(), ghosts.end(), idx);
            
            if (it != ghosts.end() && *it == idx) {
                // It is a valid ghost node
                int ghostIdx = std::distance(ghosts.begin(), it);
                int flatIdx = (ghostIdx * _numFlds) + fieldIdx;
                
                if (flatIdx >= (int)_ghostData.size()) {
                    hiperlife::Abort("GenericFieldStruct: Ghost data not allocated. Call UpdateGhosts first.");
                }
                
                return _ghostData[flatIdx]; 
            }
        }

        hiperlife::Abort("GenericFieldStruct::getValue - Global Index " + std::to_string(idx) + " not found locally or in ghosts.");
        return T(); 
    }

    friend std::ostream& operator<<(std::ostream& os, const GenericFieldStruct<T>& field)
    {
        field.Barrier();

        if (field.myRank() == 0)
        {
            os << "=== " << field.nameTag() << " [" << field.classTag() << "] ===" << std::endl;
            os << "Tags:";
            for (int i = 0; i < field.numFlds(); i++)
                os << " " << field.tag(i);
            os << std::endl;
        }

        for (int p = 0; p < field.numProcs(); p++)
        {
            field.Barrier();

            if (p == field.myRank())
            {
                os << "Rank: " << p << std::endl;

                for(int n = 0; n < field.nItem(); n++)
                {
                    bool isLocal = field.isItemInPart(n);
                    bool isGhost = field.isItemInPartGhosts(n);

                    if (isLocal || isGhost)
                    {
                        if (isLocal) 
                            os << " L " << n  << " - ";
                        else if (isGhost) 
                            os << " * " << n  << " - ";

                        os << "{ ";
                        for (int i = 0; i < field.numFlds(); i++) {
                            os << field.getValue(i ,n, IndexType::Global) << " ";
                        }
                        os << "}" << std::endl;
                    }
                }
                os << std::endl;
            }
            os.flush();
        }
        
        field.Barrier();
        return os;
    }

protected:
    
    void _performScatter() {

        std::vector<int> counts = _map.getCounts(); 
        std::vector<int> offsets = _map.getOffsets(); 
        
        int nodeBlockSize = _numFlds * sizeof(T);
        
        for(auto& c : counts) c *= nodeBlockSize;
        for(auto& o : offsets) o *= nodeBlockSize;

        int nLocBytes = _map.loc_nItem() * nodeBlockSize; 

        
        const void* sendbuf = nullptr;
        if (_map.myRank() == 0) {
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
            _comm               
        );
    }
};

} 

#endif