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
    enum class DataLayout {
    AoS, // Array of Structures: [item0_f0, item0_f1, item1_f0, item1_f1]
    SoA  // Structure of Arrays: [item0_f0, item1_f0, item0_f1, item1_f1]
};

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

    DataLayout _tmp_layout{DataLayout::AoS};

public:
    GenericFieldStruct(std::string tag, MPI_Comm comm);
    
    // GETTERS 
    int nItem() const;
    int loc_nItem() const;
    int myOffset() const;

    int myRank() const;          
    int numProcs() const;        
    int numFlds() const;

    GhostManager& ghostManager();

    void UpdateGhosts(const std::vector<int>& candidates);
    void UpdateGhosts();

    // SETTERS
    void setBlockMap(const BlockMap map);
    void setNFlds(int nflds);
    void setLocNItem(int n);
    void setValue(T value);
    void setValue(int fieldIdx, T value);
    void setFieldName(int idx, std::string name);
    int getFieldIndex(const std::string& name) const;


    void setDistFields(T* fields, int locNItem, DataLayout layout = DataLayout::AoS);
    void setDistFields(const std::vector<T>& fields, DataLayout layout = DataLayout::AoS);
    void setGlobFields(T* fields, DataLayout layout = DataLayout::AoS);

    std::vector<T> getDistData(DataLayout layout = DataLayout::AoS) const;

    void Update();

    bool isItemInPart(int globIdx) const;
    bool isItemInPartGhosts(int globIdx) const;
    std::string tag(int i) const;
    
    void setValue(int fieldIdx, int idx, IndexType type, T value);
    T getValue(int fieldIdx, int idx, IndexType type = IndexType::Global) const;

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
    void _performScatter();
};

} 

#include "GenericFieldStruct-impl.h"

#endif 