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
    /*!
     * \enum DataLayout
     * \brief Defines the memory layout for the field structures.
     */
    enum class DataLayout {
        AoS, //!< Array of Structures: [item0_f0, item0_f1, item1_f0, item1_f1]
        SoA  //!< Structure of Arrays: [item0_f0, item1_f0, item0_f1, item1_f1]
    };

    /*!
     * \class GenericFieldStruct
     * \brief Class that provides a dynamic dual-layout distributed data structure for mesh representation.
     *
     * It natively stores data as an Array of Structures (AoS) for optimal CPU cache locality 
     * during local finite element physics assembly, but it can ingest and export Structure of Arrays (SoA)
     * layouts for MPI communication and legacy inter-operation.
     */
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

    /** @name Constructor
     * @{ */
    GenericFieldStruct(std::string tag, MPI_Comm comm);
    /** @} */

    /** @name Object Distribution (Getters)
     * @{ */
    /*! \brief Returns the total number of items globally. */
    int nItem() const;
    /*! \brief Returns the number of items stored in the current partition. */
    int loc_nItem() const;
    /*! \brief Returns the global offset of the first item in the current partition. */
    int myOffset() const;
    /*! \brief Returns the MPI rank of the current partition. */
    int myRank() const;          
    /*! \brief Returns the total number of MPI processes. */
    int numProcs() const;        
    /*! \brief Returns the number of fields (variables) per item. */
    int numFlds() const;
    /*! \brief Returns a reference to the GhostManager associated with this structure. */
    GhostManager& ghostManager();
    /** @} */


    /** @name Initialization and Object Creation (Setters)
     * @{ */
    /*! \brief Sets the BlockMap that defines the parallel distribution. */
    void setBlockMap(const BlockMap map);
    /*! \brief Sets the number of fields per item. */
    void setNFlds(int nflds);
    /*! \brief Sets the number of items stored locally. */
    void setLocNItem(int n);
    /*! \brief Fills all items and fields with a uniform value. */
    void setValue(T value);
    /*! \brief Fills a specific field across all local items with a uniform value. */
    void setValue(int fieldIdx, T value);
    /*! \brief Assigns a string name (tag) to a specific field index. */
    void setFieldName(int idx, std::string name);
    /*! \brief Retrieves the index corresponding to a given field tag. */
    int getFieldIndex(const std::string& name) const;
    
    /*! \brief Sets the distributed fields using a raw pointer, defining its memory layout. */
    void setDistFields(T* fields, int locNItem, DataLayout layout = DataLayout::AoS);
    /*! \brief Sets the distributed fields using a std::vector, defining its memory layout. */
    void setDistFields(const std::vector<T>& fields, DataLayout layout = DataLayout::AoS);
    /*! \brief Sets the global fields (only meaningful in rank 0) to be scattered during Update. */
    void setGlobFields(T* fields, DataLayout layout = DataLayout::AoS);

    /*! \brief Method that checks parameters, allocates memory, and transposes data if needed. */
    void Update();

    /*! \brief Method that updates the topology of the ghost items based on candidate lists. */
    void UpdateGhosts(const std::vector<int>& candidates);
    /*! \brief Method that triggers the MPI communication to update ghost values. */
    void UpdateGhosts();
    /** @} */

    /** @name Query Methods
     * @{ */
    /*! \brief Returns a copy of the distributed data in the requested layout. */
    std::vector<T> getDistData(DataLayout layout = DataLayout::AoS) const;
    /*! \brief Returns true if the given global index belongs to the current partition. */
    bool isItemInPart(int globIdx) const;
    /*! \brief Returns true if the given global index is stored as a ghost in the current partition. */
    bool isItemInPartGhosts(int globIdx) const;
    /*! \brief Returns the string tag of a specific field index. */
    std::string tag(int i) const;
    /** @} */

    /** @name Direct Access Setters & Getters
     * @{ */
    /*! \brief Sets the value of a specific field using a local item index.
     \param fieldIdx the index of the field.
     \param localIdx a local index.
     \param value the value to set. */
    void setLocalValue(int fieldIdx, int localIdx, T value);

    /*! \brief Sets the value of a specific field using a global item index.
     \param fieldIdx the index of the field.
     \param globalIdx a global index.
     \param value the value to set. */
    void setGlobalValue(int fieldIdx, int globalIdx, T value);

    /*! \brief Sets the value of a specific field evaluating the IndexType at runtime.
     \param fieldIdx the index of the field.
     \param idx the index of the item.
     \param type the type of the index (Local or Global).
     \param value the value to set. */
    void setValue(int fieldIdx, int idx, IndexType type, T value);

    /*! \brief Returns the value of a specific field using a local item index.
     \param fieldIdx the index of the field.
     \param localIdx a local index. */
    T getLocalValue(int fieldIdx, int localIdx) const;

    /*! \brief Returns the value of a specific field using a global item index.
     \param fieldIdx the index of the field.
     \param globalIdx a global index. */
    T getGlobalValue(int fieldIdx, int globalIdx) const;

    /*! \brief Returns the value of a specific field evaluating the IndexType at runtime.
     \param fieldIdx the index of the field.
     \param idx the index of the item.
     \param type the type of the index (Local or Global). */
    T getValue(int fieldIdx, int idx, IndexType type = IndexType::Global) const;
    /** @} */

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
    /*! \brief Scatters the global fields from Rank 0 to all partitions, adapting to the selected layout. */
    void _performScatter();
};

} 

#include "GenericFieldStruct-impl.h"

#endif