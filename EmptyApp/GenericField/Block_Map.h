#ifndef BLOCK_MAP_H
#define BLOCK_MAP_H

#include <vector>
#include <string>
#include <mpi.h>
#include "hl_Core.h" 

namespace hiperlife {

/*!
 * \class BlockMap
 * \brief Class that manages the global distribution of items across MPI partitions.
 *
 * It provides the fundamental logic to translate between local indices (within a partition) 
 * and global indices (across the entire distributed system), assuming a contiguous distribution of items.
 */
class BlockMap : public hiperlife::DistributedClass {
protected:
    int _nItem{0};       //!< Total global items
    int _loc_nItem{0};   //!< Items stored locally in the partition
    std::vector<int> _offs_nItem; //!< Vector storing the global offsets for each rank: [Rank0, Rank1, ..., Total]

public:
    /** @name Constructors
     * @{ */
    BlockMap() : DistributedClass("DefaultMap", MPI_COMM_WORLD) {}
    BlockMap(std::string tag, MPI_Comm comm) : DistributedClass(tag, comm) {}
    /** @} */
    
    /** @name Object Distribution
     * @{ */
    /*! \brief Returns the total number of items globally. */
    int nItem() const { return _nItem; }
    
    /*! \brief Returns the number of items stored in the current partition. */
    int loc_nItem() const { return _loc_nItem; }
    
    /*! \brief Returns the global offset of the first item in the current partition. */
    int myOffset() const { 
        if (_offs_nItem.empty()) return 0;
        return _offs_nItem[_myRank]; 
    } 

    /*! \brief Returns the vector containing the global offsets of all partitions. */
    const std::vector<int>& getOffsets() const { return _offs_nItem; }

    /*! \brief Calculates and returns the number of items held by each partition. */
    std::vector<int> getCounts() const {
        if (_offs_nItem.empty()) return {};
        std::vector<int> counts(_numProcs);
        for(int i=0; i<_numProcs; ++i) {
            counts[i] = _offs_nItem[i+1] - _offs_nItem[i];
        }
        return counts;
    }

    /*! \brief Returns the local index corresponding to a given global index. Returns -1 if not local. */
    int getLocalIndex(int globalIdx) const;
    
    /*! \brief Returns the global index corresponding to a given local index. */
    int getGlobalIndex(int localIdx) const;

    /*! \brief Returns the ID of the MPI partition that owns the given global index. */
    int getItemPartition(int globIdx) const; 
    /** @} */

    /** @name Initialization
     * @{ */
    /*! \brief Sets the number of items stored locally. */
    void setLocNItem(int n) { _loc_nItem = n; }

    /*! \brief Gathers the local item counts from all partitions to compute the global distribution. */
    void Update(); 
    /** @} */

private:
    /*! \brief Internal method to execute the MPI_Allgather for the offsets. */
    void computeOffsets();
    
    /*! \brief Internal bisection algorithm to quickly find the owner partition of a global index. */
    int bisection(int globIdx) const;
};

} 
#endif