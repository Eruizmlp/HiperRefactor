#ifndef BLOCK_MAP_H
#define BLOCK_MAP_H

#include <vector>
#include <string>
#include <mpi.h>
#include "hl_Core.h" 

namespace hiperlife {

class BlockMap : public hiperlife::DistributedClass {
protected:
    int _nItem{0};       // Total global items
    int _loc_nItem{0};   // Local items
    std::vector<int> _offs_nItem; // Offsets [Rank0, Rank1, ..., Total]

public:
    BlockMap() : DistributedClass("DefaultMap", MPI_COMM_WORLD) {}
    
    BlockMap(std::string tag, MPI_Comm comm) : DistributedClass(tag, comm) {}
    
    int nItem() const { return _nItem; }
    int loc_nItem() const { return _loc_nItem; }
    
    int myOffset() const { 
        if (_offs_nItem.empty()) return 0;
        return _offs_nItem[_myRank]; 
    } 

    const std::vector<int>& getOffsets() const { return _offs_nItem; }

    std::vector<int> getCounts() const {
        if (_offs_nItem.empty()) return {};
        std::vector<int> counts(_numProcs);
        for(int i=0; i<_numProcs; ++i) {
            counts[i] = _offs_nItem[i+1] - _offs_nItem[i];
        }
        return counts;
    }

    int getLocalIndex(int globalIdx) const;
    int getGlobalIndex(int localIdx) const;

    int getItemPartition(int globIdx) const; 

    void setLocNItem(int n) { _loc_nItem = n; }

    void Update(); 

private:
    void computeOffsets();
    int bisection(int globIdx) const;
};

} 
#endif