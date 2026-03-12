#include "Block_Map.h"
#include <stdexcept>
#include <algorithm> 

namespace hiperlife {

    void BlockMap::Update() {
        computeOffsets();

        int calculatedTotal = _offs_nItem.back(); 
        if (_nItem == 0) {
            _nItem = calculatedTotal;
        } 
    }

    void BlockMap::computeOffsets() {
        std::vector<int> all_counts(_numProcs);
        MPI_Allgather(&_loc_nItem, 1, MPI_INT, 
                      all_counts.data(), 1, MPI_INT, 
                      _comm);

        _offs_nItem.resize(_numProcs + 1);
        _offs_nItem[0] = 0;
        for (int i = 0; i < _numProcs; ++i) {
            _offs_nItem[i+1] = _offs_nItem[i] + all_counts[i];
        }
    }

    // --- Conversiones 

    int BlockMap::getLocalIndex(int globalIdx) const {
        if (_offs_nItem.empty()) return -1;
        int start = _offs_nItem[_myRank];
        int end   = start + _loc_nItem;
        
        // Si está en mi rango, devuelvo el local. Si no, -1.
        if (globalIdx >= start && globalIdx < end) {
            return globalIdx - start;
        }
        return -1; 
    }

    int BlockMap::getGlobalIndex(int localIdx) const {
        if (localIdx >= 0 && localIdx < _loc_nItem) {
            return localIdx + myOffset();
        }
        return -1; 
    }

    int BlockMap::getItemPartition(int globIdx) const {
        return bisection(globIdx);
    }

    int BlockMap::bisection(int globIdx) const {
        if (_offs_nItem.empty()) return -1;
        if (globIdx < 0 || globIdx >= _nItem) return -1;

        auto it = std::upper_bound(_offs_nItem.begin(), _offs_nItem.end(), globIdx);        
        int rank = std::distance(_offs_nItem.begin(), it) - 1;
        
        // Seguridad extra
        if (rank < 0) rank = 0;
        if (rank >= _numProcs) rank = _numProcs - 1;
        
        return rank;
    }
}