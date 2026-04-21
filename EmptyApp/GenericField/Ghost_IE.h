#ifndef GHOSTIE_H
#define GHOSTIE_H

#include <vector>
#include <map>
#include <algorithm>
#include "hl_Core.h"
#include "GhostManager.h"

namespace hiperlife {

template <typename T>
class GhostIE : public hiperlife::DistributedClass {
protected:
    // Persistent memory to avoid reallocations
    std::vector<T> _exportBuffer;
    std::vector<T> _importBuffer;
    std::vector<MPI_Request> _requests;
    
    // Precomputed index mappings (O(1) lookups)
    std::vector<int> _exportIndices;   
    std::vector<int> _importIndices; 

    // --- NEW: Flat routing arrays to replace std::map iteration ---
    std::vector<int> _exportRanks;
    std::vector<int> _exportCounts;
    std::vector<int> _importRanks;
    std::vector<int> _importCounts;

public:
    GhostIE(std::string tag, MPI_Comm comm);

    void setupImportExport(const GhostManager& manager, int numFlds);

    void communicate(const GhostManager& manager, 
                     const std::vector<T>& localData, 
                     std::vector<T>& ghostData, 
                     int numFlds);
};

} 

#include "Ghost_IE-impl.h"

#endif