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
public:
    GhostIE(std::string tag, MPI_Comm comm);

    void communicate(const GhostManager& manager, 
                     const std::vector<T>& localData, 
                     std::vector<T>& ghostData, 
                     int numFlds);
};

} 

#include "Ghost_IE-impl.h"

#endif 