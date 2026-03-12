#include "GhostManager.h"

namespace hiperlife {

void GhostManager::setBlockMap(const BlockMap& map)
{
    _map = map;
    _mapIsSet = true;
}

void GhostManager::setGhostLists(const std::vector<int>& candidates) 
{ 
    if(!_mapIsSet) hiperlife::Abort("BlockMap not set before setting ghosts.");
    
    int start = _map.myOffset();
    int end   = start + _map.loc_nItem(); 

    std::set<int> cleanGhosts;
    for (int globIndx : candidates) {
        if (globIndx < start || globIndx >= end) {
            cleanGhosts.insert(globIndx);
        }
    }
    
    _ghosts.assign(cleanGhosts.begin(), cleanGhosts.end());
}

void GhostManager::Update()
{   
    _recv.clear(); 
    _send.clear(); 

    //_ghosts already sorted and unique by the setter.
    for (int globIndx : _ghosts)
    {
        int ownerRank = _map.getItemPartition(globIndx); 
        if (ownerRank == -1)
            hiperlife::Abort("Ghost not found in partition: " + std::to_string(globIndx));
    
        _recv[ownerRank].push_back(globIndx);
    }

    int numProcs = _map.numProcs();
    int rank = _map.myRank();

    std::vector<int> sendCounts(numProcs, 0); // Elements this rank will request
    std::vector<int> recvCounts(numProcs, 0); // Elements others will request

    for (const auto& [targetRank, requestedIds] : _recv) {
        sendCounts[targetRank] = static_cast<int>(requestedIds.size()); 
    }

    // Inform every rank how many Global IDs they must send to us
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, 
                 recvCounts.data(), 1, MPI_INT, 
                 _comm);
    
    int totalSendSize = 0;
    for(int count : sendCounts) {
        totalSendSize += count;
    }

    std::vector<int> sendBuffer{};
    sendBuffer.reserve(totalSendSize);
    
    std::vector<int> sDisplacements(numProcs, 0);
    int currDisplacement = 0;

    for(int p = 0; p < numProcs; ++p)
    {
        sDisplacements[p] = currDisplacement;
        if(sendCounts[p] > 0)
        {
            const std::vector<int>& recvTmpList = _recv[p];
            sendBuffer.insert(sendBuffer.end(), recvTmpList.begin(), recvTmpList.end());
            currDisplacement += sendCounts[p];
        }
    }

    // Prepare buffer to receive Global IDs requested by neighbors from this rank
    int totalRecvSize{0};
    for(int count : recvCounts) {
        totalRecvSize += count;
    }

    std::vector<int> recvBuffer(totalRecvSize);
    std::vector<int> rDisplacements(numProcs, 0);
    int currRecvDisp{0};

    for(int p = 0; p < numProcs; ++p)
    {
        rDisplacements[p] = currRecvDisp;
        currRecvDisp += recvCounts[p];
    }

    // Collective exchange of indices
    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sDisplacements.data(), MPI_INT,
                  recvBuffer.data(), recvCounts.data(), rDisplacements.data(), MPI_INT,
                  _comm);

    
    for(int p = 0; p < numProcs; ++p)
    {
        int count = recvCounts[p];
        if(count == 0) continue; 

        std::vector<int>& targetList = _send[p];
        targetList.reserve(count);
        int startIdx = rDisplacements[p]; 

        for(int k = 0; k < count; ++k)
        {
            int globalID = recvBuffer[startIdx + k];
            int localID = _map.getLocalIndex(globalID);

            if (localID == -1) {
                hiperlife::Abort("Error: Rank " + std::to_string(rank) + 
                                 " was requested GlobalID " + std::to_string(globalID) + 
                                 " but it is not local.");
            }

            targetList.push_back(localID);
        }
    }
}
}