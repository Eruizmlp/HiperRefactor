#ifndef GHOSTIE_IMPL_H
#define GHOSTIE_IMPL_H

namespace hiperlife {

template <typename T>
GhostIE<T>::GhostIE(std::string tag, MPI_Comm comm) 
    : DistributedClass(tag, comm) 
{}

template <typename T>
void GhostIE<T>::communicate(const GhostManager& manager, 
                             const std::vector<T>& localData, 
                             std::vector<T>& ghostData, 
                             int numFlds) 
{
    const auto& sendMap = manager.getSendMap();
    const auto& recvMap = manager.getRecvMap();

    if (sendMap.empty() && recvMap.empty()) return;

    int totalSendItems = 0;
    for (const auto& [targetRank, localIDs] : sendMap) {
        totalSendItems += localIDs.size() * numFlds;
    }
    std::vector<T> sendBuffer(totalSendItems);

    int totalRecvItems = 0;
    for (const auto& [targetRank, globalIDs] : recvMap) {
        totalRecvItems += globalIDs.size() * numFlds;
    }
    std::vector<T> recvBuffer(totalRecvItems);

    std::vector<MPI_Request> requests;
    requests.reserve(sendMap.size() + recvMap.size());

    int recvOffset = 0;
    for (const auto& [targetRank, globalIDs] : recvMap) 
    {
        int count = globalIDs.size() * numFlds;
        int byteCount = count * sizeof(T); 

        if (byteCount > 0) {
            requests.push_back(MPI_REQUEST_NULL);
            
            MPI_Irecv(&recvBuffer[recvOffset], byteCount, MPI_BYTE, 
                      targetRank, 0, this->_comm, &requests.back());
            
            recvOffset += count; 
        }
    }

    int sendOffset = 0;
    for (const auto& [targetRank, localIDs] : sendMap) 
    {
        int count = localIDs.size() * numFlds;
        int byteCount = count * sizeof(T); 

        if (byteCount == 0) continue;

        int startSendOffset = sendOffset;
        for (int localID : localIDs) {
            int flatIdx = localID * numFlds;
            for (int f = 0; f < numFlds; ++f) {
                sendBuffer[sendOffset++] = localData[flatIdx + f];
            }
        }

        requests.push_back(MPI_REQUEST_NULL);
        
        MPI_Isend(&sendBuffer[startSendOffset], byteCount, MPI_BYTE, 
                  targetRank, 0, this->_comm, &requests.back());
    }

    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    const auto& ghosts = manager.getGhosts();
    recvOffset = 0; 

    for (const auto& [targetRank, globalIDs] : recvMap) 
    {
        for (int globalID : globalIDs) 
        {
            auto it = std::lower_bound(ghosts.begin(), ghosts.end(), globalID);
            
            if (it != ghosts.end() && *it == globalID) 
            {
                int ghostIdx = std::distance(ghosts.begin(), it);
                int flatGhostIdx = ghostIdx * numFlds;

                for (int f = 0; f < numFlds; ++f) {
                    ghostData[flatGhostIdx + f] = recvBuffer[recvOffset++];
                }
            } 
            else 
            {
                hiperlife::Abort("GhostIE: Received global ID not found in ghost list.");
            }
        }
    }
}

}

#endif 