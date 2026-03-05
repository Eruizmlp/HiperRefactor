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
    GhostIE(std::string tag, MPI_Comm comm) : DistributedClass(tag, comm) {}

    void communicate(const GhostManager& manager, 
                     const std::vector<T>& localData, 
                     std::vector<T>& ghostData, 
                     int numFlds) 
    {
        const auto& sendMap = manager.getSendMap();
        const auto& recvMap = manager.getRecvMap();

        if (sendMap.empty() && recvMap.empty()) return;

        // ---------------------------------------------------------------------
        // 1. ALLOCATE CONTIGUOUS BUFFERS
        // ---------------------------------------------------------------------
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

        // ---------------------------------------------------------------------
        // 2. POST RECEIVES (MPI_Irecv)
        // ---------------------------------------------------------------------
        int recvOffset = 0;
        for (const auto& [targetRank, globalIDs] : recvMap) 
        {
            int count = globalIDs.size() * numFlds;
            int byteCount = count * sizeof(T); // Convert to raw bytes

            if (byteCount > 0) {
                requests.push_back(MPI_REQUEST_NULL);
                
                // Read into the vector address, but tell MPI it's an array of bytes
                MPI_Irecv(&recvBuffer[recvOffset], byteCount, MPI_BYTE, 
                          targetRank, 0, _comm, &requests.back());
                
                recvOffset += count; // Offset advances by T elements, not bytes!
            }
        }

        // ---------------------------------------------------------------------
        // 3. PACK DATA AND POST SENDS (MPI_Isend)
        // ---------------------------------------------------------------------
        int sendOffset = 0;
        for (const auto& [targetRank, localIDs] : sendMap) 
        {
            int count = localIDs.size() * numFlds;
            int byteCount = count * sizeof(T); // Convert to raw bytes

            if (byteCount == 0) continue;

            // Pack the scattered local data into our contiguous sendBuffer
            int startSendOffset = sendOffset;
            for (int localID : localIDs) {
                int flatIdx = localID * numFlds;
                for (int f = 0; f < numFlds; ++f) {
                    sendBuffer[sendOffset++] = localData[flatIdx + f];
                }
            }

            requests.push_back(MPI_REQUEST_NULL);
            
            // Send the packed vector address, telling MPI it's an array of bytes
            MPI_Isend(&sendBuffer[startSendOffset], byteCount, MPI_BYTE, 
                      targetRank, 0, _comm, &requests.back());
        }

        // ---------------------------------------------------------------------
        // 4. WAIT FOR ALL NETWORK TRANSFERS TO COMPLETE
        // ---------------------------------------------------------------------
        if (!requests.empty()) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }

        // ---------------------------------------------------------------------
        // 5. UNPACK RECEIVED DATA INTO ghostData
        // ---------------------------------------------------------------------
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
};

} // namespace hiperlife

#endif