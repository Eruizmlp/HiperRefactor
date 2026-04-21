#ifndef GHOSTIE_IMPL_H
#define GHOSTIE_IMPL_H

namespace hiperlife {

template <typename T>
GhostIE<T>::GhostIE(std::string tag, MPI_Comm comm) 
    : DistributedClass(tag, comm) 
{}

template <typename T>
void GhostIE<T>::setupImportExport(const GhostManager& manager, int numFlds) 
{
    const auto& sendMap = manager.getSendMap();
    const auto& recvMap = manager.getRecvMap();
    const auto& ghosts = manager.getGhosts();

    _exportRanks.clear();  _exportCounts.clear();
    _importRanks.clear();  _importCounts.clear();
    _exportIndices.clear(); _importIndices.clear();

    int totalExportItems = 0;
    for (const auto& [targetRank, localIDs] : sendMap) {
        int count = localIDs.size() * numFlds;
        if (count > 0) {
            _exportRanks.push_back(targetRank);
            _exportCounts.push_back(count);
            totalExportItems += count;
            
            for (int localID : localIDs) {
                int baseIdx = localID * numFlds;
                for (int f = 0; f < numFlds; ++f) {
                    _exportIndices.push_back(baseIdx + f);
                }
            }
        }
    }
    _exportBuffer.resize(totalExportItems);

    int totalImportItems = 0;
    for (const auto& [targetRank, globalIDs] : recvMap) {
        int count = globalIDs.size() * numFlds;
        if (count > 0) {
            _importRanks.push_back(targetRank);
            _importCounts.push_back(count);
            totalImportItems += count;
            
            for (int globalID : globalIDs) {
                auto it = std::lower_bound(ghosts.begin(), ghosts.end(), globalID);
                int ghostIdx = std::distance(ghosts.begin(), it);
                int baseIdx = ghostIdx * numFlds;
                for (int f = 0; f < numFlds; ++f) {
                    _importIndices.push_back(baseIdx + f);
                }
            }
        }
    }
    _importBuffer.resize(totalImportItems);
}

template <typename T>
void GhostIE<T>::communicate(const GhostManager&, 
                             const std::vector<T>& localData, 
                             std::vector<T>& ghostData, 
                             int numFlds) 
{
    if (_exportRanks.empty() && _importRanks.empty()) return;

    int numRequests = _importRanks.size() + _exportRanks.size();
    _requests.resize(numRequests);
    int reqIdx = 0;

    int importOffset = 0;
    for (size_t i = 0; i < _importRanks.size(); ++i) 
    {
        int count = _importCounts[i];
        MPI_Irecv(&_importBuffer[importOffset], count * sizeof(T), MPI_BYTE, 
                  _importRanks[i], 0, this->_comm, &_requests[reqIdx++]);
        importOffset += count; 
    }

    const T* p_localData = localData.data();
    T* p_exportBuffer = _exportBuffer.data();
    const int* p_exportIndices = _exportIndices.data();
    size_t exportSize = _exportIndices.size();
    
    for (size_t i = 0; i < exportSize; ++i) {
        p_exportBuffer[i] = p_localData[p_exportIndices[i]];
    }

    int exportOffset = 0;
    for (size_t i = 0; i < _exportRanks.size(); ++i) 
    {
        int count = _exportCounts[i];
        MPI_Isend(&_exportBuffer[exportOffset], count * sizeof(T), MPI_BYTE, 
                  _exportRanks[i], 0, this->_comm, &_requests[reqIdx++]);
        exportOffset += count;
    }

    if (numRequests > 0) {
        MPI_Waitall(numRequests, _requests.data(), MPI_STATUSES_IGNORE);
    }

    const T* p_importBuffer = _importBuffer.data();
    T* p_ghostData = ghostData.data();
    const int* p_importIndices = _importIndices.data();
    size_t importSize = _importIndices.size();

    for (size_t i = 0; i < importSize; ++i) {
        p_ghostData[p_importIndices[i]] = p_importBuffer[i];
    }
}

}

#endif