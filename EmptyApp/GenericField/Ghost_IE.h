#ifndef GHOSTIE_H
#define GHOSTIE_H

#include <vector>
#include <map>
#include <algorithm>
#include "hl_Core.h"
#include "GhostManager.h"

namespace hiperlife {

/*!
 * \class GhostIE
 * \brief Template class responsible for executing high-performance MPI ghost communication.
 *
 * It decouples the communication routing plan from the execution phase. By relying on pre-calculated, 
 * strictly 1D contiguous arrays (flattened topology), it guarantees zero dynamic allocations and 
 * zero mathematical operations in the critical execution loop, enabling aggressive CPU auto-vectorization (SIMD) 
 * and maximizing cache hits during the MPI data packing/unpacking phases.
 */
template <typename T>
class GhostIE : public hiperlife::DistributedClass {
protected:
    std::vector<T> _exportBuffer;       //!< Contiguous memory buffer for outgoing MPI data.
    std::vector<T> _importBuffer;       //!< Contiguous memory buffer for incoming MPI data.
    std::vector<MPI_Request> _requests; //!< Array storing active non-blocking MPI requests.
    
    std::vector<int> _exportIndices;    //!< Pre-calculated 1D flat indices for direct memory gathering.
    std::vector<int> _importIndices;    //!< Pre-calculated 1D flat indices for direct memory scattering.

    std::vector<int> _exportRanks;      //!< Flattened array of destination MPI ranks.
    std::vector<int> _exportCounts;     //!< Flattened array containing the number of items to send per rank.
    std::vector<int> _importRanks;      //!< Flattened array of source MPI ranks.
    std::vector<int> _importCounts;     //!< Flattened array containing the number of items to receive per rank.

public:

    /** @name Constructor
     * @{ */
    /*! \brief Initializes the Ghost Import/Export communicator. 
     \param tag the string identifier for the class instance.
     \param comm the MPI communicator to be used. */
    GhostIE(std::string tag, MPI_Comm comm);
    /** @} */

    /** @name Communication Plan Planning
     * @{ */
    /*! \brief Generates the optimized routing arrays needed for communication.
     * It translates the topological maps from GhostManager into 1D contiguous arrays to avoid 
     * pointer-chasing during the high-frequency solver loops.
     \param manager the GhostManager containing the send/recv topological maps.
     \param numFlds the number of fields (variables) per item. */
    void setupImportExport(const GhostManager& manager, int numFlds);
    /** @} */

   /** @name Communication Execution
     * @{ */
    /*! \brief Starts the non-blocking MPI data exchanges (Latency Hiding - Phase 1).
     * It posts all MPI_Irecv requests, packs the local data into the export buffer, 
     * and posts all MPI_Isend requests. Returns immediately without waiting for the network.
     \param manager the GhostManager.
     \param localData the array containing the local field data (source for exports).
     \param numFlds the number of fields (variables) per item. */
    void startCommunication(const GhostManager& manager, const std::vector<T>& localData, int numFlds);

    /*! \brief Completes the non-blocking MPI data exchanges (Latency Hiding - Phase 2).
     * It waits for all pending MPI requests to finish (MPI_Waitall) and unpacks 
     * the received data into the ghost array.
     \param manager the GhostManager.
     \param ghostData the array where incoming ghost data will be unpacked.
     \param numFlds the number of fields (variables) per item. */
    void completeCommunication(const GhostManager& manager, std::vector<T>& ghostData, int numFlds);
    /** @} */
};

} 

#include "Ghost_IE-impl.h"

#endif