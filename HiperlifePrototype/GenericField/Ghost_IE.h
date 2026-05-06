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
 * It uses pre-calculated, strictly 1D contiguous arrays (flattened topology) to guarantee 
 * zero dynamic allocations and maximize cache hits during the MPI data packing/unpacking phases.
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
     *  @{ */
    /*! \brief Initializes the Ghost Import/Export communicator.
     *  \param tag The string identifier for the class instance.
     *  \param comm The MPI communicator to be used.
     */
    GhostIE(std::string tag, MPI_Comm comm);
    /** @} */

    /** @name Communication Plan Planning
     *  @{ */
    /*! \brief Generates the optimized routing arrays needed for communication.
     *
     *  Translates the topological maps from GhostManager into 1D contiguous arrays.
     *  \param manager The GhostManager containing the send/recv topological maps.
     *  \param numFlds The number of fields (variables) per item.
     */
    void setupImportExport(const GhostManager& manager, int numFlds);
    /** @} */

   /** @name Communication Execution
     *  @{ */
    /*! \brief Executes the full MPI data exchange synchronously.
     *
     *  It posts all MPI_Irecv requests, packs local data, posts MPI_Isend requests, 
     *  waits for completion, and unpacks the received data into the ghost array.
     *  \param manager The GhostManager containing topology info.
     *  \param localData The array containing the local field data (source for exports).
     *  \param ghostData The array where incoming ghost data will be unpacked.
     *  \param numFlds The number of fields (variables) per item.
     */
    void startCommunication(const GhostManager& manager, 
                            const std::vector<T>& localData, 
                            std::vector<T>& ghostData, 
                            int numFlds);
    /** @} */
};

}

#include "Ghost_IE-impl.h"

#endif