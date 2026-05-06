#ifndef GHOSTMANAGER_H
    #define GHOSTMANAGER_H

    #include <vector>
    #include <map> 
    #include <algorithm> 
    #include "hl_Core.h"
    #include "Block_Map.h"

    namespace hiperlife {

    /*!
     * \class GhostManager
     * \brief Class responsible for determining the topological routing of ghost items between MPI partitions.
     *
     * It maps which local items need to be sent to other partitions, and which items need to be 
     * received from other partitions, generating the topological maps required for MPI communication.
     */
    class GhostManager : public hiperlife::DistributedClass {
    protected:
        BlockMap _map{}; 
        bool _mapIsSet{false};

        std::vector<int> _ghosts{}; //!< Sorted list of global IDs for ghost items.

        std::map<int, std::vector<int>> _recv;  //!< Map [TargetRank -> List of Global IDs to receive]
        std::map<int, std::vector<int>> _send;  //!< Map [TargetRank -> List of Global IDs to send]

    public:

        /** @name Constructor
         * @{ */
        GhostManager(std::string tag, MPI_Comm comm) : DistributedClass(tag, comm) {}
        /** @} */

        /** @name Initialization
         * @{ */
        /*! \brief Sets the BlockMap that defines the global distribution of the items. */
        void setBlockMap(const BlockMap& map);
        
        /*! \brief Sets the raw list of candidate ghost items. */
        void setGhostLists(const std::vector<int>& list);

        /*! \brief Computes the send and receive topological maps. */
        void Update();        
        /** @} */

        /** @name Query Methods
         * @{ */
        /*! \brief Returns the complete sorted list of ghost items required by the local partition. */
        const std::vector<int>& getGhosts() const { return _ghosts; }
        
        /*! \brief Returns true if a given global index is tracked as a ghost by the local partition. */
        bool isGhost(int globIdx) const {
            if (_ghosts.empty()) return false;
            // Ghosts is already sorted thats why we can apply Binary Search
            return std::binary_search(_ghosts.begin(), _ghosts.end(), globIdx);
        }

        /*! \brief Returns the topological map of items to send to other partitions. */
        const std::map<int, std::vector<int>>& getSendMap() const { return _send; }
        
        /*! \brief Returns the topological map of items to receive from other partitions. */
        const std::map<int, std::vector<int>>& getRecvMap() const { return _recv; }
        /** @} */
    };
    }

    #endif