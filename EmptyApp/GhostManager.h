    #ifndef GHOSTMANAGER_H
    #define GHOSTMANAGER_H

    #include <vector>
    #include <map> 
    #include <algorithm> 
    #include "hl_Core.h"
    #include "Block_Map.h"

    namespace hiperlife {

    class GhostManager : public hiperlife::DistributedClass {
    protected:
        BlockMap _map{}; 
        bool _mapIsSet{false};

        std::vector<int> _ghosts{};

        std::map<int, std::vector<int>> _recv;  
        std::map<int, std::vector<int>> _send;

    public:

        GhostManager(std::string tag, MPI_Comm comm) : DistributedClass(tag, comm) {}

        void setBlockMap(const BlockMap& map);

        void setGhostLists(const std::vector<int>& list);

        void Update();        

        const std::vector<int>& getGhosts() const { return _ghosts; }
        bool isGhost(int globIdx) const {
            if (_ghosts.empty()) return false;

            //Ghosts is already sorted thats why we can apply Binary Search
            return std::binary_search(_ghosts.begin(), _ghosts.end(), globIdx);
        }

        const std::map<int, std::vector<int>>& getSendMap() const { return _send; }
        const std::map<int, std::vector<int>>& getRecvMap() const { return _recv; }
    };
    }

    #endif