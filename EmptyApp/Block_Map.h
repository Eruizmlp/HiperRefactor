#ifndef BLOCK_MAP_H
#define BLOCK_MAP_H

#include <vector>
#include <string>
#include <mpi.h>
#include "hl_Core.h" 

namespace hiperlife {

class BlockMap : public hiperlife::DistributedClass {
protected:
    int _nItem{0};       // Total global items
    int _loc_nItem{0};   // Local items
    std::vector<int> _offs_nItem; // Offsets [Rank0, Rank1, ..., Total]

public:
    // Constructor por defecto
    BlockMap() : DistributedClass("DefaultMap", MPI_COMM_WORLD) {}
    
    BlockMap(std::string tag, MPI_Comm comm) : DistributedClass(tag, comm) {}
    
    // --- Getters ---
    int nItem() const { return _nItem; }
    int loc_nItem() const { return _loc_nItem; }
    
    int myOffset() const { 
        if (_offs_nItem.empty()) return 0;
        return _offs_nItem[_myRank]; 
    } 

    const std::vector<int>& getOffsets() const { return _offs_nItem; }

    std::vector<int> getCounts() const {
        if (_offs_nItem.empty()) return {};
        std::vector<int> counts(_numProcs);
        for(int i=0; i<_numProcs; ++i) {
            counts[i] = _offs_nItem[i+1] - _offs_nItem[i];
        }
        return counts;
    }

    // --- Conversión de Índices (Nuevo Refactor) ---
    int getLocalIndex(int globalIdx) const;
    int getGlobalIndex(int localIdx) const;

    // --- Utilidades Legacy (Para compatibilidad con tu cpp) ---
    // Devuelve qué rank es dueño de un índice global
    int getItemPartition(int globIdx) const; 

    // --- Setters ---
    void setLocNItem(int n) { _loc_nItem = n; }

    // --- Update ---
    void Update(); 

private:
    // Métodos auxiliares requeridos por tu implementación actual
    void computeOffsets();
    int bisection(int globIdx) const;
};

} 
#endif