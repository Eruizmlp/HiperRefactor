#ifndef DOFMANAGER_H
#define DOFMANAGER_H

#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <cstddef>

#include "hl_Core.h"
#include "hl_TypeDefs.h"
#include "hl_DistributedMesh.h"
#include "GenericFieldStruct.h"

namespace hiperlife {

/*!
 * @class DofManager
 * @brief Orchestrates the mathematical state, degrees of freedom, and constraints for a distributed finite element mesh.
 *
 * @details The DofManager acts as the primary interface for setting up the physical problem. 
 * It manages continuous fields (DOFs), auxiliary data, initial conditions, and boundary constraints. 
 * It delegates underlying memory management and MPI ghost synchronization to high-performance, 
 * cache-friendly structures (GenericFieldStruct).
 */
class DofManager : public DistributedClass {
public:
    SmartPtr<DistributedMesh> mesh = hiperlife::null; ///< Shared pointer to the distributed mesh topology.

    SmartPtr<GenericFieldStruct<double>> nodeDOFs = hiperlife::null;  ///< Primary nodal degrees of freedom.
    SmartPtr<GenericFieldStruct<double>> nodeDOFs0 = hiperlife::null; ///< Primary nodal degrees of freedom at the previous time step.
    SmartPtr<GenericFieldStruct<double>> nodeAuxF = hiperlife::null;  ///< Auxiliary fields stored at the nodes.
    SmartPtr<GenericFieldStruct<double>> elemAuxF = hiperlife::null;  ///< Auxiliary fields stored at the elements (cell-centered).

    SmartPtr<GenericFieldStruct<double>> flagConstr = hiperlife::null; ///< Flags indicating if a specific DOF is constrained (Dirichlet BC).
    SmartPtr<GenericFieldStruct<double>> valuConstr = hiperlife::null; ///< The prescribed numerical values of constrained DOFs.

    /*!
     * @brief Constructs a new DofManager associated with a specific distributed mesh.
     * @param inputMesh Shared pointer to the underlying DistributedMesh.
     * @param tag String identifier for the distributed object (defaults to "DofManager").
     */
    DofManager(SmartPtr<DistributedMesh> inputMesh, std::string tag = "DofManager");
    ~DofManager() = default;

    /** @name Initialization Methods */
    ///@{
    /*! @brief Allocates the primary nodal DOFs based on a numeric count. */
    void setNumDOFs(int dofs);
    /*! @brief Allocates the primary nodal DOFs with specific string tags. */
    void setDOFs(const std::vector<std::string>& tags);
    /*! @brief Returns the number of primary DOFs per node. */
    int numDOFs() const;

    /*! @brief Allocates auxiliary nodal fields based on a numeric count. */
    void setNumNodeAuxF(int auxflds);
    /*! @brief Allocates auxiliary nodal fields with specific string tags. */
    void setNodeAuxF(const std::vector<std::string>& tags);
    /*! @brief Returns the number of auxiliary fields per node. */
    int numNodeAuxF() const;

    /*! @brief Allocates auxiliary elemental fields based on a numeric count. */
    void setNumElemAuxF(int auxflds);
    /*! @brief Allocates auxiliary elemental fields with specific string tags. */
    void setElemAuxF(const std::vector<std::string>& tags);
    /*! @brief Returns the number of auxiliary fields per element. */
    int numElemAuxF() const;

    /*! @brief Finalizes configuration, generating BlockMaps and allocating internal contiguous memory buffers. */
    void Update();
    ///@}

    /** @name Ghost Management */
    ///@{
    /*! @brief Synchronizes ghost data across all MPI partitions for all managed field structures. 
     *  @param genImporters If true, forces a recalculation of the topological routing maps. */
    void UpdateGhosts(bool genImporters = false);
    /*! @brief Synchronizes only the constraint flags and values. */
    void UpdateConstrainGhosts(bool genImporters = false);
    /*! @brief Synchronizes only the primary DOFs. */
    void UpdateDOFsGhosts(bool genImporters = false);
    /*! @brief Synchronizes only the previous time step DOFs. */
    void UpdateDOFs0Ghosts(bool genImporters = false);
    /*! @brief Synchronizes only the auxiliary fields. */
    void UpdateAuxFGhosts(bool genImporters = false);
    ///@}

    /** @name Constraints & Boundary Conditions */
    ///@{
    void setConstraint(int dof, int idx, IndexType idxtype, double value);
    void setConstraint(const std::string& tag, int idx, IndexType idxtype, double value);
    void setConstraint(int dof, double value);
    void setConstraint(const std::string& tag, double value);

    void setBoundaryCondition(int dof, int idx, IndexType idxtype, double value);
    void setBoundaryCondition(const std::string& tag, int idx, IndexType idxtype, double value);
    
    /*! @brief Applies a uniform boundary condition across a specific geometric axis. */
    void setBoundaryCondition(int dof, MAxis ax, double value);
    void setBoundaryCondition(const std::string& tag, MAxis ax, double value);
    
    /*! @brief Applies a spatially varying boundary condition across a geometric axis (1D/2D parameterization). */
    void setBoundaryCondition(int dof, MAxis ax, std::function<double(double)> f);
    void setBoundaryCondition(int dof, MAxis ax, std::function<double(double, double)> f);
    
    void setBoundaryCondition(int dof, double value);
    void setBoundaryCondition(double value);

    void removeConstraint(int dof, int idx, IndexType idxtype);
    void removeConstraint(int dof);
    void removeBoundaryCondition(int dof, MAxis ax);

    /*! @brief Checks if a specific DOF at a given index is constrained. */
    bool constrFlag(int dof, int idx, IndexType idxtype) const;
    bool constrFlag(const std::string& tag, int idx, IndexType idxtype) const;
    
    /*! @brief Retrieves the constraint value for a specific DOF at a given index. */
    double constrValue(int dof, int idx, IndexType idxtype) const;
    double constrValue(const std::string& tag, int idx, IndexType idxtype) const;
    ///@}

    /** @name Initial Conditions */
    ///@{
    void setInitialCondition(int dof, int idx, IndexType idxtype, double value);
    void setInitialCondition(const std::string& tag, int idx, IndexType idxtype, double value);
    void setInitialCondition(int dof, double value);
    void setInitialCondition(const std::string& tag, double value);
    
    /*! @brief Applies a spatially varying initial condition using lambda functions. */
    void setInitialCondition(int dof, std::function<double(double)> f);
    void setInitialCondition(int dof, std::function<double(double, double)> f);
    void setInitialCondition(int dof, std::function<double(double, double, double)> f);
    ///@}

    /** @name Field Interpolation */
    ///@{
    /*!
     * @brief Interpolates a subset of fields evaluated at specified reference coordinates within given elements.
     * @param fieldStrs Vector of GenericFieldStruct pointers to extract data from.
     * @param fields Local field indices to interpolate.
     * @param elems Local or global element IDs containing the points.
     * @param rCoords Barycentric/reference coordinates corresponding to the elements.
     * @param type Specifies whether the provided indices are Local or Global.
     * @return A 2D vector where values[element_index][field_index] holds the interpolated result.
     */
    std::vector<std::vector<double>> interpolateField(const std::vector<SmartPtr<GenericFieldStruct<double>>>& fieldStrs, std::vector<std::vector<int>> fields, const std::vector<int>& elems, const std::vector<std::vector<double>>& rCoords, IndexType type = IndexType::Global);
    std::vector<double> interpolateField(const std::vector<SmartPtr<GenericFieldStruct<double>>>& fieldStrs, const std::vector<std::vector<int>>& fields, int elem, const std::vector<double>& rCoords, IndexType type = IndexType::Global);
    
    /*! @brief Interpolates the primary nodeDOFs at specified reference coordinates. */
    std::vector<std::vector<double>> interpolateDOFs(std::vector<int> fields, const std::vector<int>& elems, const std::vector<std::vector<double>>& rCoords, IndexType type = IndexType::Global);
    std::vector<double> interpolateDOFs(std::vector<int> fields, int elem, const std::vector<double>& rCoords, IndexType type = IndexType::Global);
    ///@}

private:
    /*! @brief Internal setup method called during Update() to construct the layout and constraint maps. */
    void _update_new();
};

} 

#endif 