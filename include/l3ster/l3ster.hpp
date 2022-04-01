#ifndef L3STER_HPP
#define L3STER_HPP

// This header includes the entire L3STER library

#include "l3ster/alloc/NodeGlobalMemoryResource.hpp"
#include "l3ster/comm/DeserializeMesh.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/global_assembly/SparsityPattern.hpp"
#include "l3ster/local_assembly/AssembleLocalSystem.hpp"
#include "l3ster/mapping/ComputeBasisDerivative.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/quad/EvalQuadrature.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/SetStackSize.hpp"

#endif // L3STER_HPP
