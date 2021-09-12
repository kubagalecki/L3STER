#include "comm/DeserializeMesh.hpp"
#include "mesh/ReadMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

bool operator==(const lstr::MeshPartition& p1, const lstr::MeshPartition& p2)
{
    using namespace lstr;

    bool result = true;

    p1.cvisit([&]< ElementTypes T1, el_o_t O1 >(const Element< T1, O1 >& el1, DomainView dv) {
        const auto matched = p2.find(el1.getId());

        if (not matched or dv.getID() != matched->second)
        {
            result = false;
            return;
        }

        std::visit(
            [&]< ElementTypes T2, el_o_t O2 >(const Element< T2, O2 >* el2) {
                if constexpr (T1 != T2 or O1 != O2)
                    result = false;
                else if (el1.getNodes() != el2->getNodes() or el1.getData().vertices != el2->getData().vertices)
                    result = false;
            },
            matched->first);
    });

    return result;
}

TEST_CASE("Mesh serialization", "[mesh-serial]")
{
    const auto  mesh              = lstr::readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), lstr::gmsh_tag);
    const auto& original_part     = mesh.getPartitions()[0];
    const auto  serialized_part   = lstr::SerializedPartition{original_part};
    const auto  deserialized_part = lstr::deserializePartition(serialized_part);

    CHECK(deserialized_part.getNodes() == original_part.getNodes());
    CHECK(deserialized_part.getGhostNodes() == original_part.getGhostNodes());
    CHECK(deserialized_part == original_part);
}
