#include "l3ster/comm/DeserializeMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

using namespace lstr;

template < el_o_t... orders >
bool compareEqual(const mesh::MeshPartition< orders... >& p1, const mesh::MeshPartition< orders... >& p2)
{
    bool result = true;

    p1.visit(
        [&]< mesh::ElementType T1, el_o_t O1 >(const mesh::Element< T1, O1 >& el1, mesh::DomainView< orders... > dv) {
            const auto matched = p2.find(el1.getId());

            if (not matched or dv.getID() != matched->second)
            {
                result = false;
                return;
            }

            std::visit(
                [&]< mesh::ElementType T2, el_o_t O2 >(const mesh::Element< T2, O2 >* el2) {
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
    const auto mesh              = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), lstr::mesh::gmsh_tag);
    const auto serialized_mesh   = SerializedPartition{mesh};
    const auto deserialized_mesh = deserializePartition< 1 >(serialized_mesh);

    CHECK(std::ranges::equal(deserialized_mesh.getOwnedNodes(), mesh.getOwnedNodes()));
    CHECK(std::ranges::equal(deserialized_mesh.getGhostNodes(), mesh.getGhostNodes()));
    CHECK(compareEqual(deserialized_mesh, mesh));
}
