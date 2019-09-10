#include "Element.hpp"
#include <iostream>

int main()
{
	try
	{
		lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1> q;
		//typename lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1>::node_array_constref a = q.getNodes();
		q.pushBackNode(1);
		q.pushBackNode(3);
		q.pushBackNode(2);
		q.pushBackNode(4);
		q.sort();
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cout << "Unknown exception was thrown" << std::endl;
		return 2;
	}
}