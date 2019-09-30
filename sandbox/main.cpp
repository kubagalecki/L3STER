//#include "Element.hpp"
//#include <iostream>
//
//int main()
//{
//	try
//	{
//		auto&& q = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1>{ {1, 2, 8, 6} };
//		return 0;
//	}
//	catch (const std::exception& e)
//	{
//		std::cout << e.what() << std::endl;
//		return 1;
//	}
//	catch (...)
//	{
//		std::cout << "Unknown exception was thrown" << std::endl;
//		return 2;
//	}
//}

#include "Factory.hpp"
#include <vector>
#include <iostream>

struct Base
{
	virtual void foo() = 0;
};

struct C1 : public Base
{
	virtual void foo() override { std::cout << "C1" << '\n'; }
};

struct C2 : public Base
{
	virtual void foo() override { std::cout << "C2" << '\n'; }
};

void init()
{
	lstr::despat::Factory<int, Base>::registerCreator<C1>(1);
	lstr::despat::Factory<int, Base>::registerCreator<C2>(2);
}

int main()
{

	try {
		init();
		std::vector<std::unique_ptr< Base >> vec;
		vec.push_back(lstr::despat::Factory<int, Base>::create(1));
		vec.push_back(lstr::despat::Factory<int, Base>::create(2));
		for (auto&& it : vec)
			it->foo();

		// Output:
		// C1
		// C2

		return 0;
	}
	catch (...)
	{
		return 1;
	}
}