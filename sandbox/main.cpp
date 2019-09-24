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

#include <memory>
#include <iostream>

struct Base
{
	virtual void foo() = 0;
};

struct C1 :public Base
{
	void foo() override
	{
		std::cout << "C1" << std::endl;
	}
};

struct C2 :public Base
{
	void foo() override
	{
		std::cout << "C2" << std::endl;
	}
};

void main()
{
	auto&& ptr1 = std::unique_ptr<Base>{ new C1 };
	auto&& ptr2 = std::unique_ptr<Base>{ new C2 };
	ptr1->foo();
	ptr2->foo();
}