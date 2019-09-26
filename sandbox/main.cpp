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

//#include <memory>
//#include <iostream>
//#include <vector>
//
//struct BaseR {
//	virtual void foo() const = 0;
//};
//
//struct R1 : BaseR {
//	void foo() const override
//	{
//		std::cout << "R1" << std::endl;
//	}
//};
//
//struct R2 : BaseR {
//	void foo() const override
//	{
//		std::cout << "R2" << std::endl;
//	}
//};
//
//struct Base
//{
//	virtual const BaseR& get() = 0;
//};
//
//struct C1 :public Base
//{
//	R1 a{};
//	virtual const R1& get() override
//	{
//		return a;
//	}
//};
//
//struct C2 :public Base
//{
//	R2 a{};
//	virtual const R2& get() override
//	{
//		return a;
//	}
//};
//
//int main()
//{
//	std::vector<Base*> a{};
//	a.push_back(new C1);
//	a.push_back(new C2);
//	a.push_back(new C1);
//	for (auto it : a)
//		it->get().foo();
//
//	return 0;
//}

#include "Creator.hpp"
#include <vector>
#include <iostream>

struct A
{
	virtual ~A(){}
};

struct B : public A
{
	virtual ~B(){}
};

struct C : public A
{
	virtual ~C() {}
};

void init()
{
	//lstr::despat::Creator<int, A, B>{ 1 };
	auto a = std::make_shared<lstr::despat::Creator<int, A, C>>(1);
	if (a)
		std::cout << 1;
	else
		std::cout << 0;
	auto x = a->create();
}

int main()
{
	
	try {
		init();
		return 0;
		std::vector<std::unique_ptr<A>> vec;
		vec.push_back(std::make_unique<B>());
		vec.push_back(std::make_unique<C>());
		return 0;
	}
	catch (...)
	{
		return 1;
	}
}