#include "Element.hpp"
#include <iostream>

int main()
{
	try
	{
		auto&& q = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1>{ {1, 2, 8, 6} };
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

//#include <array>
//#include <vector>
//#include <numeric>
//#include <algorithm>
//#include <iostream>
//#include <utility>
//
//std::ostream& operator<<(std::ostream& o, std::vector<double> a)
//{
//	for (auto it : a)
//		o << it << ' ';
//	o << '\n';
//	return o;
//}
//
//int main()
//{
//	auto&& a	= std::vector<double>	{1.1,	5.2,	3.8,	0.};
//	auto&& b	= std::vector<double>	{3.14, 3, 3, 3};
//	std::iota(b.begin(), b.end(), 0.1);
//
//	std::cout << a;
//	std::cout << b;
//
//	auto pa = &(a[0]);
//
//	b = std::move(a);
//
//	auto pb = &(b[0]);
//
//	std::cout << a;
//	std::cout << b;
//
//	std::cout << (pa == pb);
//
//
//	std::cout << (a.begin() == a.end());
//
//	/*std::sort(rrd.begin(), rrd.end(), [&a](int n1, int n2) { return a[n1] < a[n2]; });*/
//}