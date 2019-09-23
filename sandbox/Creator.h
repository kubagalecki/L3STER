// Implementation of creator classes - part of abstract factory design pattern

#pragma once

#include "Singleton.h"
#include "Factory.h"

namespace lstr
{
	namespace despat
	{
		template <typename BaseT>
		class CreatorBase
		{
		public:
			virtual BaseT*		Create() const = 0;
			virtual				~CreatorBase()			{}
		};

		template<typename KeyType, typename T, typename BaseT>
		class Creator final : public Creator<BaseT>
		{
		public:
			explicit		Creator(const KeyType& key)
			{
				Singleton<Factory< BaseT, Creator< BaseT > >>::getInstance()->registerCreator(key, this);
			}

			virtual BaseT*	Create() const override		{ return new T; }

			virtual			~Creator() override			{}
		};
	}
}