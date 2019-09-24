// Implementation of creator classes - part of abstract factory design pattern

#pragma once

#include "Factory.h"

#include <memory>

namespace lstr
{
	namespace despat
	{
		template <typename BaseT>
		class CreatorBase
		{
		public:
			virtual std::unique_ptr<BaseT>		Create() const		= 0;
			virtual								~CreatorBase()		= default;
		};

		template<typename KeyType, typename BaseT, typename T>
		class Creator final : public Creator<BaseT>
		{
		public:
			explicit Creator(const KeyType& key);
			virtual std::unique_ptr<BaseT>		Create() const override { return std::unique_ptr<BaseT>{new T}; }
			virtual								~Creator() override			= default;
		};

		template<typename KeyType, typename BaseT, typename T>
		Creator<KeyType, BaseT, T>::Creator(const KeyType& key)
		{
			Factory< KeyType, Creator< BaseT > >::registerCreator(key, this);
		}
	}
}