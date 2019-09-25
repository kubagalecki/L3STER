// Implementation of creator classes - part of abstract factory design pattern

#pragma once

#include "Factory.hpp"

#include <memory>

namespace lstr
{
	namespace despat
	{
		template <typename BaseT>
		class CreatorBase
		{
		public:
			using						ptr_t					= std::unique_ptr<BaseT>;
			virtual ptr_t				Create() const			= 0;
			virtual						~CreatorBase()			= default;
		};

		template<typename KeyType, typename BaseT, typename T>
		class Creator final : public Creator<BaseT>
		{
		public:
			// CTORS & DTORS
			explicit			Creator(const KeyType& key);
			virtual				~Creator() override				= default;

			// METHODS
			virtual ptr_t		Create() const override			{ return std::make_unique<T>(); }

		};

		template<typename KeyType, typename BaseT, typename T>
		Creator<KeyType, BaseT, T>::Creator(const KeyType& key)
		{
			Factory<KeyType, Creator< BaseT >>::registerCreator
			(
				key, std::unique_ptr<CreatorBase< BaseT >>(this)
			);
		}
	}
}