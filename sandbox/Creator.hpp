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
			using						ptr_t					= std::shared_ptr<BaseT>;
			virtual ptr_t				create() const			= 0;
			virtual						~CreatorBase()			= default;
		};

		template<typename KeyType, typename BaseT, typename T>
		class Creator final : public CreatorBase<BaseT>
		{
		public:
			// CTORS & DTORS
			explicit			Creator(const KeyType& key);
			virtual				~Creator() override				= default;

			// METHODS
			virtual typename CreatorBase<BaseT>::ptr_t		create() const override
			{ return std::make_unique<T>(); }

		};

		template<typename KeyType, typename BaseT, typename T>
		Creator<KeyType, BaseT, T>::Creator(const KeyType& key)
		{
			Factory<KeyType, CreatorBase< BaseT >>::
				registerCreator(key, std::shared_ptr<CreatorBase< BaseT >>(this));
		}
	}
}