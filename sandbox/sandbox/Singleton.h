// Implementation of the Singleton design pattern

#pragma once

namespace lstr
{
	namespace despat
	{
		template<class T>
		class Singleton
		{
		public:
			static T* getInstance()
			{
				if (!minstance)
				{
					minstance = new T;
				}
				return minstance;
			}

			Singleton()									= delete;
			Singleton(const Singleton&)					= delete;
			Singleton& operator=(const Singleton&)		= delete;
			Singleton(const Singleton&&)				= delete;
			Singleton& operator=(const Singleton&&)		= delete;
			~Singleton()								= delete;

		private:
			static T* minstance;
		};

		template<class T>
		T* Singleton<T>::minstance = nullptr;
	}
}