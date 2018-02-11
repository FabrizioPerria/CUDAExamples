#pragma once

#include <iostream>
#if __cplusplus > 201103L
#include <chrono>
template<class Resolution = std::chrono::milliseconds>
class CPUTimer {

	using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
				std::chrono::high_resolution_clock,
				std::chrono::steady_clock>;

	private:
		Clock::time_point _start = Clock::now();

	public:
		CPUTimer(void) = default;
		~CPUTimer(void)
		{
			std::cout <<
				std::chrono::duration_cast<Resolution>(Clock::now() - _start).count() <<
				std::endl;

		}
};
#else
class CPUTimer {
	private:
		clock_t _start = clock();
	public:
		CPUTimer(void) = default;
		~CPUTimer(void)
		{
			std::cout << (clock() - _start) / (float)CLOCKS_PER_SEC * 1000.0f << " ms" << std::endl;
		}
#endif
};
