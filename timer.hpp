/*
	timer.hpp
*/

#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

class Timer {

	std::chrono::high_resolution_clock::time_point beginning;
	std::chrono::high_resolution_clock::time_point end;
	bool running;

public:
	Timer() { }

	void start() {
		beginning = std::chrono::high_resolution_clock::now();
		running = true;
	}

	void stop() {
		end = std::chrono::high_resolution_clock::now();
		running = false;
	}

	// int result() {
	// 	duration<double> duration = end - beginning;
	// 	return static_cast<int>(duration.count() * 1000);
	// }

	double result() {
		if (running) stop();
		// duration<double, std::milli> duration = end - beginning; // NOTE: Result in ms
		std::chrono::duration<double> duration = end - beginning; // NOTE: Result in seconds
		return duration.count();
	}

	friend std::ostream& operator<<(std::ostream& os, Timer& timer) {
		os << timer.result();
		return os;
	}

};

#endif
