#pragma once
#include <NumCpp.hpp>

template<typename T>
inline T sigmoid(T x) {
	return 1.0 / (1.0 + nc::exp(-x));
}

template<typename T>
inline T sigmoid_d_out(T y) {
	return y * (1.0 - y);
}
