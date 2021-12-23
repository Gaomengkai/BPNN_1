#include <NumCpp.hpp>
#include <iostream>
#include "Ncmath.hpp"
using namespace nc;

class Network
{
private:
	nc::uint32 L1, L2, L3;
	int label = -1;
	NdArray<double> Xin, Xout, W1, Yin, Yout, W2, Zin, Zout;
public:
	Network(int i, int h, int o) :L1(i), L2(h), L3(o) {
		W1 = random::normal<double>({ L1,L2 });
		W2 = random::normal<double>({ L2,L3 });
		Xin = zeros<double>({ L1,1 });
		Xout = zeros<double>({ L1,1 });
		Yin = zeros<double>({ L2,1 });
		Yout = zeros<double>({ L2,1 });
		Zin = zeros<double>({ L3,1 }); 
		Zout = zeros<double>({ L3,1 });
	}
	void LoadData(NdArray<double> image,int label) {
		Xin = image;
		this->label = label;
	}
	int softmax(NdArray<double> out) {
		int t = 0;
		double mx = -1;
		for (int i = 0; i < 10; ++i) {
			if (out[i] > mx) { mx = out[i]; t = i; }
		}
		return t;
	}
	NdArray<double> anti_softmax(int x) {
		auto return_array = zeros<double>(L3,1);
		return_array[x] = 1;
		return return_array;
	}
	void front_proceed() {
		Xout = transpose(Xin);
		Yin = Xout.dot(W1);
		Yout = sigmoid(Yin);
		Zin = Yout.dot(W2);
		Zout = sigmoid(Zin);
	}
	void back_proceed() {
		// ÐÞÕýÊä³ö²ã

	}
	void printZout() {
		std::cout << softmax(Zout);
	}
};
