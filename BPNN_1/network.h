#include <NumCpp.hpp>
#include <iostream>
#include <vector>
#include "Ncmath.hpp"
using namespace nc;

class Network
{
private:
	nc::uint32 L1, L2, L3;
	int label = -1;
	NdArray<double> Xin, Xout, W1, Yin, Yout, W2, Zin, Zout;
public:
	double eta;
	Network(int i, int h, int o, double eta) :L1(i), L2(h), L3(o), eta(eta) {
		W1 = random::normal<double>({ L1,L2 });
		W2 = random::normal<double>({ L2,L3 });
		Xin = zeros<double>({ L1,1 });
		Xout = zeros<double>({ L1,1 });
		Yin = zeros<double>({ 1,L2 });
		Yout = zeros<double>({ 1,L2 });
		Zin = zeros<double>({ 1,L3 }); 
		Zout = zeros<double>({ 1,L3 });
	}
	static NdArray<double> NormalizeData(std::vector<int>& source) {
		double t[784];
		for (int i = 0; i < 784; ++i) {
			t[i] = source[i] * 0.00390625;// d b 255
		}
		return NdArray<double>(t, 784, 1);

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
		// 1 X L3
		auto return_array = zeros<double>({1,L3}); 
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
		// deltaZ:  1 X L3
		// W2: L2 X L3
		auto delta_z = (Zout - anti_softmax(label))*(sigmoid_d_out(Zout));
		W2 -= eta * (transpose(Yout).dot(delta_z));

		// ÐÞÕýÒþº¬²ã
		W1 -= eta * (Xin.dot(sigmoid_d_out(Yout) * (delta_z.dot(transpose(W2)))));
	}
	void printZout() {
		std::cout << Zout;
	}
	void printZin() {
		std::cout << Zin;
	}
	int answer() {
		return softmax(Zout);
	}
	bool answerCheck() {
		return label == softmax(Zout);
	}
};
