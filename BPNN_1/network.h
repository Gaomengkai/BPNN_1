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
	/// <summary>
	/// Constructor to create a network for tranning and calculating
	/// </summary>
	/// <param name="i">Num of Input Layer dimension</param>
	/// <param name="h">Num of Hidden Layer dimension</param>
	/// <param name="o">Num of Output Layer dimension</param>
	/// <param name="eta">η</param>
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
	static NdArray<double> NormalizeData(std::vector<int>& source, int dimension) {
		auto t = new double[dimension];
		for (int i = 0; i < dimension; ++i) {
			t[i] = source[i] * 0.00390625;// d b 255
		}
		return NdArray<double>(t, dimension, 1);

	}
	void LoadData(NdArray<double> image,int label) {
		Xin = image;
		this->label = label;
	}
	/// <summary>
	/// 标准化输出结果
	/// </summary>
	/// <param name="out">单层10维向量</param>
	/// <returns></returns>
	int target(NdArray<double> out) {
		int t = 0;
		double mx = -1;
		for (int i = 0; i < 10; ++i) {
			if (out[i] > mx) { mx = out[i]; t = i; }
		}
		return t;
	}
	/// <summary>
	/// 把x转换为一个带刺的0数组
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	NdArray<double> antiTarget(int x) {
		// 1 X L3
		auto return_array = zeros<double>({1,L3}); 
		return_array[x] = 1;
		return return_array;
	}
	/// <summary>
	/// 前向传播
	/// </summary>
	void forward() {
		Xout = transpose(Xin);
		Yin = Xout.dot(W1);
		Yout = sigmoid(Yin);
		Zin = Yout.dot(W2);
		Zout = sigmoid(Zin);
	}
	/// <summary>
	/// 反向传播
	/// </summary>
	void back() {
		// 修正输出层
		// deltaZ:  1 X L3
		// W2: L2 X L3
		auto delta_z = (Zout - antiTarget(label))*(sigmoid_d_out(Zout));
		W2 -= eta * (transpose(Yout).dot(delta_z));

		// 修正隐含层
		W1 -= eta * (Xin.dot(sigmoid_d_out(Yout) * (delta_z.dot(transpose(W2)))));
	}
	int answer() {
		return target(Zout);
	}
	bool answerCheck() {
		return label == target(Zout);
	}
};
