#pragma once
#include <iostream>
#include <vector>
using namespace std;

/// <summary>
/// 标志-图像对
/// </summary>
using OneImg = struct _oneimg{
	int label;
	std::vector<int> img;
};

class FileReader {
private:
	std::fstream fs;
	istringstream is;
	string line;
public:
	bool onlytest = false;
	FileReader(std::string filename) {
		fs.open(filename, std::ios_base::in);
		getline(fs, line);
	}
	/// <summary>
	/// 从文件中获取下一个。
	/// </summary>
	/// <returns>一个图像-标志对。如果没有图像了，则label=-1</returns>
	OneImg next() {
		if (!getline(fs, line)) return {-1,vector<int>(0)};
		is = istringstream(line);
		int label = -2;
		int t;
		char c;
		vector<int> img(784,0);
		if (!onlytest) {
			is >> label;
			is >> c;
		}
		for (int i = 0; i < 784; ++i) {
			is >> img[i];
			is >> c;
		}
		return { label,img };
	}
};