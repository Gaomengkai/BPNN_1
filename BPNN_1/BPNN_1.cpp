#include <cstdlib>
#include <iostream>
#include <vector>

#include "NumCpp.hpp"
#include "network.h"
#include "filereader.h"

using namespace std;
constexpr int MINIEPOCH_SIZE = 1000;
constexpr int REPEAT_TIMES_FOR_MINIEPOCH = 25;
constexpr int REPEAT_TIMES_FOR_TEST = 100;
constexpr int NUM_TOTALSET = 42000;
Network n(784, 30, 10, 0.35);
OneImg pairt;
int labels[MINIEPOCH_SIZE];
NdArray<double> nors[MINIEPOCH_SIZE];
int main()
{
    std::string filename = "D:\\Documents\\Code\\BPNN_1\\traindata\\train.csv";
    FileReader fr(filename);

    for (int mn = 0; mn < NUM_TOTALSET / (MINIEPOCH_SIZE + REPEAT_TIMES_FOR_TEST); mn++) {
        // Data Normalizer and Storage
        for (int i = 0; i < MINIEPOCH_SIZE; ++i) {
            pairt = fr.next();
            nors[i] = n.NormalizeData(pairt.img, 784);
            labels[i] = pairt.label;
        }
        // Train
        for (int y = 0; y < REPEAT_TIMES_FOR_MINIEPOCH; ++y)
        {
            for (int x = 0; x < MINIEPOCH_SIZE; ++x) {
                n.LoadData(nors[x], labels[x]);
                n.forward();
                n.back();
            }
        }
        n.eta -= 0.002;
        cout << mn << endl;
        // test for another REPEAT_TIMES_FOR_TEST items
        int t = 0, f = 0;
        for (int x = 0; x < REPEAT_TIMES_FOR_TEST; ++x) {
            OneImg y = fr.next();
            auto nor = n.NormalizeData(y.img, 784);
            n.LoadData(nor, y.label);
            n.forward();
            if (n.answerCheck()) t += 1;
            else f += 1;
        }
        cout << t << "/" << t + f << "\n";
    }

    // submission
    std::string filename2 = "D:\\Documents\\Code\\BPNN_1\\traindata\\test.csv";
    FileReader fr2(filename2);
    fr2.onlytest = true;
    Network& nt = n;
    const char* filename3 = "D:\\Documents\\Code\\BPNN_1\\traindata\\sub3.csv";
    FILE* fp;
    errno_t err = fopen_s(&fp, filename3, "w");
    if (err != 0 || (!fp)) {
        printf("文件打开失败！\nFailed to save file\n");
        return -1;
    }
    fprintf(fp, "ImageId,Label\n");
    int t = 0, f = 0;
    OneImg pr;
    int cnt = 0;
    puts(
        "Testing..."
    );
    while (pr = fr2.next(), pr.label != -1) {
        cnt += 1;
        nt.LoadData(nt.NormalizeData(pr.img, 784), -1);
        nt.forward();
        fprintf(fp, "%d,%d\n", cnt, nt.answer());
    }
    fclose(fp);
    puts(
        "Saved."
    );
    return 0;
}