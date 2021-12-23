#include "NumCpp.hpp"
#include "network.h"
#include "filereader.h"

#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;
constexpr int REPEAT_TIMES_FOR_ONE          = 2;
constexpr int MINIEPOCH_SIZE                = 1000;
constexpr int REPEAT_TIMES_FOR_MINIEPOCH    = 10;
constexpr int REPEAT_TIMES_FOR_TEST = 100;
int main()
{
    std::string filename = "D:\\Documents\\Code\\BPNN_1\\traindata\\train.csv";
    FileReader fr(filename);
    Network n(784, 30, 10, 0.35);

    Label_img_pair pairt;
    int labels[MINIEPOCH_SIZE];
    NdArray<double> nors[MINIEPOCH_SIZE];
    for (int mn = 0; mn < 38; mn++) {
        // Data Normalizer and Storage
        for (int i = 0; i < MINIEPOCH_SIZE; ++i) {
            pairt = fr.next();
            nors[i] = n.NormalizeData(pairt.img);
            labels[i] = pairt.label;
        }
        // Train
        for (int y = 0; y < REPEAT_TIMES_FOR_MINIEPOCH; ++y)
        {
            for (int x = 0; x < MINIEPOCH_SIZE; ++x) {
                n.LoadData(nors[x], labels[x]);
                n.front_proceed();
                n.back_proceed();
            }
        }
        n.eta -= 0.002;
        cout << mn << endl;
        // test for another 100 items
        int t = 0, f = 0;
        for (int x = 0; x < 100; ++x) {
            Label_img_pair y = fr.next();
            auto nor = n.NormalizeData(y.img);
            n.LoadData(nor, y.label);
            n.front_proceed();
            if (n.answerCheck()) t += 1;
            else f += 1;
        }
        cout << t << "/" << t + f << "\n";
    }
    // submission
    std::string filename2 = "D:\\Documents\\Code\\BPNN_1\\traindata\\test.csv";
    FileReader fr2(filename2);
    fr2.onlytest = true;
    Network &nt = n;
    const char* filename3 = "D:\\Documents\\Code\\BPNN_1\\traindata\\sub2.csv";
    FILE* fp;
    fopen_s(&fp,filename3,"w");
    fprintf(fp, "ImageId,Label\n");
    int t = 0, f = 0;
    Label_img_pair pr;
    int cnt = 0;
    cout << "testing\n";
    while (pr = fr2.next(), pr.label != -1) {
        cnt += 1;
        nt.LoadData(nt.NormalizeData(pr.img),-1);
        nt.front_proceed();
        fprintf(fp, "%d,%d\n", cnt, nt.answer());
    }
    fclose(fp);
    cout << "saved\n";
}