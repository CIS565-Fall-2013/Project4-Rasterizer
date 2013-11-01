#ifndef COLOR_READER_H
#define COLOR_READER_H
#include <string>
#include <vector>

using namespace std;

class colorReader{
private:
	float* cbo;
	vector<string> splitString(string s);
public:
	colorReader(int size, std::string file);
	~colorReader();
	float* getCBO();
};

#endif