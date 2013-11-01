#include "colorReader.h"
#include <iostream>
#include <fstream>

#include <sstream>
using namespace std;

//This function is from my own OBJ loader from CIS277
vector<string> colorReader::splitString(string s){
	vector<string> tokenVector; //vector of each split word in the string.
	stringstream ss; //create a stringstream
	for(int i = 0; i < s.length(); i++){
		if(s[i] == ' '){
			if(ss.str().length() > 0){
				tokenVector.push_back(ss.str());
			}
			//reset the stringstream
			ss.str(std::string());
			ss.clear();
		}
		else{
			ss << s[i]; //append current character
		}
	}
	if( (ss.str().size() > 0) && (ss.str()[0] != ' ') )
		tokenVector.push_back(ss.str()); //put the last word in the tokenVector

	return tokenVector;
}

colorReader::colorReader(int size, std::string file){

	cbo = new float[3*size];
	std::cout << "Opening: " << file << std::endl;
	std::ifstream currFile(file);
	int cboIdx = 0;
	if( currFile.is_open()){
		std::string currLine;
		while(currFile.good()){
			std::getline(currFile, currLine);
			//split the line using a space delimiter
			std::vector<std::string> currTokens = splitString(currLine);
			cbo[cboIdx] = atof(currTokens[0].c_str());
			cbo[cboIdx + 1] = atof(currTokens[1].c_str());
			cbo[cboIdx + 2] = atof(currTokens[2].c_str());
			//read the three values into the cbo
			cboIdx += 3;
		}
	}
	currFile.close();
}

colorReader::~colorReader(){
	delete cbo;
}