#ifndef NEW_BEAM_SEARCH_CODETRIE_H
#define NEW_BEAM_SEARCH_CODETRIE_H
#include <cstdlib>
#include <map>
#include "Utils.h"
class CodeTrie{
public:
    int value = -1; // 106:条形码结尾
    std::map<int, CodeTrie*> next;
    static void createCodeTrie(CodeTrie* startCodeTrie, CodeTrie* otherCodeTrie);
};

#endif //NEW_BEAM_SEARCH_CODETRIE_H
