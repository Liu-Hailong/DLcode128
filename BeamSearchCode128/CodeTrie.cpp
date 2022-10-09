#include "CodeTrie.h"


void createPath(CodeTrie* root, const int *x, int len){
    CodeTrie* codeTrie = root;
    int code;
    for (int i = 0; i < len; i++){
        code = x[i];
        if (codeTrie->next.find(code) == codeTrie->next.end()){
            auto *codeTire_new = new CodeTrie();
            codeTrie->next[code] = codeTire_new;
            codeTrie = codeTire_new;
        }else codeTrie = codeTrie->next[code];
    }
    codeTrie->value = x[len];
}

void CodeTrie::createCodeTrie(CodeTrie* startCodeTrie, CodeTrie* otherCodeTrie){
    // code128_start[3][7]
    for(const auto & i : code128_start) createPath(startCodeTrie, i, 6);
    // code128[103][7]
    for(const auto & i : code128) createPath(otherCodeTrie, i, 6);
    // code128_end[8]
    createPath(otherCodeTrie, code128_end, 7);
}

//int main(){
//    auto startCodeTrie = new CodeTrie();
//    auto otherCodeTrie = new CodeTrie();
//    CodeTrie::createCodeTrie(startCodeTrie, otherCodeTrie);
//    return 0;
//}
