//
// Created by 31816 on 2022/3/14.
//

#ifndef NEW_BEAM_SEARCH_CTC_BEAM_SEARCH_H
#define NEW_BEAM_SEARCH_CTC_BEAM_SEARCH_H
#include <string>
#include <utility>
#include <vector>
#include "Utils.h"
#include "PathTrie.h"
#include "ThreadPool.h"

std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
        const std::vector<std::vector<double>> &probs_seq,
        CodeTrie& startCodeTrie,
        CodeTrie& otherCodeTrie,
        size_t beam_size,
        size_t blank_id = 0);
std::vector<std::vector<std::pair<double, Output>>> ctc_beam_search_decoder_batch(
        const std::vector<std::vector<std::vector<double>>> &probs_split,
        size_t beam_size,
        size_t num_processes,
        size_t blank_id = 0);

class DecoderState{
    int abs_time_step;
    size_t beam_size;
    size_t blank_id;
    std::vector<PathTrie*> prefixes;
    PathTrie root;
public:
    DecoderState(CodeTrie& startCodeTrie, size_t beam_size, size_t blank_id);
    ~DecoderState() = default;
    void next(const std::vector<std::vector<double>> &probs_seq, CodeTrie& otherCodeTrie);
    std::vector<std::pair<double, Output>> decode();
};
#endif //NEW_BEAM_SEARCH_CTC_BEAM_SEARCH_H
