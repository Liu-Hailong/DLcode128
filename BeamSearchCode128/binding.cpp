#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <memory>
#include "ctc_beam_search.h"
int beam_decode(at::Tensor th_probs,    // 输入的概率
                at::Tensor th_seq_lens, // 输入部分的长度
                size_t beam_size,   // beam大小
                size_t num_processes,   // 使用的线程数
                size_t blank_id,    // 空白标签
                at::Tensor th_output,
                at::Tensor th_timesteps,
                at::Tensor th_codeValues,
                at::Tensor th_scores,
                at::Tensor th_out_length)
{
    const int64_t max_time = th_probs.size(1);
    const int64_t batch_size = th_probs.size(0);
    const int64_t num_classes = th_probs.size(2);

    std::vector<std::vector<std::vector<double>>> inputs;
    auto prob_accessor = th_probs.accessor<float, 3>();
    auto seq_len_accessor = th_seq_lens.accessor<int, 1>();
    // 从原始的输入中将各个部分的序列提取出来
    for (int b=0; b < batch_size; ++b) { // batch_size * time_steps * num_classes
        // avoid a crash by ensuring that an erroneous seq_len doesn't have us try to access memory we shouldn't
        int seq_len = std::min((int)seq_len_accessor[b], (int)max_time); // 序列长度等于min(当前长度、最大时间步长)
        std::vector<std::vector<double>> temp (seq_len, std::vector<double>(num_classes));// 二维向量存储中间值[seq_len * num_classes]
        for (int t=0; t < seq_len; ++t) {
            for (int n=0; n < num_classes; ++n) {
                float val = prob_accessor[b][t][n];
                temp[t][n] = val;
            }
        }
        inputs.push_back(temp);
    }
    // Tensor -> vector

    std::vector<std::vector<std::pair<double, Output>>> batch_results =
    ctc_beam_search_decoder_batch(inputs, beam_size, num_processes, blank_id);
    auto outputs_accessor = th_output.accessor<int, 3>();
    auto timesteps_accessor =  th_timesteps.accessor<int, 3>();
    auto codeValues_accessor =  th_codeValues.accessor<int, 3>();
    auto scores_accessor =  th_scores.accessor<float, 2>();
    auto out_length_accessor =  th_out_length.accessor<int, 2>();

    // 处理输出
    for (int b = 0; b < batch_results.size(); ++b){
        std::vector<std::pair<double, Output>> results = batch_results[b];
        for (int p = 0; p < results.size();++p){
            std::pair<double, Output> n_path_result = results[p];
            Output output = n_path_result.second;
            std::vector<int> output_tokens = output.tokens; // 当前输出的词组
            std::vector<int> output_timesteps = output.timesteps; // 当前词组的时间步【某个词开始的时间步】
            std::vector<int> output_CodeValues = output.CodeValues;
            for (int t = 0; t < output_tokens.size(); ++t){
                outputs_accessor[b][p][t] =  output_tokens[t]; // fill output tokens
                timesteps_accessor[b][p][t] = output_timesteps[t];
            }
	        for(int t = 0; t < output_CodeValues.size(); ++t){
                codeValues_accessor[b][p][t] = output_CodeValues[t];
            }
            scores_accessor[b][p] = n_path_result.first;    // 当前词组的概率
            out_length_accessor[b][p] = output_tokens.size();// 当前词组的长度
        }
    }
    return 1;
}

int paddle_beam_decode(at::Tensor th_probs,
                       at::Tensor th_seq_lens,
                       size_t beam_size,
                       size_t num_processes,
                       size_t blank_id,
                       at::Tensor th_output,
                       at::Tensor th_timesteps,
                       at::Tensor th_codeValues,
                       at::Tensor th_scores,
                       at::Tensor th_out_length){

    return beam_decode(th_probs, th_seq_lens, beam_size, num_processes, blank_id,th_output, th_timesteps, th_codeValues, th_scores, th_out_length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paddle_beam_decode", &paddle_beam_decode, "paddle_beam_decode");
}
