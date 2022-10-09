﻿#include "ctc_beam_search.h"

DecoderState::DecoderState(size_t beam_size, size_t blank_id)
	: abs_time_step(0), beam_size(beam_size),
	blank_id(blank_id) {
	root.score = root.log_prob_b_prev = 0.0; // 初始化得分为0.0
	prefixes.push_back(&root);
}


std::vector<std::pair<float, Output>>
get_beam_search_result(const std::vector<PathTrie*>& prefixes, size_t beam_size) {
	std::vector<PathTrie*> space_prefixes;
	if (space_prefixes.empty()) { // 将参数读进当前函数的局部变量内
		for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
			space_prefixes.push_back(prefixes[i]);
		}
	}
	std::sort(space_prefixes.begin(), space_prefixes.end(), PathTrie::prefix_compare); // 排序树的叶子节点
	std::vector<std::pair<float, Output>> output_vecs;
	for (size_t i = 0; i < beam_size && i < space_prefixes.size(); ++i) {
		std::vector<int> output;
		std::vector<int> timesteps;
		std::vector<int> CodeValue;
		space_prefixes[i]->get_path_vec(output, timesteps, CodeValue); // 获取路径
		Output outputs;
		outputs.tokens = output; // 生成output对象 存储路径
		outputs.timesteps = timesteps;  // 存储时间步
		outputs.CodeValues = CodeValue;
		std::pair<float, Output> output_pair(space_prefixes[i]->score, outputs);
		output_vecs.emplace_back(output_pair); // 装载到输出路径
	}
	return output_vecs;
}


// 生成前缀路径树
void DecoderState::next(const std::vector<std::vector<float>>& probs_seq) {
	size_t num_time_steps = probs_seq.size();
	for (size_t time_step = 0; time_step < num_time_steps; ++time_step, ++abs_time_step) {
		auto& prob = probs_seq[time_step];
		if (prob.size() > 5) break;
		for (size_t index = 0; index < prob.size(); index++) {
			for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) { // 遍历前缀列表 （有序的 从大到小）
				auto prefix = prefixes[i];
				float log_prob_c = prob[index];
				if (index == 0) {
					prefix->log_prob_b_cur = log_sum_exp(prefix->log_prob_b_cur, prob[blank_id] + prefix->score);
					continue;
				}
				if (index == prefix->character)
					prefix->log_prob_nb_cur = log_sum_exp(prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
				auto prefix_new = prefix->get_path_trie(index, abs_time_step, log_prob_c);
				auto log_p = -NUM_FLT_INF;
				if (index == prefix->character && prefix->log_prob_b_prev > -NUM_FLT_INF)
					log_p = log_prob_c + prefix->log_prob_b_prev;
				else if (index != prefix->character)
					log_p = log_prob_c + prefix->score;
				prefix_new->log_prob_nb_cur = log_sum_exp((double)prefix_new->log_prob_nb_cur, log_p);
			}
		}
		prefixes.clear(); // 清空当前vector
		root.iterate_to_vec(prefixes); // 遍历树获取当前的序列端点节点
		if (prefixes.size() >= beam_size) {
			std::nth_element(prefixes.begin(), prefixes.begin() + beam_size, prefixes.end(), PathTrie::prefix_compare); // 选出前beamsize的节点
			for (size_t i = beam_size; i < prefixes.size(); ++i) prefixes[i]->remove();//向父节点递归删除当前节点，条件子节点数为0
			prefixes.resize(beam_size);
		}
	}
}

std::vector<std::pair<float, Output>> DecoderState::decode() {
	return get_beam_search_result(prefixes, beam_size); // 获取输出结果
}

std::vector<std::pair<float, Output>>
ctc_beam_search_decoder(const std::vector<std::vector<float>>& probs_seq, size_t beam_size, size_t blank_id) {
	DecoderState state(beam_size, blank_id); // 初始化解码类
	state.next(probs_seq);
	return state.decode();
}

std::vector<std::vector<std::pair<float, Output>>>
ctc_beam_search_decoder_batch(const std::vector<std::vector<std::vector<float>>>& probs_split, size_t beam_size, size_t num_processes, size_t blank_id) {
	ThreadPool pool(num_processes); // 线程池数量初始化线程池
	size_t batch_size = probs_split.size();
	std::vector<std::future<std::vector<std::pair<float, Output>>>> res;
	for (size_t i = 0; i < batch_size; ++i)
		res.emplace_back(pool.enqueue(ctc_beam_search_decoder, std::cref(probs_split[i]), beam_size, blank_id));
	std::vector<std::vector<std::pair<float, Output>>> batch_results;
	for (size_t i = 0; i < batch_size; ++i) batch_results.emplace_back(res[i].get());
	return batch_results;
}