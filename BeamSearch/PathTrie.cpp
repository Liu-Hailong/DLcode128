#include "PathTrie.h"

PathTrie::PathTrie() {
    log_prob_b_prev = -NUM_FLT_INF;
    log_prob_nb_prev = -NUM_FLT_INF;
    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;
    log_prob_c = -NUM_FLT_INF;
    score = -NUM_FLT_INF;

    ROOT_ = -1;
    character = ROOT_;
    timestep = 0;
    exists_ = true;
    parent = nullptr;
    value = -1;
}

PathTrie::~PathTrie() {
    for (auto child : children_) {
        delete child.second;
    }
}

PathTrie* PathTrie::get_path_trie(int new_char, int new_timestep, float cur_log_prob_c, bool reset) {
    // 查找当前节点下的子节点中是否存在当前添加的字符
    // 存在 判断替换最大概率 当前
    auto child = children_.begin();
    for (child = children_.begin(); child != children_.end(); ++child) {
        if (child->first == new_char) {// 当前子路径中存在对应的字符，替换目标路径的概率
            if (child->second->log_prob_c < cur_log_prob_c) {
                child->second->log_prob_c = cur_log_prob_c;
                child->second->timestep = new_timestep;
            }
            break;
        }
    }
    if (child != children_.end()) {
        if (!child->second->exists_) {
            child->second->exists_ = true;
            child->second->log_prob_b_prev = -NUM_FLT_INF;
            child->second->log_prob_nb_prev = -NUM_FLT_INF;
            child->second->log_prob_b_cur = -NUM_FLT_INF;
            child->second->log_prob_nb_cur = -NUM_FLT_INF;
        }
        return (child->second);
    } else {
        auto* new_path = new PathTrie;
        new_path->character = new_char;
        new_path->timestep = new_timestep;
        new_path->parent = this;
        new_path->log_prob_c = cur_log_prob_c;
        children_.emplace_back(new_char, new_path);
        return new_path;
    }
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,  // 存储路径
                                 std::vector<int>& timesteps, // 存储的是当前路径内的时间步序列
                                 std::vector<int>& CodeValue,
                                 int stop,  // 终点节点的值
                                 size_t max_steps) {    // 默认搜索长度为上线int的最大值
    if (character == stop || character == ROOT_ || output.size() == max_steps) {
        std::reverse(output.begin(), output.end());
        std::reverse(timesteps.begin(), timesteps.end());
        std::reverse(CodeValue.begin(), CodeValue.end());
        return this;
    } else {
        output.push_back(character);
        timesteps.push_back(timestep);
        if (isOver) CodeValue.push_back(value);
        return parent->get_path_vec(output, timesteps, CodeValue, stop, max_steps);
    }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {
    // 如果是序列端点节点更新当前节点，并存储索引到output里面，以便于后续通过引用遍历序列端点
    if (exists_) {
        log_prob_b_prev = log_prob_b_cur;       // 将当前遍历生成的数据转移到prev里面，每一次遍历都是对cur进行相关操作，但是需要知道操作之前的数据，所以需要备份数据
        log_prob_nb_prev = log_prob_nb_cur;
        log_prob_b_cur = -NUM_FLT_INF;
        log_prob_nb_cur = -NUM_FLT_INF;
        score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev); // 更新到当前端点的得分->非blank和blank对数概率叠加
        output.push_back(this); // 将当前端点节点索引存入output vector中
    }
    // 深度优先遍历（dfs）
    for (auto child : children_) child.second->iterate_to_vec(output);
}

void PathTrie::remove() {
    exists_ = false; // 将当前节点标志为非序列端点节点
    if (children_.empty()) { // 我没有了子，自己主动与父节点断绝关系
        auto child = parent->children_.begin();
        for (child = parent->children_.begin(); child != parent->children_.end(); ++child) {
            if (child->first == character) {
                parent->children_.erase(child);
                break;
            }
        }
        // 如果父节点由于我的断绝关系导致父节点也没有了子嗣，父节点也会触发主动断绝父类关系的行为
        if (parent->children_.empty() && !parent->exists_) parent->remove();
        // 断绝完父类关系后，主动结果自己
        delete this;
    }
}

bool PathTrie::prefix_compare(PathTrie *x, PathTrie *y) {
    if (x->score == y->score) return x->character < y->character;
    return x->score > y->score;
}
