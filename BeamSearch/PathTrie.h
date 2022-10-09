#ifndef NEW_BEAM_SEARCH_PATHTRIE_H
#define NEW_BEAM_SEARCH_PATHTRIE_H
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <utility>
#include "Utils.h"

class PathTrie {
public:
    PathTrie();
    ~PathTrie();
    float log_prob_b_prev;  // 前缀为blank的对数概率
    float log_prob_nb_prev; // 前缀不为blank的对数概率
    float log_prob_b_cur;   // 当前为blank的对数概率
    float log_prob_nb_cur;  // 当前不为blank的对数概率
    float log_prob_c;       // 当前值的概率
    float score;            // 从根节点到当前节点的对数概率
    int character;          // 当前节点存储的字符值
    int timestep;           // 开始当前节点时的时间步
    PathTrie* parent;       //当前节点的父节点地址

    // 以下用于code128的规则校验
    // code128编码树 索引
    int value = -1;
    bool isOver = false;
    static bool prefix_compare(PathTrie* x, PathTrie* y);
    // 从当前对象生成新的路径
    PathTrie* get_path_trie(int new_char, int new_timestep, float log_prob_c, bool reset = true);
    PathTrie* get_path_vec(std::vector<int>& output, std::vector<int>& timesteps, std::vector<int>& CodeValue, int stop = -1, size_t max_steps = std::numeric_limits<size_t>::max());
    // 生成序列端点（非叶子节点，但包含叶子节点）的向量
    void iterate_to_vec(std::vector<PathTrie*>& output);
    // 当前根节点等于当前存储的值，即为空
    void remove();

private:
    int ROOT_;
    bool exists_;

    std::vector<std::pair<int, PathTrie*>> children_;
};




#endif //NEW_BEAM_SEARCH_PATHTRIE_H
