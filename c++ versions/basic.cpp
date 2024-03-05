#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

class BasicTokenizer {
private:
    std::string train_data;
    std::vector<char> chars;
    std::unordered_map<char, int> string_to_index;
    std::unordered_map<int, char> index_to_string;
    std::unordered_map<std::pair<char, char>, int, boost::hash<std::pair<char, char>>> merges;
    std::unordered_map<int, std::vector<char>> vocab;

    std::vector<int> encode(const std::string& text) {
        std::vector<int> encoded;
        for (char c : text) {
            encoded.push_back(string_to_index[c]);
        }
        return encoded;
    }

    std::string decode(const std::vector<int>& integer) {
        std::string decoded;
        for (int i : integer) {
            decoded.push_back(index_to_string[i]);
        }
        return decoded;
    }

    std::unordered_map<std::pair<char, char>, int, boost::hash<std::pair<char, char>>> get_stats(const std::vector<int>& ids) {
        std::unordered_map<std::pair<char, char>, int, boost::hash<std::pair<char, char>>> counts;
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            counts[{ids[i], ids[i + 1]}]++;
        }
        return counts;
    }

    std::vector<int> merge(const std::vector<int>& ids, const std::pair<char, char>& pair, int idx) {
        std::vector<int> new_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == pair.first && ids[i + 1] == pair.second) {
                new_ids.push_back(idx);
                i++;
            } else {
                new_ids.push_back(ids[i]);
            }
        }
        return new_ids;
    }

public:
    BasicTokenizer(const std::string& train_text) : train_data(train_text) {
        chars = std::vector<char>(train_text.begin(), train_text.end());
        std::sort(chars.begin(), chars.end());
        for (size_t i = 0; i < chars.size(); ++i) {
            string_to_index[chars[i]] = i;
            index_to_string[i] = chars[i];
        }
    }

    std::unordered_map<int, std::vector<char>> train_model(int n_merges) {
        std::vector<int> ids = encode(train_data);

        for (int i = 0; i < n_merges; ++i) {
            auto stats = get_stats(ids);
            auto pair = std::max_element(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->first;
            int idx = chars.size() + i;
            ids = merge(ids, pair, idx);
            merges[pair] = idx;
        }

        for (const auto& merge_pair : merges) {
            int idx = merge_pair.second;
            vocab[idx] = vocab[merge_pair.first.first];
            vocab[idx].insert(vocab[idx].end(), vocab[merge_pair.first.second].begin(), vocab[merge_pair.first.second].end());
        }

        return vocab;
    }

    std::vector<int> text_encode(const std::string& text) {
        std::vector<int> tokens = encode(text);
        while (true) {
            auto stats = get_stats(tokens);
            auto pair = std::min_element(stats.begin(), stats.end(), [this](const auto& a, const auto& b) {
                return merges[a.first] < merges[b.first];
            })->first;
            if (merges.find(pair) == merges.end()) {
                break;
            }
            int idx = merges[pair];
            tokens = merge(tokens, pair, idx);
        }
        return tokens;
    }

    std::string text_decode(const std::vector<int>& tokens) {
        std::string text;
        for (int token : tokens) {
            text += decode(vocab[token]);
        }
        return text;
    }

    void save_model(const std::string& prefix) {
        std::ofstream model_file(prefix + ".model");
        model_file << "minibpe v1\n";
        for (const auto& merge_pair : merges) {
            model_file << merge_pair.first.first << " " << merge_pair.first.second << "\n";
        }
        model_file.close();

        std::ofstream vocab_file(prefix + ".vocab");
        for (const auto& vocab_pair : vocab) {
            vocab_file << "[" << text_decode({vocab_pair.first}) << "] " << vocab_pair.first << "\n";
        }
        vocab_file.close();

        std::cout << "Model saved successfully!" << std::endl;
    }
};