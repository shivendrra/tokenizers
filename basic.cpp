#include <iostream>
#include <vector>
#include <unordered_map>

class BasicTokenizer {
private:
    std::vector<char> chars;
    std::string train_data;
    int vocab_size;
    std::unordered_map<char, int> string_to_index;
    std::unordered_map<int, char> index_to_string;
    std::unordered_map<std::pair<int, int>, int, pair_hash> merges;
    std::unordered_map<int, std::vector<char>> vocab;

    std::unordered_map<std::pair<int, int>, int, pair_hash> get_stats(std::vector<int>& ids) {
        std::unordered_map<std::pair<int, int>, int, pair_hash> counts;
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            std::pair<int, int> pair(ids[i], ids[i + 1]);
            counts[pair]++;
        }
        return counts;
    }

    std::vector<int> merge(std::vector<int>& ids, std::pair<int, int>& pair, int idx) {
        std::vector<int> new_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == pair.first && ids[i + 1] == pair.second) {
                new_ids.push_back(idx);
                i += 1;
            } else {
                new_ids.push_back(ids[i]);
            }
        }
        return new_ids;
    }

public:
    BasicTokenizer(std::string train_text) {
        this->chars = get_unique_chars(train_text);
        this->train_data = train_text;
        this->vocab_size = this->chars.size();
        for (size_t i = 0; i < this->chars.size(); ++i) {
            this->string_to_index[this->chars[i]] = i;
            this->index_to_string[i] = this->chars[i];
            this->vocab[i] = std::vector<char>{this->chars[i]};
        }
    }

    std::vector<int> encode(std::string& str) {
        std::vector<int> encoded;
        for (char& c : str) {
            encoded.push_back(this->string_to_index[c]);
        }
        return encoded;
    }

    std::string decode(std::vector<int>& ids) {
        std::string decoded;
        for (int& id : ids) {
            decoded += this->index_to_string[id];
        }
        return decoded;
    }

    std::pair<std::unordered_map<int, std::vector<char>>, std::unordered_map<std::pair<int, int>, int, pair_hash>> train_model(int n_merges) {
        std::vector<int> tokens = encode(this->train_data);
        std::vector<int> ids = tokens;

        std::unordered_map<std::pair<int, int>, int, pair_hash> merges;
        std::unordered_map<int, std::vector<char>> vocab = this->vocab;

        for (int i = 0; i < n_merges; ++i) {
            auto stats = get_stats(ids);
            auto it = std::max_element(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
            std::pair<int, int> pair = it->first;
            int idx = this->vocab_size + i;
            ids = merge(ids, pair, idx);
            merges[pair] = idx;

            // Update vocabulary
            vocab[idx] = vocab[pair.first];
            vocab[idx].insert(vocab[idx].end(), vocab[pair.second].begin(), vocab[pair.second].end());
        }

        this->vocab = vocab;
        this->merges = merges;

        return std::make_pair(vocab, merges);
    }

    std::vector<int> text_encode(std::string& en_text) {
        std::vector<int> tokens = encode(en_text);
        while (true) {
            auto stats = get_stats(tokens);
            auto it = std::min_element(stats.begin(), stats.end(), [this](const auto& a, const auto& b) {
                return this->merges[a.first] < this->merges[b.first];
            });
            std::pair<int, int> pair = it->first;
            if (this->merges.find(pair) == this->merges.end()) {
                break;
            }
            int idx = this->merges[pair];
            tokens = merge(tokens, pair, idx);
        }
        return tokens;
    }

    std::string text_decode(std::vector<int>& de_text) {
        std::string decoded;
        for (int& idx : de_text) {
            for (char& c : this->vocab[idx]) {
                decoded += c;
            }
        }
        return decoded;
    }
};