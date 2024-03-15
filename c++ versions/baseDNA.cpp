#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <sstream>

class DNAtokenizer {
public:
    DNAtokenizer() {
        chars = { "\n", "A", "C", "G", "T" };
        vocab_size = chars.size();
        merges = {};
        vocab = {};
        buildVocab();
    }

    void train(const std::string& train_data, int target_vocab) {
        std::vector<int> ids = encode(train_data);

        merges.clear();
        int n_merges = target_vocab - vocab_size + 1;
        for (int i = 0; i < n_merges; ++i) {
            auto stats = getStats(ids);
            auto pair = std::max_element(stats.begin(), stats.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            int idx = vocab_size + i;
            ids = merge(ids, pair->first, idx);
            merges[pair->first] = idx;
        }

        for (const auto& [pair, idx] : merges) {
            vocab[idx] = vocab[pair.first] + vocab[pair.second];
        }

        vocab_size = vocab.size();
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> encoded;
        for (char c : text) {
            std::string char_str(1, c);
            encoded.push_back(string_to_index[char_str]);
        }
        return encoded;
    }

    std::string decode(const std::vector<int>& de_text) {
        std::string decoded;
        for (int idx : de_text) {
            decoded += index_to_string[idx];
        }
        return decoded;
    }

    void saveModel(const std::string& model_prefix) {
        std::ofstream modelFile(model_prefix + ".model");
        for (const auto& [pair, idx] : merges) {
            modelFile << pair.first << " " << pair.second << "\n";
        }
        modelFile.close();

        std::ofstream vocabFile(model_prefix + "_vocab.json");
        for (const auto& [idx, seq] : vocab) {
            vocabFile << idx << " " << seq << "\n";
        }
        vocabFile.close();

        std::cout << "Model files saved successfully!\n";
    }

    void loadModel(const std::string& model_path) {
        merges.clear();
        vocab.clear();
        vocab_size = 0;

        std::ifstream modelFile(model_path);
        std::string line;
        int idx = vocab_size;
        while (std::getline(modelFile, line)) {
            std::istringstream iss(line);
            int idx1, idx2;
            iss >> idx1 >> idx2;
            merges[{idx1, idx2}] = idx;
            ++idx;
        }
        modelFile.close();

        for (const auto& [pair, idx] : merges) {
            vocab[idx] = vocab[pair.first] + vocab[pair.second];
        }

        vocab_size = vocab.size();
    }

private:
    std::vector<std::string> chars;
    std::unordered_map<std::string, int> string_to_index;
    std::unordered_map<int, std::string> index_to_string;
    std::unordered_map<int, std::string> vocab;
    std::unordered_map<std::pair<int, int>, int> merges;
    int vocab_size;

    void buildVocab() {
        int index = 0;
        for (const auto& ch : chars) {
            string_to_index[ch] = index;
            index_to_string[index] = ch;
            vocab[index] = ch;
            ++index;
        }
    }

    std::unordered_map<std::pair<int, int>, int> getStats(const std::vector<int>& ids) {
        std::unordered_map<std::pair<int, int>, int> counts;
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            std::pair<int, int> pair(ids[i], ids[i + 1]);
            counts[pair]++;
        }
        return counts;
    }

    std::vector<int> merge(const std::vector<int>& ids, const std::pair<int, int>& pair, int idx) {
        std::vector<int> new_ids;
        for (size_t i = 0; i < ids.size();) {
            if (i + 1 < ids.size() && ids[i] == pair.first && ids[i + 1] == pair.second) {
                new_ids.push_back(idx);
                i += 2;
            }
            else {
                new_ids.push_back(ids[i]);
                ++i;
            }
        }
        return new_ids;
    }
};