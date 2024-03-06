#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

class PerCharTokenizer {
private:
    int vocab_size;
    std::unordered_map<char, int> string_to_index;
    std::unordered_map<int, char> index_to_string;
    std::unordered_map<int, char> vocab;

public:
    PerCharTokenizer() {
        vocab_size = 0;
    }

    std::vector<int> encode(const std::string& str) {
        std::vector<int> encoded;
        for (char c : str) {
            if (string_to_index.find(c) != string_to_index.end()) {
                encoded.push_back(string_to_index[c]);
            } else {
                int special_index = string_to_index.size();
                string_to_index[c] = special_index;
                index_to_string[special_index] = c;
                encoded.push_back(special_index);
            }
        }
        return encoded;
    }

    std::string decode(const std::vector<int>& vec) {
        std::string decoded;
        for (int i : vec) {
            if (index_to_string.find(i) != index_to_string.end()) {
                decoded.push_back(index_to_string[i]);
            }
        }
        return decoded;
    }

    void train(const std::string& train_text) {
        std::unordered_map<char, int> char_counts;
        for (char c : train_text) {
            char_counts[c]++;
        }
        std::vector<char> unique_chars;
        for (auto& pair : char_counts) {
            unique_chars.push_back(pair.first);
        }
        std::sort(unique_chars.begin(), unique_chars.end());
        for (int i = 0; i < unique_chars.size(); ++i) {
            vocab[i] = unique_chars[i];
            string_to_index[unique_chars[i]] = i;
            index_to_string[i] = unique_chars[i];
        }
        vocab_size = unique_chars.size();
    }

    void save_model(const std::string& prefix) {
        std::string file_name = prefix + ".model";
        std::ofstream f(file_name);
        if (!f.is_open()) {
            std::cerr << "Error: Failed to open file for writing: " << file_name << std::endl;
            return;
        }
        f << "per-char v1\n";
        for (auto& pair : vocab) {
            f << pair.first << " " << pair.second << "\n";
        }
        f.close();
        std::cout << "File saved successfully!!" << std::endl;
    }

    void load(const std::string& model_path) {
        std::ifstream f(model_path);
        if (!f.is_open()) {
            std::cerr << "Error: Failed to open file for reading: " << model_path << std::endl;
            return;
        }
        std::string version;
        std::getline(f, version);
        if (version != "per-char v1") {
            std::cerr << "Error: Invalid model file format." << std::endl;
            return;
        }
        vocab.clear();
        string_to_index.clear();
        index_to_string.clear();
        std::string line;
        while (std::getline(f, line)) {
            std::string::size_type pos = line.find(' ');
            if (pos != std::string::npos) {
                int idx = std::stoi(line.substr(0, pos));
                char ch = line.substr(pos + 1)[0];
                vocab[idx] = ch;
                string_to_index[ch] = idx;
                index_to_string[idx] = ch;
            }
        }
        f.close();
        vocab_size = vocab.size();
    }
};