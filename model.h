#ifndef model_h
#define model_h

#include <fstream>
#include <iostream>
#include <vector>

/**
 * Borrowed from https://github.com/Yelp/xgboost-predictor-java
 *   --------------------------------------------------------
 * 1 |        Split condition / Leaf Value (32 bits)        |
 *   --------------------------------------------------------
 * 2 | Right Child Offset (31 bits) | False is left (1 bit) | (Zero iff node is leaf)
 *   --------------------------------------------------------
 * 3 |  Feature Index (31 bits) | Default is right (1 bit)  | (Original ID iff node is leaf)
 *   --------------------------------------------------------
 */
typedef struct {
  float split_value = 0.0f;
  int child_offset = 0;
  int feature_index = 0;
} node_t;

int process_line(std::ifstream& infile, std::vector<node_t>& vec)
{
  std::string line;
  if (!std::getline(infile, line)) {
    return 0;
  }
  auto start = line.find_first_not_of(" \t");
  line = line.substr(start);

  auto colon_index = line.find_first_of(":");
  line = line.substr(colon_index + 1);

  if (line.find("leaf=") == 0) {
    // leaf node
    auto comma_index = line.find_first_of(",");
    const float split_value = std::stof(line.substr(5, comma_index - 5));
    vec.push_back({ split_value, 0, -1 });

    return vec.size() - 1;

    // auto cover_index = line.find("cover=");
    // const float cover = std::stof(line.substr(cover_index + 6));
  } else {
    auto f_index = line.find_first_of("f");
    auto lt_index = line.find_first_of("<");
    const int feature_index = std::stoi(line.substr(f_index + 1, lt_index - f_index));

    auto rb_index = line.find_first_of("]");
    const float split_value = std::stof(line.substr(lt_index + 1, rb_index - lt_index));
    line = line.substr(rb_index + 6); // strip "] yes="

    auto comma_index = line.find_first_of(",");
    auto yes_value = stoi(line.substr(0, comma_index));
    line = line.substr(comma_index + 4); // strip ",no="

    comma_index = line.find_first_of(",");
    auto no_value = stoi(line.substr(0, comma_index));
    line = line.substr(comma_index + 9); // strip ",missing="

    comma_index = line.find_first_of(",");
    auto missing_value = stoi(line.substr(0, comma_index));

    // auto cover_index = line.find("cover=");
    // const float cover = std::stof(line.substr(cover_index + 6));

    if (missing_value == yes_value) {
      vec.push_back({ split_value, 0, feature_index << 1 });
    } else if (missing_value == no_value) {
      vec.push_back({ split_value, 0, ((feature_index << 1) | 0x1) });
    }
    const int curr_index = vec.size() - 1;
    process_line(infile, vec);                               // process left subtree
    const int right_child_index = process_line(infile, vec); // process right subtree
    vec[curr_index].child_offset = right_child_index;
    return curr_index;
  }
}

std::vector<node_t> read_model_preorder(std::string filename)
{
  std::ifstream infile(filename.c_str(), std::ios_base::in);
  std::vector<node_t> vec;

  if (infile) {
    std::string line;
    std::getline(infile, line); // consume "booster[0]" line
    process_line(infile, vec);  // recursively build vector
  }
  infile.close();
  return vec;
}

std::vector<node_t> read_model_breadth_first(std::string filename)
{
  std::ifstream infile(filename.c_str(), std::ios_base::in);
  std::vector<node_t> vec;

  if (!infile) {
    return vec;
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (line.find("booster[") == 0) {
      continue;
    }
    auto start = line.find_first_not_of(" \t");
    line = line.substr(start);

    auto colon_index = line.find_first_of(":");
    const int node_index = std::stoi(line.substr(0, colon_index));
    if (node_index >= (int)vec.size()) {
      vec.resize(node_index + 1);
    }
    line = line.substr(colon_index + 1);

    if (line.find("leaf=") == 0) {
      // leaf node
      auto comma_index = line.find_first_of(",");
      const float split_value = std::stof(line.substr(5, comma_index - 5));
      vec[node_index] = { split_value, 0, -1 };

    } else {
      auto f_index = line.find_first_of("f");
      auto lt_index = line.find_first_of("<");
      const int feature_index = std::stoi(line.substr(f_index + 1, lt_index - f_index));

      auto rb_index = line.find_first_of("]");
      const float split_value = std::stof(line.substr(lt_index + 1, rb_index - lt_index));
      line = line.substr(rb_index + 6); // strip "] yes="

      auto comma_index = line.find_first_of(",");
      auto yes_value = stoi(line.substr(0, comma_index));
      line = line.substr(comma_index + 4); // strip ",no="

      comma_index = line.find_first_of(",");
      auto no_value = stoi(line.substr(0, comma_index));
      line = line.substr(comma_index + 9); // strip ",missing="

      comma_index = line.find_first_of(",");
      auto missing_value = stoi(line.substr(0, comma_index));

      if (missing_value == yes_value) {
        vec[node_index] = { split_value, node_index * 2 + 2, feature_index << 1 };
      } else if (missing_value == no_value) {
        vec[node_index] = { split_value, node_index * 2 + 2, ((feature_index << 1) | 0x1) };
      }
    }
  }
  infile.close();
  return vec;
}

/**
 * booster[0]:
 * 0:[f1<50.4594994] yes=1,no=2,missing=1,gain=540253,cover=452099.844
 * 	1:[f0<90.763504] yes=3,no=4,missing=3,gain=74883.8438,cover=240563.969
 * 		3:[f7<2.78450012] yes=7,no=8,missing=7,gain=19625.002,cover=38636.8867
 * 			7:leaf=-0.0825883001,cover=23387.2773
 * 			8:leaf=0.063217625,cover=15249.6104
 * 		4:[f2<122.399002] yes=9,no=10,missing=9,gain=67479.4688,cover=201927.078
 * 			9:leaf=0.138557851,cover=194053.016
 * 			10:leaf=-0.160050601,cover=7874.06641
 * 	2:[f0<163.686493] yes=5,no=6,missing=6,gain=72340.4062,cover=211535.859
 * 		5:[f11<0.522500038] yes=11,no=12,missing=11,gain=46015.3516,cover=116936.227
 * 			11:leaf=-0.0962326154,cover=92482.0312
 * 			12:leaf=0.0580137558,cover=24454.1953
 * 		6:[f8<79.897995] yes=13,no=14,missing=13,gain=2682.90625,cover=94599.6406
 * 			13:leaf=-0.183263466,cover=93676.1719
 * 			14:leaf=-0.0119630694,cover=923.46814
 **/
// child offset always points to the right child
std::vector<node_t> create_model_breadth_first()
{
  std::vector<node_t> arr(15);
  arr[0].split_value = 50.4594994;
  arr[0].child_offset = 2;
  arr[0].feature_index = 1 << 1;

  arr[1].split_value = 90.763504;
  arr[1].child_offset = 4;
  arr[1].feature_index = 0 << 1;

  arr[2].split_value = 163.686493;
  arr[2].child_offset = 6;
  arr[2].feature_index = 0 << 1;
  arr[2].feature_index |= 0x1L;

  arr[3].split_value = 2.78450012;
  arr[3].child_offset = 8;
  arr[3].feature_index = 7 << 1;

  arr[4].split_value = 122.399002;
  arr[4].child_offset = 10;
  arr[4].feature_index = 2 << 1;

  arr[5].split_value = 0.522500038;
  arr[5].child_offset = 12;
  arr[5].feature_index = 11 << 1;

  arr[6].split_value = 79.897995;
  arr[6].child_offset = 14;
  arr[6].feature_index = 8 << 1;

  arr[7].split_value = -0.0825883001;
  arr[7].child_offset = 0;
  arr[7].feature_index = -1;

  arr[8].split_value = 0.063217625;
  arr[8].child_offset = 0;
  arr[8].feature_index = -1;

  arr[9].split_value = 0.138557851;
  arr[9].child_offset = 0;
  arr[9].feature_index = -1;

  arr[10].split_value = -0.160050601;
  arr[10].child_offset = 0;
  arr[10].feature_index = -1;

  arr[11].split_value = -0.0962326154;
  arr[11].child_offset = 0;
  arr[11].feature_index = -1;

  arr[12].split_value = 0.0580137558;
  arr[12].child_offset = 0;
  arr[12].feature_index = -1;

  arr[13].split_value = -0.183263466;
  arr[13].child_offset = 0;
  arr[13].feature_index = -1;

  arr[14].split_value = -0.0119630694;
  arr[14].child_offset = 0;
  arr[14].feature_index = -1;
  return arr;
}

#endif
