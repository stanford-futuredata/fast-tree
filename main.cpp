
#include <assert.h>
#include <bitset>
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdint.h>
#include <string.h>

// Checks if bit i is set in n.
#define IS_SET(n, i) (n & (0x1L << i))
// Drops the last n bits
#define DROP_LAST_N(val, n) (val >> n)

// Ordered comparison of NaN and 1.0 gives false.
// Ordered comparison of 1.0 and 1.0 gives true.
// Ordered comparison of NaN and Nan gives false.
// Unordered comparison of NaN and 1.0 gives true.
// Unordered comparison of 1.0 and 1.0 gives false.
// Unordered comparison of NaN and NaN gives true.

// #define _CMP_EQ_OQ    0x00 /* Equal (ordered, non-signaling)  */
// #define _CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
// #define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
// #define _CMP_UNORD_Q  0x03 /* Unordered (non-signaling)  */
// #define _CMP_NEQ_UQ   0x04 /* Not-equal (unordered, non-signaling)  */
// #define _CMP_NLT_US   0x05 /* Not-less-than (unordered, signaling)  */
// #define _CMP_NLE_US   0x06 /* Not-less-than-or-equal (unordered, signaling)  */
// #define _CMP_ORD_Q    0x07 /* Ordered (nonsignaling)   */
// #define _CMP_EQ_UQ    0x08 /* Equal (unordered, non-signaling)  */
// #define _CMP_NGE_US   0x09 /* Not-greater-than-or-equal (unord, signaling)  */
// #define _CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */
// #define _CMP_FALSE_OQ 0x0b /* False (ordered, non-signaling)  */
// #define _CMP_NEQ_OQ   0x0c /* Not-equal (ordered, non-signaling)  */
// #define _CMP_GE_OS    0x0d /* Greater-than-or-equal (ordered, signaling)  */
// #define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
// #define _CMP_TRUE_UQ  0x0f /* True (unordered, non-signaling)  */
// #define _CMP_EQ_OS    0x10 /* Equal (ordered, signaling)  */
// #define _CMP_LT_OQ    0x11 /* Less-than (ordered, non-signaling)  */
// #define _CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
// #define _CMP_UNORD_S  0x13 /* Unordered (signaling)  */
// #define _CMP_NEQ_US   0x14 /* Not-equal (unordered, signaling)  */
// #define _CMP_NLT_UQ   0x15 /* Not-less-than (unordered, non-signaling)  */
// #define _CMP_NLE_UQ   0x16 /* Not-less-than-or-equal (unord, non-signaling)  */
// #define _CMP_ORD_S    0x17 /* Ordered (signaling)  */
// #define _CMP_EQ_US    0x18 /* Equal (unordered, signaling)  */
// #define _CMP_NGE_UQ   0x19 /* Not-greater-than-or-equal (unord, non-sign)  */
// #define _CMP_NGT_UQ   0x1a /* Not-greater-than (unordered, non-signaling)  */
// #define _CMP_FALSE_OS 0x1b /* False (ordered, signaling)  */
// #define _CMP_NEQ_OS   0x1c /* Not-equal (ordered, signaling)  */
// #define _CMP_GE_OQ    0x1d /* Greater-than-or-equal (ordered, non-signaling)  */
// #define _CMP_GT_OQ    0x1e /* Greater-than (ordered, non-signaling)  */
// #define _CMP_TRUE_US  0x1f /* True (unordered, signaling)  */

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

int evaluate_tree_simd(node_t* tree, float* test_input)
{

  float split_values[8];
  float test_values[8];
  for (int i = 0; i < 8; ++i) {
    split_values[i] = tree[i].split_value;
    test_values[i] = test_input[tree[i].feature_index];
  }
  __m256 split_register = _mm256_loadu_ps(split_values);
  __m256 test_register = _mm256_loadu_ps(test_values);
  uint8_t mask = _mm256_movemask_ps(_mm256_cmp_ps(split_register, test_register, _CMP_LE_OS));
  std::bitset<8> bits(mask);
  std::cout << "mask: " << bits << std::endl;
  uint8_t opp_mask = ~(mask);
  std::bitset<8> opp_bits(opp_mask);
  std::cout << "opp mask: " << opp_bits << std::endl;

  int count = 0;
  while (opp_mask) {
    int index = ffs(opp_mask) - 1;
    std::cout << "index: " << index << std::endl;
    mask &= ~(1 << index);
    count++;
  }
  std::cout << std::endl;
  std::cout << "count: " << count << std::endl;
  return count;
}

/**
 * Evaluate single test input using strategy from
 * https://www.sysml.cc/doc/89.pdf, but for tree that's not reordered based on
 * cover statistic. The right child offset is always stored in child_offset
 * field; the left child offset is always in the same relative location to the
 * parent node
 **/
float evaluate_tree_regression_yelp_no_cover(std::unique_ptr<node_t[]>& tree, float* test_input)
{
  int curr_offset = 0;
  node_t curr = tree[curr_offset];
  while (curr.child_offset != 0) { // while the current node is not a leaf
    const int feature_index = DROP_LAST_N(curr.feature_index, 1);
    const float value = test_input[feature_index];
    if (std::isnan(value)) {
      // missing value
      if (IS_SET(curr.feature_index, 0)) {
        // default is right branch
        curr_offset = curr.child_offset;
      } else {
        // default is left branch
        curr_offset = curr_offset * 2 + 1;
      }
    } else if (test_input[feature_index] < curr.split_value) {
      // left branch
      curr_offset = curr_offset * 2 + 1;
    } else {
      // right branch
      curr_offset = curr.child_offset;
    }
    curr = tree[curr_offset];
  }
  return curr.split_value;
}

int evaluate_tree_regression_treelite(node_t*, float*)
{
  // TODO
  return 0;
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
std::unique_ptr<node_t[]> create_model_no_cover()
{
  std::unique_ptr<node_t[]> arr = std::make_unique<node_t[]>(15);
  // node_t* arr = new node_t[15];
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

std::unique_ptr<float[]> read_test_data(std::string filename, int num_rows, int num_cols, const float missing_val)
{
  std::ifstream infile(filename.c_str(), std::ios_base::in);
  std::unique_ptr<float[]> values = std::make_unique<float[]>(num_rows * num_cols);

  std::string buffer;
  if (infile) {
    for (int i = 0; i < num_rows; i++) {
      float* d = &values[i * num_cols];
      for (int j = 0; j < num_cols; j++) {
        float f;
        infile >> f;
        if (f == missing_val) {
          f = std::numeric_limits<float>::quiet_NaN();
        }
        if (j != num_cols - 1) {
          std::getline(infile, buffer, ',');
        }
        d[j] = f;
      }
      std::getline(infile, buffer);
    }
  }
  infile.close();
  return values;
}

int main(int, char**)
{
  const int NUM_ROWS = 550000;
  const int NUM_COLS = 30;
  const float MISSING_VAL = -999.0;
  const std::string FILENAME = "../higgs-boson/data/test_raw.csv";

  // TODO: read model from file and organize model into correct data structure
  std::unique_ptr<node_t[]> model = create_model_no_cover();
  std::unique_ptr<float[]> test_inputs = read_test_data(FILENAME, NUM_ROWS, NUM_COLS, MISSING_VAL);

  std::ofstream predictions_outfile;
  const std::string predictions_fname = "predictions.csv";
  predictions_outfile.open(predictions_fname);
  for (int i = 0; i < NUM_ROWS * NUM_COLS; i += NUM_COLS) {
    float prediction = evaluate_tree_regression_yelp_no_cover(model, &test_inputs[i]);
    predictions_outfile << std::fixed << std::setprecision(17) << prediction << std::endl;
    if (i % 1000 == 0) {
      std::cout << "Prediction " << i / NUM_COLS << ": " << std::fixed << std::setprecision(17) << prediction << std::endl;
    }
  }
  predictions_outfile.close();
  return 0;
}
