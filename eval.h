#ifndef eval_h
#define eval_h

#include "model.h"
#include <cmath>
#include <map>
#include <vector>

// Checks if bit i is set in n.
#define IS_SET(n, i) (n & (0x1L << i))
// Drops the last n bits
#define DROP_LAST_N(val, n) (val >> n)

std::map<int, int> NODE_COUNTS;

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
float evaluate_tree_regression_treelite(float* test_input)
{
  if (std::isnan(test_input[1]) || test_input[1] < 50.4594994) {
    if (std::isnan(test_input[0]) || test_input[0] < 90.763504) {
      if (std::isnan(test_input[7]) || test_input[7] < 2.78450012) {
        return -0.0825883001;
      } else {
        return 0.063217625;
      }
    } else {
      if (std::isnan(test_input[2]) || test_input[2] < 122.399002) {
        return 0.138557851;
      } else {
        return -0.160050601;
      }
    }
  } else {
    if (std::isnan(test_input[0]) || test_input[0] >= 163.686493) {
      if (std::isnan(test_input[8]) || test_input[8] < 79.897995) {
        return -0.183263466;
      } else {
        return -0.0119630694;
      }
    } else {
      if (std::isnan(test_input[11]) || test_input[11] < 0.522500038) {
        return -0.0962326154;
      } else {
        return 0.0580137558;
      }
    }
  }
  return 0;
}

/**
 * Evaluate single test input using strategy from
 * https://www.sysml.cc/doc/89.pdf with cover statistic re-ordering.  If the
 * first bit in `child_offset` is off, then the right child offset is stored in
 * `child_offset`; otherwise, it's thethe left child offset. Whichever child is
 * not referenced by `child_offset` is adjacent to the current node. Note:
 * `child_offset` is used as a *true* offset here, i.e., it signifies the
 * relative distance between the current node and the right child---in
 * contrast, the `preorder` uses `child_offset` as an absolute index for the
 * right child
 **/
float evaluate_tree_regression_yelp_preorder_cover(std::vector<node_t>& tree, float* test_input)
{
  int curr_index = 0;
  node_t curr = tree[curr_index];
  while (curr.child_offset != 0) { // while the current node is not a leaf
    NODE_COUNTS[curr_index]++;
    const int feature_index = DROP_LAST_N(curr.feature_index, 1);
    const float value = test_input[feature_index];
    if (std::isnan(value)) {
      // missing value
      if (IS_SET(curr.feature_index, 0)) {
        // default is right branch
        if (IS_SET(curr.child_offset, 0)) {
          // `child_offset` refers to left child
          curr_index++;
        } else {
          // `child_offset` refers to right child
          curr_index += DROP_LAST_N(curr.child_offset, 1);
        }
      } else {
        // default is left branch
        if (IS_SET(curr.child_offset, 0)) {
          // `child_offset` refers to left child
          curr_index += DROP_LAST_N(curr.child_offset, 1);
        } else {
          // `child_offset` refers to right child
          curr_index++;
        }
      }
    } else if (test_input[feature_index] < curr.split_value) {
      // left branch
      if (IS_SET(curr.child_offset, 0)) {
        // `child_offset` refers to left child
        curr_index += DROP_LAST_N(curr.child_offset, 1);
      } else {
        // `child_offset` refers to right child
        curr_index++;
      }
    } else {
      // right branch
      if (IS_SET(curr.child_offset, 0)) {
        // `child_offset` refers to left child
        curr_index++;
      } else {
        // `child_offset` refers to right child
        curr_index += DROP_LAST_N(curr.child_offset, 1);
      }
    }
    curr = tree[curr_index];
  }
  NODE_COUNTS[curr_index]++; // for leaf node
  return curr.split_value;
}

/**
 * Evaluate single test input using strategy from
 * https://www.sysml.cc/doc/89.pdf, but without cover statistic re-ordering.
 * The right child offset is always stored in `child_offset`; the left child
 * offset is always adjacent to the current node
 **/
float evaluate_tree_regression_yelp_preorder(std::vector<node_t>& tree, float* test_input)
{
  int curr_index = 0;
  node_t curr = tree[curr_index];
  while (curr.child_offset != 0) { // while the current node is not a leaf
    NODE_COUNTS[curr_index]++;
    const int feature_index = DROP_LAST_N(curr.feature_index, 1);
    const float value = test_input[feature_index];
    if (std::isnan(value)) {
      // missing value
      if (IS_SET(curr.feature_index, 0)) {
        // default is right branch
        curr_index = curr.child_offset;
      } else {
        // default is left branch
        curr_index++;
      }
    } else if (test_input[feature_index] < curr.split_value) {
      // left branch
      curr_index++;
    } else {
      // right branch
      curr_index = curr.child_offset;
    }
    curr = tree[curr_index];
  }
  NODE_COUNTS[curr_index]++; // for leaf node
  return curr.split_value;
}

/**
 * Evaluate single test input using strategy from
 * https://www.sysml.cc/doc/89.pdf, but for a tree that's breadth-first ordered
 * The right child offset is always stored in `child_offset`; the left child
 * offset is always 2 * curr_index + 1 from the current node
 **/
float evaluate_tree_regression_yelp_breadth_first(std::vector<node_t>& tree, float* test_input)
{
  int curr_index = 0;
  node_t curr = tree[curr_index];
  while (curr.child_offset != 0) { // while the current node is not a leaf
    NODE_COUNTS[curr_index]++;
    const int feature_index = DROP_LAST_N(curr.feature_index, 1);
    const float value = test_input[feature_index];
    if (std::isnan(value)) {
      // missing value
      if (IS_SET(curr.feature_index, 0)) {
        // default is right branch
        curr_index = curr.child_offset;
      } else {
        // default is left branch
        curr_index = curr_index * 2 + 1;
      }
    } else if (test_input[feature_index] < curr.split_value) {
      // left branch
      curr_index = curr_index * 2 + 1;
    } else {
      // right branch
      curr_index = curr.child_offset;
    }
    curr = tree[curr_index];
  }
  NODE_COUNTS[curr_index]++; // for leaf node
  return curr.split_value;
}

#endif
