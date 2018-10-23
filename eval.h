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

int evaluate_tree_regression_treelite(node_t*, float*)
{
  // TODO
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
