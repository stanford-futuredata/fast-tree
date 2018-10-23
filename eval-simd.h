#ifndef eval_h
#define eval_h

#include "model.h"
#include <bitset>
#include <immintrin.h>
#include <iostream>
#include <string.h>

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

#endif
