#ifndef eval_simd_h
#define eval_simd_h

#include "model.h"
#include <algorithm>
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
float evaluate_tree_simd(std::vector<node_t>& tree, float *lookup_table, float* test_input)
{
  float split_values[8];
  float test_values[8];
  for (int i = 0; i < 7; ++i) {
    split_values[i] = tree[i].split_value;
    float test_value = test_input[tree[i].feature_index >> 1];
    if (std::isnan(test_value)) {
      if (i == 2)
        test_value = 999.0;
      else
        test_value = -999.0;
    }
    test_values[i] = test_value;
  }
  split_values[7] = 0.0;
  test_values[7] = 0.0;

  __m256 split_register = _mm256_loadu_ps(split_values);
  __m256 test_register = _mm256_loadu_ps(test_values);
  uint8_t mask = _mm256_movemask_ps(_mm256_cmp_ps(test_register, split_register, _CMP_LT_OS));
  return lookup_table[mask];
}

/**-0.18326346576213837
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
 * ___1_11 -> 7:-0.0825883001
 * ___0_11 -> 8:0.063217625
 *
 * __1__01 -> 9:0.138557851
 * __0__01 -> 10:-0.160050601
 *
 * _1__1_0 -> 11:-0.09623261541
 * _0__1_0 -> 12:0.0580137558
 *
 * 1___0_0 -> 13:-0.183263466
 * 0___0_0 -> 14:-0.0119630694
 **/

/*
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
  */

  // std::bitset<8> bits(mask);
  // std::cout << "mask: " << bits << std::endl;
  // uint8_t opp_mask = ~(mask);
  // std::bitset<8> opp_bits(opp_mask);
  // std::cout << "opp mask: " << opp_bits << std::endl;

#endif
