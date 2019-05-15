// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/bitpack.h>

#include <gtest/gtest.h>

#include <common/baseline.h>
#include <common/common_bitpack.h>
#include <common/helper.h>
#include <dabnn/mat.h>

/*
TEST(bitpack, pack_mat_128) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 256;
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;
    float a_data[ALEN];
    fill_rand_float(a_data, ALEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float);
    bnn::Mat a_binary(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);
    bnn::Mat expected(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);
    pack_mat_128(a, a_binary);

    baseline_pack_mat(a, expected);

    FORZS(i, a_binary.total(), 2) {
        // LOG(INFO) << i;
        ASSERT_EQ(bitcount(*(static_cast<uint64_t *>(a_binary) + i)) +
                      bitcount(*(static_cast<uint64_t *>(a_binary) + i + 1)),
                  bitcount(*(static_cast<uint64_t *>(expected) + i)) +
                      bitcount(*(static_cast<uint64_t *>(expected) + i + 1)));
    }
}

TEST(bitpack, pack_mat_64) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 256;
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;
    float a_data[ALEN];
    fill_rand_float(a_data, ALEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float);
    bnn::Mat a_binary(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);
    bnn::Mat expected(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);
    pack_mat_64(a, a_binary);

    baseline_pack_mat(a, expected);

    FORZS(i, a_binary.total(), 2) {
        // LOG(INFO) << i;
        ASSERT_EQ(bitcount(*(static_cast<uint64_t *>(a_binary) + i)) +
                      bitcount(*(static_cast<uint64_t *>(a_binary) + i + 1)),
                  bitcount(*(static_cast<uint64_t *>(expected) + i)) +
                      bitcount(*(static_cast<uint64_t *>(expected) + i + 1)));
    }
}

TEST(bitpack, pack_mat_fallback) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 256;
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;
    float a_data[ALEN];
    fill_rand_float(a_data, ALEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float);
    bnn::Mat a_binary(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);
    bnn::Mat expected(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);
    pack_mat_128(a, a_binary);

    pack_128_fallback(a_data, expected.data, ALEN);

    ASSERT_EQ(a_binary, expected);
}
*/

TEST(bitpack, temp) {
    float a[128];
    fill_rand_float(a, 128);
    PNT(a[0] > 0, a[1] > 0, a[2] > 0, a[3] > 0);
    uint64_t b[2] = {};
    uint64_t c[2] = {};
    pack_64_bitset(a, b);
    pack_64_bitset(a + 64, b + 1);
    pack_128_2(a, c, 128);

    PNT(binrep(a[0], 16));
    PNT(binrep(a[16], 16));
    PNT(binrep(b, 16));

    weight_pack_2(b, 2);
    PNT(binrep(b, 16));
    PNT(binrep(c, 16));
    // PNT(std::bitset<64>(*b).count() + std::bitset<64>(*(b+1)).count());
    // PNT(std::bitset<64>(*c).count() + std::bitset<64>(*(c+1)).count());

    // ASSERT_EQ(a_binary, expected);
}
