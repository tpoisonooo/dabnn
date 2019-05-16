// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/bitpack.h>

#include <gtest/gtest.h>

#include <common/baseline.h>
#include <common/common_bitpack.h>
#include <common/helper.h>
#include <dabnn/mat.h>

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
        const auto bc1 = 128 - bitcount(*(static_cast<uint64_t *>(a_binary) + i)) -
                            bitcount(*(static_cast<uint64_t *>(a_binary) + i + 1));
        const auto bc2 = bitcount(*(static_cast<uint64_t *>(expected) + i)) +
                      bitcount(*(static_cast<uint64_t *>(expected) + i + 1));
        if (bc1 != bc2) {
            PNT(i, bc1, bc2);
            PNT(binrep(*(static_cast<uint64_t *>(a_binary) + i), true));
            PNT(binrep(*(static_cast<uint64_t *>(a_binary) + i + 1), true));
            PNT(binrep(*(static_cast<uint64_t *>(expected) + i), true));
            PNT(binrep(*(static_cast<uint64_t *>(expected) + i + 1), true));
        }
    }
}

TEST(bitpack, bitset_bitfield) {
    float fs[64];
    fill_rand_float(fs, 64);
    uint64_t buf1, buf2;
    ::pack_64_bitfield(fs, &buf1);
    ::pack_64_bitset(fs, &buf2);
    ASSERT_EQ(buf1, buf2);
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

/*
TEST(bitpack, temp) {
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
    weight_pack_2(static_cast<uint64_t *>(expected.data), expected.total());

    FORZS(i, a_binary.total(), 1) {
        // LOG(INFO) << i;
        ASSERT_EQ(*(static_cast<uint64_t *>(a_binary) + i), *(static_cast<uint64_t *>(expected) + i));
    }
}
*/
