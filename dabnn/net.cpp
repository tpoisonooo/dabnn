// Copyright 2019 JD.com Inc. JD AI

#include "net.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <chrono>
#include <vector>

#include <common/flatbuffers_helper.h>
#include <dabnn/bitpack.h>
#include <dabnn/layers/Add.h>
#include <dabnn/layers/Affine.h>
#include <dabnn/layers/AvePool.h>
#include <dabnn/layers/BinConv.h>
#include <dabnn/layers/Binarize.h>
#include <dabnn/layers/Concat.h>
#include <dabnn/layers/FloatConv.h>
#include <dabnn/layers/MaxPool.h>
#include <dabnn/layers/Relu.h>
#include <dabnn/layers/Shuffle.h>
#include <dabnn/layers/Split.h>

using std::string;
using std::vector;

namespace bnn {
string get_bin_imm_name(string name) { return name + "_bin"; }

string get_pad_imm_name(string name) { return "pad_for_" + name + "_cal"; }

void Net::read(const std::string &path) {
    auto fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::invalid_argument("Open file error " + std::to_string(errno));
    }
    size_t fsize = static_cast<size_t>(lseek(fd, 0, SEEK_END));
    auto data = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        throw std::invalid_argument("mmap failed, errno = " +
                                    std::to_string(errno));
    }
    read_impl(data);
}

void Net::read_buf(const void *ptr) { read_impl(ptr); }

void Net::read_impl(const void *ptr) {
    const auto model = flatbnn::GetModel(ptr);
    model_ = model;
    prepare();
}

void Net::prepare() {
    BNN_ASSERT(!(strict && !run_fconv), "fconv must be run in strict mode");
    for (const auto &tensor : *model_->inputs()) {
        Shaper::Shape shape(tensor->shape()->begin(), tensor->shape()->end());
        const auto name = tensor->name()->str();

        shaper.AddShape(name, shape);
        add_mat(name, std::make_shared<Mat>(shape[1], shape[2], shape[3],
                                            bnn::DataType::Float));

        input_name_ = name;

        break;
    }

    for (const auto &tensor : *model_->initializers()) {
        if (tensor->data_type() == flatbnn::DataType::Bit) {
            // This shape is the same as that of flatbuffers
            Shaper::Shape shape(tensor->shape()->begin(),
                                tensor->shape()->end());
            const auto *data = tensor->bin_data()->Data();
            const auto len = shaper.total(shape);
            // len /8 is for uint8_t -> uint64_t
            auto buf = std::make_shared<std::vector<uint64_t>>(len / 8);
            // PNT(shape, len, buf->size(), len*sizeof(uint8_t));
            memcpy(buf->data(), data, len * sizeof(uint8_t));
            // LOG(INFO) << "a";
            if (Shaper::c(shape) % 128 != 0) {
                weight_pack_2(buf->data(), buf->size());
            }

            const auto name = tensor->name()->str();

            shaper.AddShape(name, shape);

            add_mat(name,
                    std::make_shared<Mat>(shape[0], shape[1], shape[2],
                                          shape[3], buf->data(),
                                          bnn::DataType::Bit, false));
            binary_bufs_.push_back(buf);
        } else if (tensor->data_type() == flatbnn::DataType::Float32) {
            Shaper::Shape shape(tensor->shape()->begin(),
                                tensor->shape()->end());
            const auto *data = tensor->float32_data()->Data();
            const auto name = tensor->name()->str();

            shaper.AddShape(name, shape);

            if (shape.size() == 4) {
                // conv weight
                const auto len = shape[0] * shape[1] * shape[2] * shape[3];
                auto buf = std::make_shared<std::vector<float>>(len);
                memcpy(buf->data(), data, len * sizeof(float));
                add_mat(name, std::make_shared<Mat>(
                                  shape[0], shape[1], shape[2], shape[3],
                                  const_cast<uint8_t *>(data),
                                  bnn::DataType::Float, false));
            } else if (shape.size() == 1) {
                // bias or affine weight
                auto buf = std::make_shared<std::vector<float>>(shape[0]);
                memcpy(buf->data(), data, shape[0] * sizeof(float));
                add_mat(name, std::make_shared<Mat>(shape[0], buf->data(),
                                                    DataType::Float));
                float_bufs_.push_back(buf);
            }
        }
    }

    for (const auto *layer : *model_->layers()) {
        VLOG(5) << layer_type_to_str(layer->type());
        const std::string name =
            layer->name() != nullptr ? layer->name()->str() : "";
        switch (layer->type()) {
            case flatbnn::LayerType::FpConv2D: {
                ADD_LAYER(fp_conv2d, Conv, input, strides, dilations, pads,
                          weight, bias, output);
                BNN_ASSERT(pads.size() == 2 ||
                               (pads.size() == 4 && pads[0] == pads[2] &&
                                pads[1] == pads[3]),
                           pads);
                BNN_ASSERT(strides.size() == 2 || (strides.size() == 4 &&
                                                   strides[0] == strides[2] &&
                                                   strides[1] == strides[3]),
                           strides);

                if (run_fconv) {
                    if (bias != "") {
                        layers.push_back(std::make_shared<FloatConv>(
                            get_weak(), name, input, weight, bias, output,
                            pads[0], pads[1], strides[0], strides[1], 1));
                    } else {
                        layers.push_back(std::make_shared<FloatConv>(
                            get_weak(), name, input, weight, output, pads[0],
                            pads[1], strides[0], strides[1], 1));
                    }
                }

                break;
            }
            case flatbnn::LayerType::Binarize: {
                ADD_LAYER_WITH_DATA_TYPE(binarize, Binarize, DataType::Bit,
                                         input, output);
                layers.push_back(std::make_shared<Binarize>(get_weak(), name,
                                                            input, output));
                break;
            }
            case flatbnn::LayerType::BinConv2D: {
                ADD_LAYER_WITH_DATA_TYPE(bin_conv2d, Conv, DataType::Float,
                                         input, strides, dilations, pads,
                                         weight, output);
                BNN_ASSERT(pads.size() == 2 ||
                               (pads.size() == 4 && pads[0] == pads[2] &&
                                pads[1] == pads[3]),
                           pads);
                BNN_ASSERT(strides.size() == 2 || (strides.size() == 4 &&
                                                   strides[0] == strides[2] &&
                                                   strides[1] == strides[3]),
                           strides);

                layers.push_back(std::make_shared<BinConv>(
                    get_weak(), name, input, weight, output, pads[0], pads[1],
                    strides[0], strides[1]));
                break;
            }
            case flatbnn::LayerType::Affine: {
                ADD_INPLACE_LAYER(affine, Affine, input, a, b, output);
                layers.push_back(
                    std::make_shared<Affine>(get_weak(), name, input, a, b));
                break;
            }
            case flatbnn::LayerType::Add: {
#ifdef BNN_CHECK_CONSISTENCY
                ADD_LAYER(add, Eltwise, input1, input2, output)
                layers.push_back(std::make_shared<Add>(get_weak(), name, input1,
                                                       input2, output));
#else
                ADD_INPLACE_LAYER(add, Eltwise, input1, input2, output)
                layers.push_back(
                    std::make_shared<Add>(get_weak(), name, input1, input2));
#endif
                break;
            }
            case flatbnn::LayerType::MaxPool: {
                ADD_LAYER(maxpool, Pool, input, strides, pads, kernel_shape,
                          output);

                layers.push_back(std::make_shared<MaxPool>(
                    get_weak(), name, input, output, kernel_shape[0],
                    kernel_shape[1], pads[0], pads[1], strides[0], strides[1]));
                break;
            }
            case flatbnn::LayerType::AvePool: {
                ADD_LAYER(avepool, Pool, input, strides, pads, kernel_shape,
                          output);

                layers.push_back(std::make_shared<AvePool>(
                    get_weak(), name, input, output, kernel_shape[0],
                    kernel_shape[1], pads[0], pads[1], strides[0], strides[1]));
                break;
            }
            case flatbnn::LayerType::Concat: {
                ADD_LAYER(concat, Concat, inputs, axis, output);
                BNN_ASSERT(axis == 3, "");

                layers.push_back(std::make_shared<Concat>(
                    get_weak(), name, inputs[0], inputs[1], output));
                break;
            }
            case flatbnn::LayerType::Relu: {
                ADD_INPLACE_LAYER(relu, Relu, input, output);

                layers.push_back(
                    std::make_shared<Relu>(get_weak(), name, input));
                break;
            }
            case flatbnn::LayerType::Split: {
                ADD_LAYER_MULTI_OUTPUTS(split, Split, input, outputs);
                layers.push_back(std::make_shared<Split>(
                    get_weak(), name, input, outputs[0], outputs[1]));
                break;
            }
            case flatbnn::LayerType::Shuffle: {
                ADD_INPLACE_LAYER(shuffle, Shuffle, input, output);
                layers.push_back(
                    std::make_shared<Shuffle>(get_weak(), name, input));
                break;
            }
            default: {
                throw std::runtime_error("Not supported op " +
                                         layer_type_to_str(layer->type()));
                break;
            }
        }
    }
}

void Net::run(void *input) {
    BNN_ASSERT(!(strict && !run_fconv), "fconv must be run in strict mode");
    uint64_t t = 0;

    mat_map_[input_name_]->external_memory = true;
    mat_map_[input_name_]->data = input;

    for (const auto &layer : layers) {
        VLOG(5) << layer->to_str();
        layer->forward();
    }

    VLOG(2) << "t = " << t;
    VLOG(2) << "-------";
}

std::shared_ptr<Mat> Net::get_blob(const std::string &name) {
    return mat_map_.at(name);
}

void Net::add_mat(const std::string &name, std::shared_ptr<Mat> mat) {
    mat_map_[name] = mat;
}

std::weak_ptr<Net> Net::get_weak() { return shared_from_this(); }

std::shared_ptr<Net> Net::create() {
    struct make_shared_enabler : public Net {};
    return std::make_shared<make_shared_enabler>();
}

#ifdef BNN_BENCHMARK
void Net::print_time() {
    double total_time = 0;
    for (const auto &kv : layer_time_) {
        total_time += kv.second;
    }
    for (const auto &kv : layer_time_) {
        const auto &name = kv.first;
        const auto &time = kv.second;
        const auto &percent = time / total_time * 100;
        PNT(name, time, percent);
    }
}
#endif

}  // namespace bnn
