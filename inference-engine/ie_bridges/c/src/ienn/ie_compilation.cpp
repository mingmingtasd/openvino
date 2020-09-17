// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ie_compilation.h"

#include <string>
#include <utility>

#include "constants.h"
#include "utils.h"

#include "ngraph/ngraph.hpp"
namespace InferenceEngine {
using namespace ngraph;
namespace {

int32_t ConvertDims(const std::vector<uint32_t>& dimensions,
                    std::vector<size_t>& dims) {
  // IE default layout is NCHW
  dims.resize(dimensions.size());
  if (dimensions.size() == 1) {
    dims[0] = dimensions[0];
  } else if (dimensions.size() == 2) {
    dims[0] = dimensions[0];
    dims[1] = dimensions[1];
  } else if (dimensions.size() == 3) {
    // HWC -> CHW
    dims[0] = dimensions[2];
    dims[1] = dimensions[0];
    dims[2] = dimensions[1];
  } else if (dimensions.size() == 4) {
    // NHWC -> NCHW
    dims[0] = dimensions[0];
    dims[1] = dimensions[3];
    dims[2] = dimensions[1];
    dims[3] = dimensions[2];
  } else {
    std::cout << "Tensor rank " << dimensions.size() << " is not supproted"
              << std::endl;
    return error_t::BAD_DATA;
  }
  return error_t::NOT_ERROR;
}

std::shared_ptr<ngraph::Node> AddActivationByFusedCode(
    int32_t fuse_code,
    std::shared_ptr<ngraph::Node> input_node) {
  std::shared_ptr<ngraph::Node> activation_node;
  if (fuse_code == fuse_t::FUSED_RELU) {
    activation_node = std::make_shared<op::v0::Relu>(input_node->output(0));
  } else if (fuse_code == fuse_t::FUSED_RELU1) {
    activation_node =
        std::make_shared<op::v0::Clamp>(input_node->output(0), -1.0, 1.0);
  } else if (fuse_code == fuse_t::FUSED_RELU6) {
    activation_node =
        std::make_shared<op::v0::Clamp>(input_node->output(0), 0.0, 6.0);
  } else if (fuse_code == fuse_t::FUSED_NONE) {
    return input_node;
  }
  return activation_node;
}
}  // namespace

Compilation::Compilation(ModelInfoPtr model)
    : model_(model), network_(nullptr) {}

void Compilation::SetPreference(int32_t prefer) {
  preference_ = prefer;
}

int32_t Compilation::GetPreference() {
  return preference_;
}

int32_t Compilation::Compile() {
  int32_t result;
  for (size_t i = 0; i < model_->inputs.size(); ++i) {
    result = AddInput(model_->inputs[i]);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model_->operations.size(); ++i) {
    const Operation& operation = model_->operations[i];
    const int32_t type = operation.type;

    if (operation.outputs.size() != 1) {
      std::cout << "Only 1 output is supported";
      return error_t::BAD_DATA;
    }

    int32_t result = error_t::NOT_ERROR;
    if (type == operation_t::ADD || type == operation_t::MUL) {
      result = AddElementwise(operation);
    } else if (type == operation_t::CONV_2D ||
               type == operation_t::DEPTHWISE_CONV_2D ||
               type == operation_t::ATROUS_CONV_2D ||
               type == operation_t::ATROUS_DEPTHWISE_CONV_2D) {
      result = AddConvolution(operation);
    } else if (type == operation_t::AVERAGE_POOL_2D ||
               type == operation_t::MAX_POOL_2D) {
      result = AddPooling(operation);
    } else if (type == operation_t::SOFTMAX) {
      result = AddSoftmax(operation);
    } else if (type == operation_t::RESHAPE) {
      result = AddReshape(operation);
    } else if (type == operation_t::CONCATENATION) {
      result = AddConcatenation(operation);
    } else if (type == operation_t::FULLY_CONNECTED) {
      result = AddFullyConnected(operation);
    } else if (type == operation_t::RESIZE_BILINEAR_NN) {
      result = AddResizeBilinear(operation);
    } else if (type == operation_t::LOGISTIC) {
      result = AddSigmoid(operation);
    } else if (type == operation_t::ARGMAX) {
      result = AddArgmax(operation);
    } else {
      std::cout << "Operation type " << type << " is not supported."
                << std::endl;
      return error_t::BAD_DATA;
    }

    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model_->outputs.size(); ++i) {
    result = AddOutput(model_->outputs[i]);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  try {
    std::shared_ptr<ngraph::Function> ngraph_function;
    ngraph_function =
        std::make_shared<Function>(ngraph_outputs_, ngraph_inputs_);
    network_.reset(new CNNNetwork(ngraph_function));
    InputsDataMap input_info(network_->getInputsInfo());
    for (auto itr : input_info) {
      if (itr.second->getLayout() == Layout::NCHW) {
        itr.second->setLayout(Layout::NHWC);
      }
      itr.second->setPrecision(Precision::FP32);
    }

    OutputsDataMap output_info(network_->getOutputsInfo());
    for (auto itr : output_info) {
      SizeVector dims = itr.second->getTensorDesc().getDims();
      if (itr.second->getLayout() == Layout::NCHW) {
        itr.second->setLayout(Layout::NHWC);
      }
      itr.second->setPrecision(Precision::FP32);
    }
  } catch (const std::exception& ex) {
    std::cout << "[IE] exception " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddInput(uint32_t index) {
  const Operand& operand = model_->operands[index];
  SizeVector dims;
  int32_t result = ConvertDims(operand.dimensions, dims);
  if (result != error_t::NOT_ERROR) {
    return result;
  }
  try {
    // MYRIAD only supports FP16 models, but it can accept FP32 inputs. So we
    // can also set the inputs precison to FP32.
    auto input_node = std::make_shared<op::Parameter>(
        preference_ != PREFER_LOW_POWER ? element::f32 : element::f16,
        Shape(dims));
    index_op_map_[index] = input_node->output(0);
    ngraph_inputs_.push_back(input_node);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add input layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddOutput(uint32_t index) {
  if (index_op_map_.find(index) == index_op_map_.end()) {
    std::cout << "Layer for output index " << index << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto output_node = std::make_shared<op::Result>(index_op_map_[index]);
    ngraph_outputs_.push_back(output_node);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add output layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConstant(uint32_t index,
                                 std::vector<size_t> specified_dims) {
  try {
    const Operand& operand = model_->operands[index];
    SizeVector dims;
    int32_t result = ConvertDims(operand.dimensions, dims);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
    size_t rank = dims.size();
    Layout layout =
        rank == 1
            ? Layout::C
            : rank == 2 ? Layout::HW : rank == 3 ? Layout::CHW : Layout::NCHW;
    // GNA also only support FP32 representing with PREFER_ULTRA_LOW_POWER.
    bool fp32_precision = preference_ != PREFER_LOW_POWER;
    Blob::Ptr blob;
    if (fp32_precision) {
      // GNA only accepts FP32 precision, cpu/gpu use FP32 currently.
      blob = make_shared_blob<float>({Precision::FP32, dims, layout});
    } else {
      // MYRIAD only accepts FP16 precision.
      blob = make_shared_blob<int16_t>({Precision::FP16, dims, layout});
    }
    blob->allocate();
    const float* src =
        reinterpret_cast<const float*>(model_->values[index].buffer);
    std::shared_ptr<op::Constant> constant_node;
    SizeVector const_dims =
        specified_dims.size() == 0 ? dims : specified_dims;
    if (fp32_precision) {
      float* dst = blob->buffer().as<float*>();
      result = Reorder<float>(dst, src, operand.dimensions);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      // The layout of constant_node depends on the shape we set
      // e.g. 4D shape will use default NCHW
      constant_node = std::make_shared<op::Constant>(
          element::f32, Shape(const_dims), blob->buffer().as<float*>());
    } else {
      int16_t* dst = blob->buffer().as<int16_t*>();
      result = Reorder<int16_t>(dst, src, operand.dimensions);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      constant_node = std::make_shared<op::Constant>(
          element::f16, Shape(const_dims), blob->buffer().as<int16_t*>());
    }
    index_op_map_[index] = constant_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add constant " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddElementwise(const Operation& operation) {
  // Setup element-wise parameters.
  ElementWiseParams params;
  int32_t result = GetElementWiseParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  for (size_t i = 0; i < 2; ++i) {
    const uint32_t input_index = operation.inputs[i];
    if (index_op_map_.find(input_index) == index_op_map_.end()) {
      // Setup constants
      if (model_->values.find(input_index) != model_->values.end()) {
        result = AddConstant(input_index);
        if (result != error_t::NOT_ERROR) {
          return result;
        }
      } else {
        std::cout << "The layer for operand index " << input_index
                  << " is not ready";
        return error_t::BAD_DATA;
      }
    }
  }
  try {
    std::shared_ptr<ngraph::Node> elementwise_node;
    auto input0 = index_op_map_[operation.inputs[0]];
    auto input1 = index_op_map_[operation.inputs[1]];
    if (operation.type == ADD) {
      elementwise_node = std::make_shared<op::v1::Add>(input0, input1);
    } else if (operation.type == MUL) {
      elementwise_node = std::make_shared<op::v1::Multiply>(input0, input1);
    } else {
      std::cout << "Operation type " << operation.type << " is not supported";
      return error_t::BAD_DATA;
    }
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] =
        AddActivationByFusedCode(params.fuse_code, elementwise_node)->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add elementwise layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConvolution(const Operation& operation) {
  // Setup convolution params.
  ConvParams params;
  int32_t result = GetConvParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    std::vector<ptrdiff_t> padBegin{params.padding_top, params.padding_left};
    std::vector<ptrdiff_t> padEnd{params.padding_bottom, params.padding_right};
    std::vector<size_t> strides{params.stride_height, params.stride_width};
    std::vector<size_t> dilations{params.dilation_height,
                                  params.dilation_width};
    const uint32_t weights_index = operation.inputs[1];
    const uint32_t bias_index = operation.inputs[2];
    std::shared_ptr<ngraph::Node> conv_node;

    if (params.depthwise) {
      if (params.depthwise_multiplier != 1) {
        std::cout << "depthwise_multiplier " << params.depthwise_multiplier
                  << " is not supported";
        return error_t::BAD_DATA;
      }

      if (params.output_channel !=
          params.input_channel * params.depthwise_multiplier) {
        std::cout
            << "Failed assertion: outputFeatureChannels "
            << params.output_channel
            << " in IE depthwise convolution descriptor must be multiplie of "
               "inFeatureChannels "
            << params.input_channel;
        return error_t::BAD_DATA;
      }
      SizeVector weights_dims{params.depth_out, 1, 1, params.filter_height,
                              params.filter_width};
      AddConstant(weights_index, weights_dims);
      conv_node = std::make_shared<op::v1::GroupConvolution>(
          index_op_map_[input_index], index_op_map_[weights_index],
          Strides(strides), CoordinateDiff(padBegin), CoordinateDiff(padEnd),
          Strides(dilations));
    } else {
      AddConstant(weights_index);
      conv_node = std::make_shared<op::v1::Convolution>(
          index_op_map_[input_index], index_op_map_[weights_index],
          Strides(strides), CoordinateDiff(padBegin), CoordinateDiff(padEnd),
          Strides(dilations));
    }
    // add bias and fused-code
    const uint32_t output_index = operation.outputs[0];
    // Since cldnn doesn't support mixed layout
    // We reshape the bias to 4D NCHW aligned with conv_node
    AddConstant(bias_index, {1, params.bias_length, 1, 1});
    auto add_node = std::make_shared<op::v1::Add>(conv_node->output(0),
                                                  index_op_map_[bias_index]);
    index_op_map_[output_index] =
        AddActivationByFusedCode(params.fuse_code, add_node)->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add pooling layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddPooling(const Operation& operation) {
  // Setup pooling params.
  PoolingParams params;
  int32_t result = GetPoolingParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation.outputs[0];
    std::shared_ptr<ngraph::Node> pooling_node;
    if (operation.type == operation_t::MAX_POOL_2D) {
      pooling_node = std::make_shared<op::v1::MaxPool>(
          index_op_map_[input_index],
          std::vector<size_t>{params.stride_width, params.stride_height},
          Shape{params.padding_top, params.padding_left},
          Shape{params.padding_bottom, params.padding_right},
          std::vector<size_t>{params.filter_height, params.filter_width},
          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
    } else {
      pooling_node = std::make_shared<op::v1::AvgPool>(
          index_op_map_[input_index],
          std::vector<size_t>{params.stride_width, params.stride_height},
          Shape{params.padding_top, params.padding_left},
          Shape{params.padding_bottom, params.padding_right},
          std::vector<size_t>{params.filter_height, params.filter_width}, true,
          op::RoundingType::FLOOR, op::PadType::EXPLICIT);
    }
    index_op_map_[output_index] =
        AddActivationByFusedCode(params.fuse_code, pooling_node)->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add pooling layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddSoftmax(const Operation& operation) {
  // Setup softmax params.
  SoftmaxParams params;
  int32_t result = GetSoftmaxParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  // Check beta.
  if (params.beta != 1.0) {
    std::cout << "beta " << params.beta << " is not supported.";
    return error_t::BAD_DATA;
  }
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }

  const uint32_t output_index = operation.outputs[0];
  try {
    auto softmax_node = std::make_shared<op::v1::Softmax>(
        index_op_map_[input_index], params.beta);
    index_op_map_[output_index] = softmax_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add softmax node " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }

  return error_t::NOT_ERROR;
}

int32_t Compilation::AddReshape(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation.outputs[0];
    const Operand& output = model_->operands[output_index];
    SizeVector dims;
    int32_t result = ConvertDims(output.dimensions, dims);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
    auto target_shape_node = std::make_shared<op::Constant>(
        element::i64, Shape{output.dimensions.size()}, dims);
    auto reshape_node = std::make_shared<op::v1::Reshape>(
        index_op_map_[input_index], target_shape_node->output(0), true);
    index_op_map_[output_index] = reshape_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add reshape layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConcatenation(const Operation& operation) {
  // Setup concatenation params.
  ConcatParams params;
  int32_t result = GetConcatParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;

  std::vector<std::shared_ptr<ngraph::Node>> input_nodes;
  for (size_t i = 0; i < operation.inputs.size() - 1; ++i) {
    const uint32_t input_index = operation.inputs[i];
    if (index_op_map_.find(input_index) == index_op_map_.end()) {
      // Setup constants
      if (model_->values.find(input_index) != model_->values.end()) {
        result = AddConstant(input_index);
        if (result != error_t::NOT_ERROR) {
          return result;
        }
      } else {
        std::cout << "The layer for operand index " << input_index
                  << " is not ready" << std::endl;
        return error_t::BAD_DATA;
      }
    }
    input_nodes.push_back(index_op_map_[input_index].get_node_shared_ptr());
  }
  int axis = 0;
  const uint32_t rank = model_->operands[operation.inputs[0]].dimensions.size();
  if (rank == 1 || rank == 2) {
    axis = params.axis;
  } else if (rank == 3) {
    // HWC -> CHW
    if (params.axis == 0) {
      axis = 1;
    } else if (params.axis == 1) {
      axis = 2;
    } else if (params.axis == 2) {
      axis = 0;
    }
  } else if (rank == 4) {
    // NHWC -> NCHW
    if (params.axis == 0) {
      axis = 0;
    } else if (params.axis == 1) {
      axis = 2;
    } else if (params.axis == 2) {
      axis = 3;
    } else if (params.axis == 3) {
      axis = 1;
    }
  } else {
    std::cout << "rank " << rank << " is not supported.";
    return error_t::BAD_DATA;
  }

  const uint32_t output_index = operation.outputs[0];
  try {
    auto concat_node = std::make_shared<ngraph::op::Concat>(input_nodes, axis);
    index_op_map_[output_index] = concat_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add concat layer " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }

  return error_t::NOT_ERROR;
}

int32_t Compilation::AddFullyConnected(const Operation& operation) {
  // Setup fully connected params.
  FullyConnectedParams params;
  int32_t result = GetFullyConnectedParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t weights_index = operation.inputs[1];
    const uint32_t bias_index = operation.inputs[2];
    AddConstant(weights_index);
    AddConstant(bias_index);
    std::vector<int32_t> dims{params.input_batch_size, params.input_size};
    auto target_shape_node =
        std::make_shared<op::Constant>(element::i64, Shape{2}, dims);
    auto reshaped_input_node = std::make_shared<op::v1::Reshape>(
        index_op_map_[input_index], target_shape_node->output(0), true);
    // Need to transpose the weights.
    auto matmul_node = std::make_shared<op::v0::MatMul>(
        reshaped_input_node->output(0), index_op_map_[weights_index], false,
        true);
    auto add_node = std::make_shared<op::v1::Add>(matmul_node->output(0),
                                                  index_op_map_[bias_index]);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] =
        AddActivationByFusedCode(params.fuse_code, add_node)->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add fc layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddResizeBilinear(const Operation& operation) {
  // Setup resize bilinear params.
  ResizeBilinearParams params;
  int32_t result = GetResizeBilinearParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;

  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready" << std::endl;
    return error_t::BAD_DATA;
  }

  ngraph::op::InterpolateAttrs attrs;
  attrs.axes = {2, 3};
  attrs.mode = "linear";
  attrs.align_corners = params.align_corners;
  attrs.antialias = false;
  attrs.pads_begin = {0, 0, 0, 0};
  attrs.pads_end = {0, 0, 0, 0};

  auto target_shape_node = op::Constant::create<int64_t>(
      element::i64, Shape{2}, {params.new_height, params.new_width});

  try {
    const uint32_t output_index = operation.outputs[0];
    auto resize_node = std::make_shared<op::v0::Interpolate>(
        index_op_map_[input_index], target_shape_node->output(0), attrs);
    index_op_map_[output_index] = resize_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add resize bilinear layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddArgmax(const Operation& operation) {
  ArgmaxParams params;
  int32_t result = GetArgmaxParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return error_t::BAD_DATA;

  if (params.axis != 3) {
    std::cout << "Only support axis for channel.";
    return error_t::BAD_DATA;
  }

  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready" << std::endl;
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation.outputs[0];
    const auto k = op::Constant::create(element::i64, Shape{}, {1});
    auto topk_node = std::make_shared<op::v1::TopK>(
        index_op_map_[input_index], k->output(0), 1, "max", "value");
    // output(1) with top k indices for each slice along axis dimension
    index_op_map_[output_index] = topk_node->output(1);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add argmax layer " << ex.what() << std::endl;
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddSigmoid(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto sigmoid_node =
        std::make_shared<op::v0::Sigmoid>(index_op_map_[input_index]);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = sigmoid_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add sigmoid layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

}  // namespace InferenceEngine
