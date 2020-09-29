// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>
#include <gflags/gflags.h>

#include <inference_engine.hpp>
#include <limits>
#include <memory>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/classification_results.h>
#include <string>
#include <vector>

#include "ngraph_function_creation_sample.hpp"
#include "ngraph/ngraph.hpp"

using namespace InferenceEngine;
using namespace ngraph;

// bool ParseAndCheckCommandLine(int argc, char* argv[]) {
//     slog::info << "Parsing input parameters" << slog::endl;

//     gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
//     if (FLAGS_h) {
//         showUsage();
//         showAvailableDevices();
//         return false;
//     }

//     if (FLAGS_nt <= 0 || FLAGS_nt > 10) {
//         throw std::logic_error("Incorrect value for nt argument. It should be greater than 0 and less than 10.");
//     }

//     return true;
// }

std::shared_ptr<Function> createNgraphFunction() {
 auto a_shape = Shape{1,1,2,2};

 auto a = std::make_shared<op::Parameter>(
        element::Type_t::f32, a_shape);

 std::vector<size_t> data = {1,1,1,1};

 std::shared_ptr<Node> b = std::make_shared<op::Constant>(
         element::Type_t::f32, a_shape, data);

std::shared_ptr<Node> c =
std::make_shared<op::v1::Add>(a->output(0), b->output(0));

auto result_full = std::make_shared<op::Result>(c->output(0));

std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(
        result_full, ngraph::ParameterVector{ a }, "lenet");
return fnPtr;
}

int main(int argc, char* argv[]) {
    try {
        slog::info << "test========"<< slog::endl;
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // if (!ParseAndCheckCommandLine(argc, argv)) {
        //     return 0;
        // }

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        //--------------------------- 2. Create network using ngraph function -----------------------------------

        CNNNetwork network(createNgraphFunction());
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) {
            throw std::logic_error("Sample supports topologies only with 1 input");
        }

        auto inputInfoItem = *inputInfo.begin();
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setLayout(Layout::NCHW);

        // --------------------------- Prepare output blobs -----------------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(network.getOutputsInfo());
        std::string firstOutputName;

        for (auto& item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("Output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }

        if (outputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks with a single output");
        }

        DataPtr& output = outputInfo.begin()->second;
        auto outputName = outputInfo.begin()->first;

        
        output->setPrecision(Precision::FP32);
        // output->setLayout(Layout::NC);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = exeNetwork.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto& item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            data[0]= 1; 
            data[1]= 1; 
            data[2]= 1; 
            data[3]= 1; 
        }
        inputInfo = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr outputBlob = infer_request.GetBlob(firstOutputName);
        const float* src = outputBlob->buffer()
              .as<PrecisionTrait<Precision::FP32>::value_type*>();
              std::cout<<std::endl;
     
        std::cout<< "output_data: "<<std::endl;
        for (int i=0;i<4;i++)
        std::cout  << src[0]<<" "<<std::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    slog::info << "This sample is an API example, for performance measurements, "
                 "use the dedicated benchmark_app tool"
                << slog::endl;
    return 0;
}
