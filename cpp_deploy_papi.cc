/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/contrib/papi.h>
#include <tvm/runtime/profiling.h>


#include <cstdio>

void Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  ICHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* A;
  DLTensor* B;
  DLTensor* C;
  DLTensor* out;
  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[2] = {64, 64};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &A);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &B);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &C);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &out);
  std::cout << "Arrays have been allocated" << std::endl;
  for (int i = 0; i < shape[0]; ++i) {
    // static_cast<float*>(B->data)[i] = i;
    // static_cast<float*>(C->data)[i] = i;
    for (int j = 0; j < shape[1]; ++j) {
      static_cast<float*>(A->data)[i*shape[0] + j] = i*shape[0] + j;
      static_cast<float*>(B->data)[i*shape[0] + j] = i*shape[0] + j;
      static_cast<float*>(C->data)[i*shape[0] + j] = i*shape[0] + j;
    }
  }

  // Get the PAPI collector running
 //   * Example usage:
 // * \code{.cpp}
 // * Device cpu, gpu;
 // * Profiler prof({cpu, gpu});
 // * my_gpu_kernel(); // do a warmup iteration
 // * prof.Start();
 // * prof.StartCall("my_gpu_kernel", gpu);
 // * my_gpu_kernel();
 // * prof.StopCall();
 // * prof.StartCall("my_cpu_function", cpu);
 // * my_cpu_function();
 // * prof.StopCall();
 // * prof.Stop();
 // * std::cout << prof.Report << std::endl; // print profiling report
 // * \endcode
 tvm::Device dev = {kDLCPU, 0};
 tvm::Map<tvm::runtime::profiling::DeviceWrapper, tvm::Array<tvm::String>> metrics({
    {kDLCPU,
     {"perf::CYCLES", "perf::STALLED-CYCLES-FRONTEND", "perf::STALLED-CYCLES-BACKEND",
      "perf::INSTRUCTIONS", "perf::CACHE-MISSES"}},
    {kDLCUDA, {"cuda:::event:elapsed_cycles_sm:device=0"}}});


tvm::runtime::profiling::MetricCollector papi_collector = tvm::runtime::profiling::CreatePAPIMetricCollector(metrics);

 std::cout << "papi_collector created" << std::endl;

 tvm::runtime::profiling::Profiler prof = tvm::runtime::profiling::Profiler({dev}, {papi_collector});
 std::cout << "Profiler created" << std::endl;
 f(A, B, C, out); // warmup
 std::cout << "Warmup perfomed" << std::endl;
 prof.Start();
 prof.StartCall("matmul_add_dyn", dev);
 f(A, B, C, out);
 prof.StopCall();

 // std::cout << prof.Report << std::endl;
 // print profiling report

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  std::cout << "Arrays have been set" << std::endl;
  f(A, B, C, out);

  std::cout << "Function has been run" << std::endl;
  // Print out the output
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      if (j < shape[1])
        std::cout << static_cast<float*>(out->data)[i*64 + j] << ", ";
      else {
        std::cout << static_cast<float*>(out->data)[i*64 + j] << std::endl;
      }
    }
  }
  LOG(INFO) << "Finish verification...";
  TVMArrayFree(A);
  TVMArrayFree(B);
  TVMArrayFree(C);
  TVMArrayFree(out);
}

void DeploySingleOp() {
  // Normally we can directly
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("lib/test_dll.so");
  LOG(INFO) << "Verify dynamic loading from test_dll.so";
  Verify(mod_dylib, "matmul_add_dyn");
}


int main(void) {
  DeploySingleOp();
  return 0;
}
