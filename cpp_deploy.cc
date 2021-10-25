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

      // static_cast<float*>(A->data)[i*64 + j] = i;
      // static_cast<float*>(B->data)[i*64 + j] = 2;
      // static_cast<float*>(C->data)[i*64 + j] = 1;
    }
  }
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  std::cout << "Arrays have been set" << std::endl;
  f(A, B, C, out);

  std::cout << "Function has been run" << std::endl;
  // Print out the output
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      // ICHECK_EQ(static_cast<float*>(out->data)[i][j], i + 1.0f);
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
  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  // LOG(INFO) << "Verify load function from system lib";
  // tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  // Verify(mod_syslib, "matmal_add_sys");
}

// void DeployGraphExecutor() {
//   LOG(INFO) << "Running graph executor...";
//   // load in the library
//   DLDevice dev{kDLCPU, 0};
//   tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/test_relay_add.so");
//   // create the graph executor module
//   tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
//   tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
//   tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
//   tvm::runtime::PackedFunc run = gmod.GetFunction("run");

//   // Use the C++ API
//   tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);
//   tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);

//   for (int i = 0; i < 2; ++i) {
//     for (int j = 0; j < 2; ++j) {
//       static_cast<float*>(x->data)[i * 2 + j] = i * 2 + j;
//     }
//   }
//   // set the right input
//   set_input("x", x);
//   // run the code
//   run();
//   // get the output
//   get_output(0, y);

//   for (int i = 0; i < 2; ++i) {
//     for (int j = 0; j < 2; ++j) {
//       ICHECK_EQ(static_cast<float*>(y->data)[i * 2 + j], i * 2 + j + 1);
//     }
//   }
// }

int main(void) {
  DeploySingleOp();
  // DeployGraphExecutor();
  return 0;
}
