# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# brief Example code on load and run TVM module.s
# file python_deploy.py

import tvm
from tvm import te
import numpy as np


def verify(mod, fname):
    # Get the function from the module
    f = mod.get_function(fname)

    dev = tvm.cpu()
    N = L = M = 64
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = np.random.uniform(size=(N, M)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)

    # Invoke the function
    f(a_tvm, b_tvm, c_tvm, out_tvm)

    # Verify correctness of function
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    print("Finish Python verification...")


if __name__ == "__main__":
    # The normal dynamic loading method for deployment
    mod_dylib = tvm.runtime.load_module("lib/test_dll.so")
    print("Verify dynamic loading from test_dll.so")
    verify(mod_dylib, "matmul_add_dyn")
    # There might be methods to use the system lib way in
    # python, but dynamic loading is good enough for now.
