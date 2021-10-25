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
"""Script to prepare test_addone.so"""
import tvm
import numpy as np
from tvm import te
from tvm import relay
import os


def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul"
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


def prepare_test_libs(base_path, N, L, M, dtype):
    A, B, C, out = matmul_add(N, L, M, dtype)
    s = te.create_schedule(out.op)
    # Compile library as dynamic library
    f_dylib = tvm.build(s, [A, B, C, out], "llvm", name="matmul_add_dyn")
    dylib_path = os.path.join(base_path, "test_dll.so")
    f_dylib.export_library(dylib_path)

    # Compile library in system library mode
    f_syslib = tvm.build(s, [A, B, C, out], "llvm --system-lib", name="matmul_add_sys")
    syslib_path = os.path.join(base_path, "test_sys.o")
    f_syslib.save(syslib_path)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    N = L = M = 64
    prepare_test_libs(os.path.join(curr_path, "lib"), N, L, M, "float32")
    # prepare_graph_lib(os.path.join(curr_path, "lib"))
