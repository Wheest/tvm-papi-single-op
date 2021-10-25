<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


Testing PAPI for a single function
=========================
This project looks at how to deploy the TVM PAPI integration for standalone functions.
For discussion, see this thread on the [TVM Forums](https://discuss.tvm.apache.org/t/papi-counters-with-basic-matmul-relay-function/11263).

Clone this directory into your TVM root, in `apps`. 

Type the following command to run the sample code under the current folder(need to build TVM first).
```bash
./run_example.sh
```

It will test the basic C++ deployment of the `matmul_add` function, then attempt to run the standalone PAPI version.

Currently, this PAPI version does not work.

The code of the PAPI version is in `cpp_deploy_papi.cc`.  To try building it, run `make papi`.

