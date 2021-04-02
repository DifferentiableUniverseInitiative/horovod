// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Modifications copyright (C) 2019 Intel Corporation
// Modifications copyright Microsoft
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <boost/python.hpp>

#include "operations.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common.h"
#include "fusion_buffer_manager.h"
#include "global_state.h"
#include "hashes.h"
#include "logging.h"
#include "message.h"
#include "ops/operation_manager.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "utils/env_parser.h"

#if HAVE_MPI
#define OMPI_SKIP_MPICXX
#include "mpi.h"
#include "mpi/mpi_context.h"
#include "mpi/mpi_controller.h"
#include "ops/mpi_operations.h"
#include "ops/adasum_mpi_operations.h"
#endif

#if HAVE_GPU
#include "ops/gpu_operations.h"
#if HAVE_MPI
#include "ops/mpi_gpu_operations.h"
#endif
#endif

#if HAVE_NCCL
#include "ops/nccl_operations.h"
#if HAVE_MPI
#include "ops/adasum_gpu_operations.h"
#endif
#endif

#if HAVE_DDL && HAVE_MPI
#include "mpi/ddl_mpi_context_manager.h"
#include "ops/ddl_operations.h"
#endif

#if HAVE_CCL
#include "ops/ccl_operations.h"
#endif

#if HAVE_GLOO
#include "gloo/gloo_controller.h"
#include "ops/gloo_operations.h"
#endif
 
#include <unistd.h>
/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * whichever hardware-optimized communication libraries are enabled.
 *
 * The primary logic of the allreduce, allgather and broadcast currently
 * support in MPI, NCCL, CUDA/ROCm, Gloo, oneCCL, DDL. The background thread
 * which facilitates controller operations is run in BackgroundThreadLoop().
 * The provided ops are:
 *      - HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all processes in the global communicator.
 *      - HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all processes in the
 *          global communicator.
 *      - HorovodBroadcast:
 *          Perform a broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * Additionally, this library provides C APIs to initialize Horovod and query
 * rank, local rank and world size.  These are used in Python directly through
 * ctypes.
 */

namespace horovod {
namespace common {

namespace {

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

#if HAVE_MPI
MPIContext mpi_context;
#endif

#if HAVE_GLOO
GlooContext gloo_context;
#endif

#if HAVE_GPU
GPUContext gpu_context;
#endif

#if HAVE_NCCL
NCCLContext nccl_context;
#endif

#if HAVE_DDL
DDLContext ddl_context;
#endif

#if HAVE_CCL
CCLContext ccl_context;
#endif

#if HAVE_SUBCOMM
//std::vector<std::unique_ptr<OperationManager>> op_manager;
std::vector<OperationManager *> op_manager;
#else
std::unique_ptr<OperationManager> op_manager;
#endif

#if HAVE_SUBCOMM

std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;
std::vector<std::shared_ptr<AllreduceOp>> adasum_ops;
std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops;

OperationManager* CreateOperationManager(HorovodGlobalState& state, int iComm) {
  // Order of these operations is very important. Operations will be checked
  // sequentially from the first to the last. The first 'Enabled' operation will
  // be executed.
/**
  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops;
  std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops;
**/

  // for(int i=0; i<state.num_nccl_streams; i++) {
  {
    LOG(DEBUG, "CreateOperationManager(), entered with iComm = " << iComm);

    allreduce_ops.resize(0);
    allgather_ops.resize(0);
    broadcast_ops.resize(0);
    adasum_ops.resize(0);
    alltoall_ops.resize(0);

#if HAVE_MPI && HAVE_GPU
    std::shared_ptr<MPIController> pMPIController = std::static_pointer_cast<MPIController>(state.controller[iComm]);
    if (pMPIController->IsEnabled()) {
#if HOROVOD_GPU_ALLREDUCE == 'M'
      allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new MPI_GPUAllreduce(&state.controller[iComm].mpi_ctx_, &gpu_context, &state)));

#elif HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
      adasum_ops.push_back(std::shared_ptr<AllreduceOp>(new AdasumGpuAllreduceOp(pMPIController->GetMPIContext(), &nccl_context, &gpu_context, &state, iComm)));

      allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new NCCLHierarchicalAllreduce(
            &nccl_context, pMPIController->GetMPIContext(), &gpu_context, &state, iComm)));

#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
      allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new DDLAllreduce(&ddl_context, &gpu_context, &state, iComm)));
#endif

#if HOROVOD_GPU_ALLGATHER == 'M'
      allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPI_GPUAllgather(pMPIController->GetMPIContext(), &gpu_context, &state, iComm)));
#endif
      allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPIHierarchicalAllgather(pMPIController->GetMPIContext(), &state)));

#if HOROVOD_GPU_ALLTOALL == 'M'
      alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
        new MPI_GPUAlltoall(pMPIController->GetMPIContext(), &gpu_context, &state)));
      LOG(DEBUG, "CreateOperationManager(), pushed MPI_GPUAlltoall into alltoall_ops");
#endif
    }
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new NCCLAllreduce(&nccl_context, &gpu_context, &state, iComm)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_BROADCAST == 'N'
      broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new NCCLBroadcast(&nccl_context, &gpu_context, &state, iComm)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLGATHER == 'N'
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
      new NCCLAllgather(&nccl_context, &gpu_context, &state, iComm)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLTOALL == 'N'
    alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
      new NCCLAlltoall(&nccl_context, &gpu_context, &state, iComm)));
    LOG(DEBUG, "CreateOperationManager(), pushed NCCLAlltoall into alltoall_ops");
#else
    LOG(DEBUG, "CreateOperationManager(), DID NOT push NCCLAlltoall into alltoall_ops");
#endif

#if HAVE_GLOO
    LOG(DEBUG, "CreateOperationManager(), HAVE_GLOO ERROR !!!!");
    if (gloo_context.IsEnabled()) {
      allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new GlooAllreduce(&gloo_context, &state)));
      allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new GlooAllgather(&gloo_context, &state)));
      broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new GlooBroadcast(&gloo_context, &state)));
      alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new GlooAlltoall(&gloo_context, &state)));
    }
#endif

#if HAVE_CCL
    LOG(DEBUG, "CreateOperationManager(), HAVE_CCL ERROR !!!!");
    if (state.cpu_operation == LibType::CCL) {
      allreduce_ops.push_back(
        std::make_shared<CCLAllreduce>(&ccl_context, &state));
      allgather_ops.push_back(
        std::make_shared<CCLAllgather>(&ccl_context, &state));
      broadcast_ops.push_back(
        std::make_shared<CCLBroadcast>(&ccl_context, &state));
      alltoall_ops.push_back(
        std::make_shared<CCLAlltoall>(&ccl_context, &state));
    }
#endif

#if HAVE_MPI
    /**
    if (pMPIController->IsEnabled()){
      adasum_ops.push_back(
        std::shared_ptr<AllreduceOp>(new AdasumMPIAllreduceOp(pMPIController->GetMPIContext(), &state)));
      allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new MPIAllreduce(pMPIController->GetMPIContext(),&state)));
      allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MPIAllgather(pMPIController->GetMPIContext(), &state)));
      broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new MPIBroadcast(pMPIController->GetMPIContext(), &state)));
      alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new MPIAlltoall(pMPIController->GetMPIContext(), &state, iComm)));
      LOG(DEBUG, "CreateOperationManager(), MPIAlltoall pushed into alltoall_ops ");
    }
    **/
#endif

  }

  std::shared_ptr<JoinOp> join_op(new JoinOp(&state));
  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  LOG(DEBUG, "CreateOperationManager(), before calling new OperationManager");
  return new OperationManager(&state.parameter_manager, allreduce_ops,
                              allgather_ops, broadcast_ops, alltoall_ops,
                              join_op, adasum_ops, error_op, iComm);
}

#else 	// HAVE_SUBCOMM

OperationManager* CreateOperationManager(HorovodGlobalState& state) {
  // Order of these operations is very important. Operations will be checked
  // sequentially from the first to the last. The first 'Enabled' operation will
  // be executed.
  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops;
  std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops;

#if HAVE_MPI && HAVE_GPU
  if (mpi_context.IsEnabled()) {
#if HOROVOD_GPU_ALLREDUCE == 'M'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new MPI_GPUAllreduce(&mpi_context, &gpu_context, &state)));

#elif HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
    adasum_ops.push_back(std::shared_ptr<AllreduceOp>(new AdasumGpuAllreduceOp(&mpi_context, &nccl_context, &gpu_context, &state)));

    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new NCCLHierarchicalAllreduce(
            &nccl_context, &mpi_context, &gpu_context, &state)));

#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new DDLAllreduce(&ddl_context, &gpu_context, &state)));
#endif

#if HOROVOD_GPU_ALLGATHER == 'M'
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPI_GPUAllgather(&mpi_context, &gpu_context, &state)));
#endif
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPIHierarchicalAllgather(&mpi_context, &state)));

#if HOROVOD_GPU_ALLTOALL == 'M'
    alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
        new MPI_GPUAlltoall(&mpi_context, &gpu_context, &state)));
#endif
  }
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new NCCLAllreduce(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_BROADCAST == 'N'
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new NCCLBroadcast(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLGATHER == 'N'
  allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
      new NCCLAllgather(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLTOALL == 'N'
  alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
      new NCCLAlltoall(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_GLOO
  if (gloo_context.IsEnabled()) {
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new GlooAllreduce(&gloo_context, &state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new GlooAllgather(&gloo_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new GlooBroadcast(&gloo_context, &state)));
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new GlooAlltoall(&gloo_context, &state)));
  }
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL) {
    allreduce_ops.push_back(
        std::make_shared<CCLAllreduce>(&ccl_context, &state));
    allgather_ops.push_back(
        std::make_shared<CCLAllgather>(&ccl_context, &state));
    broadcast_ops.push_back(
        std::make_shared<CCLBroadcast>(&ccl_context, &state));
    alltoall_ops.push_back(
        std::make_shared<CCLAlltoall>(&ccl_context, &state));
  }
#endif

#if HAVE_MPI
  if (mpi_context.IsEnabled()){
    adasum_ops.push_back(
        std::shared_ptr<AllreduceOp>(new AdasumMPIAllreduceOp(&mpi_context, &state)));
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new MPIAllreduce(&mpi_context,&state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MPIAllgather(&mpi_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new MPIBroadcast(&mpi_context, &state)));
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new MPIAlltoall(&mpi_context, &state)));
  }
#endif

  std::shared_ptr<JoinOp> join_op(new JoinOp(&state));
  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  return new OperationManager(&state.parameter_manager, allreduce_ops,
                              allgather_ops, broadcast_ops, alltoall_ops,
                              join_op, adasum_ops, error_op);
}
#endif // HAVE_SUBCOMM

// Process a Response by doing a reduction, a gather, a broadcast, or
// raising an error.
#if HAVE_SUBCOMM
void PerformOperation(Response response, HorovodGlobalState& state, int iComm) {
  LOG(DEBUG, "PerformOperation for iComm = " << iComm);
  std::vector<TensorTableEntry> entries;
  auto& timeline = state.timeline;
  if (response.response_type() != Response::JOIN) {
    state.tensor_queue.GetTensorEntriesFromResponse(response, entries,
                                                             state.joined[iComm]);

    for (auto& e : entries) {
      timeline.Start(e.tensor_name, response.response_type());
    }

    state.current_nccl_stream = iComm;

    if (entries.size() > 1) {
      LOG(DEBUG, "PerformOperation, entries.size = " << entries.size());
      auto first_entry = entries[0];
      // Note: it is OK for different entries to come from different frameworks
      // since buffer allocated here is guaranteed to survive at least till the
      // end of this operation.
      Status status = state.fusion_buffer.InitializeBuffer(
          state.controller[iComm]->TensorFusionThresholdBytes(),
          first_entry.device, first_entry.context,
          state.current_nccl_stream,
          [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
          [&]() { timeline.ActivityEndAll(entries); });
      if (!status.ok()) {
        LOG(DEBUG, state.controller[iComm]->GetRank()) << " : InitializeBuffer Failed";
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          // Callback can be null if the rank sent Join request.
          if (e.callback != nullptr) {
            e.callback(status);
          }
        }
        return;
      }
    }

    // On GPU data readiness is signalled by ready_event.
    std::vector<TensorTableEntry> waiting_tensors;
    for (auto& e : entries) {
      if (e.ready_event != nullptr) {
        timeline.ActivityStart(e.tensor_name, WAIT_FOR_DATA);
        waiting_tensors.push_back(e);
      }
    }
    while (!waiting_tensors.empty()) {
      for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
        if (it->ready_event->Ready()) {
          timeline.ActivityEnd(it->tensor_name);
          timeline.ActivityStart(it->tensor_name, WAIT_FOR_OTHER_TENSOR_DATA);
          it = waiting_tensors.erase(it);
        } else {
          ++it;
        }
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
    for (auto& e : entries) {
      if (e.ready_event != nullptr) {
        timeline.ActivityEnd(e.tensor_name);
      }
    }
  }

  Status status;
  LOG(DEBUG, "PerformOperation, before ExecuteOperation.");
  try {
    status = op_manager[iComm]->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    LOG(DEBUG, state.controller[iComm]->GetRank()) << "ExecuteOperation Failed";
    status = Status::UnknownError(ex.what());
  }
  LOG(DEBUG, "PerformOperation, after ExecuteOperation.");

  if (!status.in_progress()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, status.ok() ? e.output : nullptr);
      // Callback can be null if the rank sent Join request.
      if (e.callback != nullptr) {
        e.callback(status);
      }
    }
  }
  LOG(DEBUG, "PerformOperation, after check in_progress");
}

#else

void PerformOperation(Response response, HorovodGlobalState& state) {
  std::vector<TensorTableEntry> entries;
  auto& timeline = horovod_global.timeline;
  if (response.response_type() != Response::JOIN) {
    horovod_global.tensor_queue.GetTensorEntriesFromResponse(response, entries,
                                                             state.joined);

    for (auto& e : entries) {
      timeline.Start(e.tensor_name, response.response_type());
    }

    if (entries.size() > 1) {
      auto first_entry = entries[0];
      // Note: it is OK for different entries to come from different frameworks
      // since buffer allocated here is guaranteed to survive at least till the
      // end of this operation.
      Status status = horovod_global.fusion_buffer.InitializeBuffer(
          horovod_global.controller->TensorFusionThresholdBytes(),
          first_entry.device, first_entry.context,
          horovod_global.current_nccl_stream,
          [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
          [&]() { timeline.ActivityEndAll(entries); });
      if (!status.ok()) {
        LOG(DEBUG, horovod_global.controller->GetRank()) << "InitializeBuffer Failed";
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          // Callback can be null if the rank sent Join request.
          if (e.callback != nullptr) {
            e.callback(status);
          }
        }
        return;
      }
    }

    // On GPU data readiness is signalled by ready_event.
    std::vector<TensorTableEntry> waiting_tensors;
    for (auto& e : entries) {
      if (e.ready_event != nullptr) {
        timeline.ActivityStart(e.tensor_name, WAIT_FOR_DATA);
        waiting_tensors.push_back(e);
      }
    }
    while (!waiting_tensors.empty()) {
      for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
        if (it->ready_event->Ready()) {
          timeline.ActivityEnd(it->tensor_name);
          timeline.ActivityStart(it->tensor_name, WAIT_FOR_OTHER_TENSOR_DATA);
          it = waiting_tensors.erase(it);
        } else {
          ++it;
        }
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
    for (auto& e : entries) {
      if (e.ready_event != nullptr) {
        timeline.ActivityEnd(e.tensor_name);
      }
    }[0]
  }

  Status status;
  try {
    status = op_manager->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    LOG(DEBUG, horovod_global.controller->GetRank()) << "ExecuteOperation Failed";
    status = Status::UnknownError(ex.what());
  }

  if (!status.in_progress()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, status.ok() ? e.output : nullptr);
      // Callback can be null if the rank sent Join request.
      if (e.callback != nullptr) {
        e.callback(status);
      }
    }
  }
}
#endif

// The background thread loop coordinates all the controller processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for
//      dealing with MPI.
//      2. We want to gracefully handle errors, when all processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires every process to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never
//      make progress if we have a thread pool limit.
bool RunLoopOnce(HorovodGlobalState& state);

#if HAVE_SUBCOMM

void BackgroundThreadLoop(HorovodGlobalState& state) {
LOG(INFO, "BackgroundThreadLoop(), Start.");
#if HAVE_CCL
  // Initialize ccl context
  if (state.cpu_operation == LibType::CCL) {
    ccl_context.Initialize();
  }
#endif

#if HAVE_DDL
  // If DDL is enabled, let DDL ops manage MPI environment.
  auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, gpu_context);
#else
  // Otherwise, let MPI ops be in charge.
  auto mpi_ctx_manager = MPIContextManager();
#endif

LOG(INFO, "BackgroundThreadLoop(), after creating MPIContextManager()");

#if HAVE_GLOO
#if HAVE_MPI
    if (mpi_context.IsEnabled()) {
      // Initialize gloo context if mpi context is available
      gloo_context.InitializeFromMPI(mpi_context, ParseGlooIface());
    }
    else
#endif
    {
      gloo_context.Initialize(ParseGlooIface());
    }
#endif

LOG(INFO, "BackgroundThreadLoop(), before state.num_nccl_streams.");

#if HAVE_GPU
  // Set number of GPU streams to use
  auto horovod_num_nccl_streams =
      std::getenv(HOROVOD_NUM_NCCL_STREAMS);
  if (horovod_num_nccl_streams != nullptr &&
      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
  }
  LOG(INFO, "BackgroundThreadLoop(), state.num_nccl_streams from getenv = ") << state.num_nccl_streams;

#if HAVE_NCCL
  nccl_context.nccl_comms.resize(state.num_nccl_streams);
#endif
  gpu_context.streams.resize(state.num_nccl_streams);

  // Create finalizer thread pool (one thread per stream)
  gpu_context.finalizer_thread_pool.create(state.num_nccl_streams);
#endif

  auto timeline_env = std::getenv(HOROVOD_TIMELINE);
  auto horovod_timeline = timeline_env != nullptr ? std::string(timeline_env) : std::string("");

  LOG(INFO, "BackgroundThreadLoop(), before parameter_manager.");

  // Override Tensor Fusion threshold, if it's set.
  state.parameter_manager.SetTensorFusionThresholdBytes(128 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.parameter_manager.SetCycleTimeMs(1);
  auto horovod_cycle_time = std::getenv(HOROVOD_CYCLE_TIME);
  if (horovod_cycle_time != nullptr) {
    state.parameter_manager.SetCycleTimeMs(
        std::strtof(horovod_cycle_time, nullptr), true);
  }

  // Override response cache capacity, if it's set.
  state.parameter_manager.SetCacheEnabled(true);
  auto horovod_cache_capacity = std::getenv(HOROVOD_CACHE_CAPACITY);
  if (horovod_cache_capacity != nullptr) {
    uint32_t cache_capacity = std::strtol(horovod_cache_capacity, nullptr, 10);
    state.cache_capacity = cache_capacity;
    state.parameter_manager.SetCacheEnabled(cache_capacity > 0, true);
  }
  for(int n=0; n < state.num_nccl_streams; n++) {
    LOG(DEBUG, "BackgroundThreadLoop(), for n = " << n << ", state.cache_capacity = " << state.cache_capacity << 
	    ", parameter_manager.CacheEnabled = " << (int)state.parameter_manager.CacheEnabled() );

    state.response_cache[n].set_capacity(
      (int)state.parameter_manager.CacheEnabled() * state.cache_capacity);
    LOG(DEBUG, "BackgroundThreadLoop(), for n = " << n << ", state.response_cache[n].capacity = " << state.response_cache[n].capacity() );

    state.controller[n]->SetResponseCache(state.response_cache[n]);
    LOG(DEBUG, "BackgroundThreadLoop(), for n = " << n << ", state.controller[n].response_cache_.capacity = " << state.controller[n]->GetResponseCache().capacity() );

    state.controller[n]->GetResponseCache().set_capacity((int)state.parameter_manager.CacheEnabled() * state.cache_capacity);
    LOG(DEBUG, "BackgroundThreadLoop(), for n = " << n << ", Directly : state.controller[n].response_cache_.capacity = " << state.controller[n]->GetResponseCache().capacity() );
  }

  // Set flag to control use of batched memcopy kernel on GPU
  auto horovod_batch_d2d_memcopies =
      std::getenv(HOROVOD_BATCH_D2D_MEMCOPIES);
  if (horovod_batch_d2d_memcopies != nullptr &&
      std::strtol(horovod_batch_d2d_memcopies, nullptr, 10) == 0) {
    state.batch_d2d_memcopies = false;
  }

  // Check if group fusion should be disabled
  SetBoolFromEnv(HOROVOD_DISABLE_GROUP_FUSION, state.disable_group_fusion, true);

#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
  // Hierarchical allreduce is not supported without NCCL or DDL
  state.parameter_manager.SetHierarchicalAllreduce(false, true);
#endif

  // Creation of a Controller object per sub-communicator
  if (state.control_operation == LibType::MPI){

    LOG(INFO, "BackgroundThreadLoop(), initialization loop on num_nccl_streams start");
    for(int i=0; i<state.num_nccl_streams ; i++) {

      std::vector<int> v = state.controller[i]->GetRanks();
      LOG(INFO, "BackgroundThreadLoop(), stream no = " << i );
      LOG(INFO, "BackgroundThreadLoop(), ranks : " );
      for (std::vector<int>::const_iterator iv = v.begin(); iv != v.end(); ++iv)
        LOG(INFO, "BackgroundThreadLoop(), ranks : " << *iv );

      if (std::find(v.begin(), v.end(), state.controller[0]->GetRank()) == v.end()) {
        LOG(INFO, "BackgroundThreadLoop(), This rank was not found in this Communicator : " << state.controller[0]->GetRank());
	continue;
      }

      else {
        // For i = 0, the actions below were done by InitializeHorovodOnce()
        if(i > 0) {
          std::shared_ptr<MPIController> pMPIController = std::static_pointer_cast<MPIController>(state.controller[i]);
          //pMPIController->SetIndex(i);
          state.controller[i]->SetIndex(i);
          LOG(INFO, "BackgroundThreadLoop(), This rank : " << state.controller[0]->GetRank() << " was found in this Communicator : " << i);
  
          pMPIController->Enable();
        
          pMPIController->Initialize(state.controller[i]->GetRanks(), mpi_ctx_manager);
          LOG(INFO, "BackgroundThreadLoop(), mpi_ctx_ initialized for MPI_controller : " << i);
  
          // Initialize controller
          state.controller[i]->Initialize();
          LOG(INFO, "BackgroundThreadLoop(), Controller[] initialized() for : " << i);
        }
  
        bool is_coordinator = state.controller[i]->IsCoordinator();
        bool is_homogeneous = state.controller[i]->IsHomogeneous();
        int size = state.controller[i]->GetSize();
        int local_size = state.controller[i]->GetLocalSize();
        int local_rank = state.controller[i]->GetLocalRank();
  	
        // Set background thread affinity
        parse_and_set_affinity(std::getenv(HOROVOD_THREAD_AFFINITY), local_size, local_rank);
  
        // Open the timeline file on coordinator.
        bool should_enable_timeline = false;
        if (is_coordinator) {
           state.timeline.Initialize(horovod_timeline,
                                static_cast<unsigned int>(size));
        }
  
        should_enable_timeline = true;
        if (horovod_timeline == "DISABLED") {
            should_enable_timeline = false;
        }
  
        state.controller[i]->SetTimelineEnabled(should_enable_timeline);
  
        ParseStallInspectorFromEnv(state.controller[i]->GetStallInspector());
        bool mark_cycles = false;
        SetBoolFromEnv(HOROVOD_TIMELINE_MARK_CYCLES, mark_cycles,
                 true);
        state.controller[i]->SetMarkCyclesInTimelinePending(mark_cycles);
        state.mark_cycles_in_timeline = mark_cycles;
  
  
        // Set flag for hierarchical allgather. Ignore if Horovod is running on a
        // single node.
        auto horovod_hierarchical_allgather =
            std::getenv(HOROVOD_HIERARCHICAL_ALLGATHER);
        state.parameter_manager.SetHierarchicalAllgather(false);
        if (horovod_hierarchical_allgather != nullptr) {
          bool value = std::strtol(horovod_hierarchical_allgather, nullptr, 10) > 0 &&
                   (size != local_size);
          state.parameter_manager.SetHierarchicalAllgather(value, true);
        }
  
        // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
        // single node.
        auto horovod_hierarchical_allreduce =
            std::getenv(HOROVOD_HIERARCHICAL_ALLREDUCE);
        state.parameter_manager.SetHierarchicalAllreduce(false);
        if (horovod_hierarchical_allreduce != nullptr) {
          bool value = std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
                   (size != local_size);
          state.parameter_manager.SetHierarchicalAllreduce(value, true);
        }
  
        // Issue warning if hierarchical allreduce is enabled in heterogeneous cluster
        if (is_coordinator &&
              (state.parameter_manager.HierarchicalAllreduce() ||
               state.parameter_manager.HierarchicalAllgather()) &&
              !is_homogeneous) {
          std::cerr
                << "WARNING: Using different number of ranks per node might cause "
                   "performance loss in hierarchical allgather and "
                   "hierarchical allreduce. Consider assigning the same "
                   "number of ranks to each node, or disabling hierarchical "
                   "allgather and hierarchical allreduce.";
        }
  
        // Enable auto-tuning.
        auto horovod_autotune = std::getenv(HOROVOD_AUTOTUNE);
        if (horovod_autotune != nullptr &&
            std::strtol(horovod_autotune, nullptr, 10) > 0) {
          auto horovod_autotune_log = std::getenv(HOROVOD_AUTOTUNE_LOG);
          state.parameter_manager.Initialize(state.controller[i]->GetRank(), RANK_ZERO,
                                         horovod_autotune_log != nullptr
                                             ? std::string(horovod_autotune_log)
                                             : "");
          state.parameter_manager.SetAutoTuning(true);
        }
      }
    }
    LOG(INFO, "BackgroundThreadLoop(), initialization loop on num_nccl_streams end");
  }

  // Set chunk size for MPI based Adasum allreduce algorithms
  auto horovod_adasum_mpi_chunk_size = std::getenv(HOROVOD_ADASUM_MPI_CHUNK_SIZE);
  if (horovod_adasum_mpi_chunk_size != nullptr) {
    state.adasum_mpi_chunk_size = std::strtol(horovod_adasum_mpi_chunk_size, nullptr, 10);
  }

  for(int i=0 ; i<state.num_nccl_streams; i++) {
    LOG(INFO, "BackgroundThreadLoop(), calling CreateOperationManager for controller : " << i);
    //std::unique_ptr<OperationManager> dummy_op_manager;
    //op_manager.push_back(dummy_op_manager);
    op_manager.push_back(CreateOperationManager(state, i));
    LOG(INFO, "BackgroundThreadLoop(), returned from CreateOperationManager for controller : " << i);

  }

  // Signal that initialization is completed.
  state.initialization_done = true;
  LOG(INFO, state.controller[0]->GetRank() << " : Horovod Initialized for this rank");

  // Iterate until shutdown.
  try {
    sleep(5);
    while (RunLoopOnce(state)) {
      LOG(INFO, "BackgroundThreadLoop(), RunLoopOnce() done, while loop");
    }
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Horovod background loop uncaught exception: " << ex.what();
  }

    // Finalize all contexts
#if HAVE_NCCL
  nccl_context.ShutDown();
#endif

#if HAVE_GLOO
  gloo_context.Finalize();
#endif

  LOG(DEBUG, state.controller[0]->GetRank()) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // Notify all outstanding operations that Horovod has been shut down
  // and finalize tensor queue.
  std::vector<StatusCallback> callbacks;
  for(int j=0; j<state.num_nccl_streams; j++)
    state.tensor_queue.FinalizeTensorQueue(callbacks);
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }

#if HAVE_GPU
  gpu_context.Finalize();
#endif

#if HAVE_MPI
  mpi_context.Finalize(mpi_ctx_manager);
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL){
    ccl_context.Finalize();
  }
#endif

}

#else 	// NOT HAVE_SUBCOMM

void BackgroundThreadLoop(HorovodGlobalState& state) {
#if HAVE_CCL
  // Initialize ccl context
  if (state.cpu_operation == LibType::CCL) {
    ccl_context.Initialize();
  }
#endif

#if HAVE_MPI
  // Initialize mpi context
#if HAVE_DDL
  // If DDL is enabled, let DDL ops manage MPI environment.
  auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, gpu_context);
#else
  // Otherwise, let MPI ops be in charge.
  auto mpi_ctx_manager = MPIContextManager();
#endif
  mpi_context.Initialize(state.controller->GetRanks(), mpi_ctx_manager);
#endif

#if HAVE_GLOO
#if HAVE_MPI
    if (mpi_context.IsEnabled()) {
      // Initialize gloo context if mpi context is available
      gloo_context.InitializeFromMPI(mpi_context, ParseGlooIface());
    }
    else
#endif
    {
      gloo_context.Initialize(ParseGlooIface());
    }
#endif
  // Initialize controller
  state.controller->Initialize();

  bool is_coordinator = state.controller->IsCoordinator();
  bool is_homogeneous = state.controller->IsHomogeneous();
  int size = state.controller->GetSize();
  int local_size = state.controller->GetLocalSize();
  int local_rank = state.controller->GetLocalRank();

  // Set background thread affinity
  parse_and_set_affinity(std::getenv(HOROVOD_THREAD_AFFINITY), local_size, local_rank);

#if HAVE_GPU
#if HAVE_SUBCOMM
#else
  // Set number of GPU streams to use
  auto horovod_num_nccl_streams =
      std::getenv(HOROVOD_NUM_NCCL_STREAMS);
  if (horovod_num_nccl_streams != nullptr &&
      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
  }
#endif

#if HAVE_NCCL
  nccl_context.nccl_comms.resize(state.num_nccl_streams);
#endif
  gpu_context.streams.resize(state.num_nccl_streams);

  // Create finalizer thread pool (one thread per stream)
  gpu_context.finalizer_thread_pool.create(state.num_nccl_streams);
#endif

  // Open the timeline file on coordinator.
  auto timeline_env = std::getenv(HOROVOD_TIMELINE);
  auto horovod_timeline = timeline_env != nullptr ? std::string(timeline_env) : std::string("");
  bool should_enable_timeline = false;
  if (is_coordinator) {
    state.timeline.Initialize(horovod_timeline,
                              static_cast<unsigned int>(size));
  }

  should_enable_timeline = true;
  if (horovod_timeline == "DISABLED") {
    should_enable_timeline = false;
  }

  state.controller->SetTimelineEnabled(should_enable_timeline);

  ParseStallInspectorFromEnv(state.controller->GetStallInspector());
  bool mark_cycles = false;
  SetBoolFromEnv(HOROVOD_TIMELINE_MARK_CYCLES, mark_cycles,
                 true);
  state.controller->SetMarkCyclesInTimelinePending(mark_cycles);
  state.mark_cycles_in_timeline = mark_cycles;

  // Override Tensor Fusion threshold, if it's set.
  state.parameter_manager.SetTensorFusionThresholdBytes(128 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.parameter_manager.SetCycleTimeMs(1);
  auto horovod_cycle_time = std::getenv(HOROVOD_CYCLE_TIME);
  if (horovod_cycle_time != nullptr) {
    state.parameter_manager.SetCycleTimeMs(
        std::strtof(horovod_cycle_time, nullptr), true);
  }

  // Override response cache capacity, if it's set.
  state.parameter_manager.SetCacheEnabled(true);
  auto horovod_cache_capacity = std::getenv(HOROVOD_CACHE_CAPACITY);
  if (horovod_cache_capacity != nullptr) {
    uint32_t cache_capacity = std::strtol(horovod_cache_capacity, nullptr, 10);
    state.cache_capacity = cache_capacity;
    state.parameter_manager.SetCacheEnabled(cache_capacity > 0, true);
  }
  state.response_cache.set_capacity(
      (int)state.parameter_manager.CacheEnabled() * state.cache_capacity);

  // Set flag for hierarchical allgather. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allgather =
      std::getenv(HOROVOD_HIERARCHICAL_ALLGATHER);
  state.parameter_manager.SetHierarchicalAllgather(false);
  if (horovod_hierarchical_allgather != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allgather, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetHierarchicalAllgather(value, true);
  }
  // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allreduce =
      std::getenv(HOROVOD_HIERARCHICAL_ALLREDUCE);
  state.parameter_manager.SetHierarchicalAllreduce(false);
  if (horovod_hierarchical_allreduce != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetHierarchicalAllreduce(value, true);
  }

#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
  // Hierarchical allreduce is not supported without NCCL or DDL
  state.parameter_manager.SetHierarchicalAllreduce(false, true);
#endif

  // Issue warning if hierarchical allreduce is enabled in heterogeneous cluster
  if (is_coordinator &&
      (state.parameter_manager.HierarchicalAllreduce() ||
       state.parameter_manager.HierarchicalAllgather()) &&
      !is_homogeneous) {
    std::cerr
        << "WARNING: Using different number of ranks per node might cause "
           "performance loss in hierarchical allgather and "
           "hierarchical allreduce. Consider assigning the same "
           "number of ranks to each node, or disabling hierarchical "
           "allgather and hierarchical allreduce.";
  }

  // Set flag to control use of batched memcopy kernel on GPU
  auto horovod_batch_d2d_memcopies =
      std::getenv(HOROVOD_BATCH_D2D_MEMCOPIES);
  if (horovod_batch_d2d_memcopies != nullptr &&
      std::strtol(horovod_batch_d2d_memcopies, nullptr, 10) == 0) {
    state.batch_d2d_memcopies = false;
  }

  // Check if group fusion should be disabled
  SetBoolFromEnv(HOROVOD_DISABLE_GROUP_FUSION, state.disable_group_fusion, true);

  // Enable auto-tuning.
  auto horovod_autotune = std::getenv(HOROVOD_AUTOTUNE);
  if (horovod_autotune != nullptr &&
      std::strtol(horovod_autotune, nullptr, 10) > 0) {
    auto horovod_autotune_log = std::getenv(HOROVOD_AUTOTUNE_LOG);
    state.parameter_manager.Initialize(state.controller->GetRank(), RANK_ZERO,
                                       horovod_autotune_log != nullptr
                                           ? std::string(horovod_autotune_log)
                                           : "");
    state.parameter_manager.SetAutoTuning(true);
  }

  // Set chunk size for MPI based Adasum allreduce algorithms
  auto horovod_adasum_mpi_chunk_size = std::getenv(HOROVOD_ADASUM_MPI_CHUNK_SIZE);
  if (horovod_adasum_mpi_chunk_size != nullptr) {
    state.adasum_mpi_chunk_size = std::strtol(horovod_adasum_mpi_chunk_size, nullptr, 10);
  }

  op_manager.reset(CreateOperationManager(state));

  // Signal that initialization is completed.
  state.initialization_done = true;
  LOG(INFO, horovod_global.controller->GetRank()) << "Horovod Initialized";

  // Iterate until shutdown.
  try {
    while (RunLoopOnce(state));
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Horovod background loop uncaught exception: " << ex.what();
  }

    // Finalize all contexts
#if HAVE_NCCL
  nccl_context.ShutDown();
#endif

#if HAVE_GLOO
  gloo_context.Finalize();
#endif

  LOG(DEBUG, horovod_global.controller->GetRank()) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // Notify all outstanding operations that Horovod has been shut down
  // and finalize tensor queue.
  std::vector<StatusCallback> callbacks;
  horovod_global.tensor_queue.FinalizeTensorQueue(callbacks);
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }

#if HAVE_GPU
  gpu_context.Finalize();
#endif

#if HAVE_MPI
  mpi_context.Finalize(mpi_ctx_manager);
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL){
    ccl_context.Finalize();
  }
#endif

}

#endif

bool RunLoopOnce(HorovodGlobalState& state) {
  LOG(DEBUG, "RunLoopOnce(), entered, state.num_nccl_streams = " << state.num_nccl_streams);
  // This delay determines thread frequency and communication message latency
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = state.last_cycle_start +
                        std::chrono::microseconds(long(
                            state.parameter_manager.CycleTimeMs() * 1000.)) -
                        start_time;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

#if HAVE_SUBCOMM

  bool answer = true;

  for(int i=0; i<state.num_nccl_streams; i++) {
    LOG(DEBUG, "RunLoopOnce(), in loop for nccl_stream = " << i);
    if (state.mark_cycles_in_timeline) {
      // Mark start of the new cycle.
      state.timeline.MarkCycleStart();
    }
  
    auto response_list =
        state.controller[i]->ComputeResponseList(horovod_global.shut_down, state);
  
    LOG(DEBUG, "RunLoopOnce(), for nccl_stream = " << i << ", finished ComputeResponseList") ;
    state.mark_cycles_in_timeline =
        state.controller[i]->MarkCyclesInTimelinePending();
  
    // Get tensor name and size data for autotuning.
    int64_t total_tensor_size = 0;
    std::vector<std::string> tensor_names;
    if (state.parameter_manager.IsAutoTuning()) {
      total_tensor_size = horovod_global.tensor_queue.GetTensorDataForAutotuner(
          response_list, tensor_names);
    }
  
    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    int rank = state.controller[i]->GetRank();
    for (auto& response : response_list.responses()) {
      if (!state.group_table.empty()) {
        // Deregister any completed groups
        state.group_table.DeregisterGroups(response.tensor_names());
      }
  
      LOG(TRACE, rank) << "Performing " << response.tensor_names_string();
      LOG(TRACE, rank) << "Processing " << response.tensor_names().size()
                       << " tensors";
      PerformOperation(response, horovod_global, i);
      LOG(TRACE, rank) << "Finished performing "
                       << response.tensor_names_string();
    }
  
    if (state.parameter_manager.IsAutoTuning()) {
      bool should_sync =
          state.parameter_manager.Update(tensor_names, total_tensor_size);
  
      if (should_sync) {
        state.controller[i]->SynchronizeParameters();
      }
    }
    // BAckgroup Thread to shtdown if all sub-communicators shutdown
    answer = answer && !response_list.shutdown();
  }
  return answer;

#else	// NOT HAVE_SUBCOMM

  if (state.mark_cycles_in_timeline) {
    // Mark start of the new cycle.
    state.timeline.MarkCycleStart();
  }

  auto response_list =
      state.controller->ComputeResponseList(horovod_global.shut_down, state);

  state.mark_cycles_in_timeline =
      state.controller->MarkCyclesInTimelinePending();

  // Get tensor name and size data for autotuning.
  int64_t total_tensor_size = 0;
  std::vector<std::string> tensor_names;
  if (state.parameter_manager.IsAutoTuning()) {
    total_tensor_size = horovod_global.tensor_queue.GetTensorDataForAutotuner(
        response_list, tensor_names);
  }

  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  int rank = state.controller->GetRank();
  for (auto& response : response_list.responses()) {
    if (!state.group_table.empty()) {
      // Deregister any completed groups
      state.group_table.DeregisterGroups(response.tensor_names());
    }

    LOG(TRACE, rank) << "Performing " << response.tensor_names_string();
    LOG(TRACE, rank) << "Processing " << response.tensor_names().size()
                     << " tensors";
    PerformOperation(response, horovod_global);
    LOG(TRACE, rank) << "Finished performing "
                     << response.tensor_names_string();
  }

  if (state.parameter_manager.IsAutoTuning()) {
    bool should_sync =
        state.parameter_manager.Update(tensor_names, total_tensor_size);

    if (should_sync) {
      state.controller->SynchronizeParameters();
    }
  }

  return !response_list.shutdown();
#endif
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce(const int* ranks, int nranks) {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.control_operation = ParseControllerOpsFromEnv();
    horovod_global.cpu_operation = ParseCPUOpsFromEnv();
#if HAVE_MPI

#if HAVE_SUBCOMM
    // Enable mpi is it's used either in cpu data transfer or controller
    if (horovod_global.control_operation == LibType::MPI){
      bool* bj = new bool(); 
      horovod_global.joined.push_back(*bj);
      int* ijs = new int(); 
      horovod_global.joined_size.push_back(*ijs);
      ResponseCache* rc = new ResponseCache();
      horovod_global.response_cache.push_back(*rc);
      std::shared_ptr<Controller> ctlr ;
      horovod_global.controller.push_back(ctlr);

      MPIContext* mpictx = new MPIContext;
      horovod_global.controller[0].reset(new MPIController(
          horovod_global.response_cache[0],
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, horovod_global.group_table,
          *mpictx));
    }

    if (horovod_global.cpu_operation == LibType::MPI ||
        horovod_global.control_operation == LibType::MPI) {
      std::shared_ptr<MPIController> pMPIController = 
	      std::static_pointer_cast<MPIController>(horovod_global.controller[0]);
#if HAVE_DDL
      // If DDL is enabled, let DDL ops manage MPI environment.
      auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, gpu_context);
#else
      // Otherwise, let MPI ops be in charge.
      auto mpi_ctx_manager = MPIContextManager();
#endif
      //pMPIController->SetIndex(0);
      horovod_global.controller[0]->SetIndex(0);
      pMPIController->Enable();
      pMPIController->Initialize(horovod_global.controller[0]->GetRanks(), mpi_ctx_manager);
      LOG(TRACE, "InitializeHorovodOnce, MPIController[0] initialized and enabled") ;

      std::vector<int> world_ranks ;
      int world_nranks;
      MPI_Comm_size(MPI_COMM_WORLD, &world_nranks);
      for(int k=0 ; k<world_nranks; k++)
        world_ranks.push_back(k);
      horovod_global.controller[0]->SetRanks(world_ranks, world_nranks);
      horovod_global.controller[0]->SetSize(world_nranks);
      LOG(TRACE, "InitializeHorovodOnce, nranks for controller 0 = ") << world_nranks;
      horovod_global.process_groups.push_back(world_ranks);

      std::unordered_map<std::vector<int32_t>, ncclComm_t> um = { 
        {world_ranks, (ncclComm_t) nullptr}
      };
      nccl_context.nccl_comms.push_back(um);
      LOG(TRACE) << "InitializeHorovodOnce(), Pushed process_group WORLD_COMM " << " into nccl_context.nccl_comms.";
    
      horovod_global.controller[0]->Initialize();
      LOG(TRACE, "InitializeHorovodOnce(), Controller[0] initialized") ;

      LOG(DEBUG, "InitializeHorovodOnce(), First part of init done.");

      if(horovod_global.controller[0]->GetRank() != 0) {
        int retbuf;
        int message = horovod_global.controller[0]->GetRank()+1 ;
        int ret_code = MPI_Gather(&message, 1, MPI_INT, &retbuf, 1, MPI_INT, RANK_ZERO, 
		    pMPIController->GetMpiContext().mpi_comm);
        LOG(DEBUG, "InitializeHorovodOnce(), in TEST , NOT is_coordinator, ret_code = " << ret_code << 
			", retbuf = " << retbuf);
        LOG(DEBUG, "InitializeHorovodOnce(), in TEST , NOT is_coordinator : (mpi_ctx_.mpi_comm == horovod::common::GetMpiWorldComm() = " << 
		(pMPIController->GetMpiContext().mpi_comm_ptr == horovod_global.mpi_world_comm_ptr) );
      }
      else {
        auto recvcounts = new int[4];
        recvcounts[0] = 1;
        auto sendcounts = new int[1];
        sendcounts[0] = 1;
        int ret_code = MPI_Gather(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, RANK_ZERO, 
		    pMPIController->GetMpiContext().mpi_comm);
        LOG(DEBUG, "InitializeHorovodOnce()in TEST , is_coordinator, ret_code = " << ret_code);
	for(int j=0; j<4; j++)
          LOG(DEBUG, "InitializeHorovodOnce()in TEST , is_coordinator, j = " << j << 
			  ", recvcounts[j] = " << recvcounts[j]);
        LOG(DEBUG, "InitializeHorovodOnce(), in TEST , is_coordinator, ret_code = " << ret_code);
        LOG(DEBUG, "InitializeHorovodOnce(), in TEST , is_coordinator : (mpi_ctx_.mpi_comm == horovod::common::GetMpiWorldComm() = " << 
		(pMPIController->GetMpiContext().mpi_comm_ptr == horovod_global.mpi_world_comm_ptr) );
      }
      LOG(DEBUG, "InitializeHorovodOnce(), First part of init done, TEST done");
    }
#else
    // Enable mpi is it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::MPI ||
        horovod_global.control_operation == LibType::MPI) {
      mpi_context.Enable();
    }
    if (horovod_global.control_operation == LibType::MPI){
      horovod_global.controller.reset(new MPIController(
          horovod_global.response_cache,
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, horovod_global.group_table,
          mpi_context));
      horovod_global.controller->SetRanks(ranks, nranks);
    }
#endif
#endif

#if HAVE_GLOO
    LOG(DEBUG) << "InitializeHorovodOnce, HAVE_GLOO";
    // Enable gloo is it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::GLOO ||
        horovod_global.control_operation == LibType::GLOO) {
      gloo_context.Enable();
    }

    if (horovod_global.control_operation == LibType::GLOO) {
      horovod_global.controller.reset(new GlooController(
          horovod_global.response_cache,
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, horovod_global.group_table,
          gloo_context));
    }
#endif
  }
}

} // namespace

// For TEST
MPI_Comm* GetMpiWorldComm() {
    return horovod_global.mpi_world_comm_ptr;
}

void SetMpiWorldComm(MPI_Comm* ptr) {
    horovod_global.mpi_world_comm_ptr = ptr;
}

Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void horovod_init(const int* ranks, int nranks) {
  LOG(DEBUG) << "horovod_init(), nranks = " << nranks;
  InitializeHorovodOnce(ranks, nranks);
}

#if HAVE_MPI
void horovod_init_comm(MPI_Comm comm) {
  LOG(DEBUG , "horovod_init_comm(), nranks = 0") ;
  MPI_Comm_dup(comm, &mpi_context.mpi_comm);
  InitializeHorovodOnce(nullptr, 0);
}
#endif

void horovod_shutdown() {
  LOG(DEBUG , "horovod_shutdown() start.");
  if (horovod_global.background_thread.joinable()) {
    horovod_global.timeline.Shutdown();
    horovod_global.shut_down = true;
    horovod_global.background_thread.join();

    // Reset the initialization flag to allow restarting with horovod_init(...)
    horovod_global.initialize_flag.clear();
    horovod_global.shut_down = false;
    horovod_global.initialization_done = false;
  }
}

bool horovod_is_initialized() {
  return horovod_global.initialization_done;
}

bool horovod_start_timeline(const char* file_name, bool mark_cycles) {
  if (!horovod_global.initialization_done) {
    return false;
  }
  for(int j=0; j<horovod_global.num_nccl_streams;j++) {
    bool is_coordinator = horovod_global.controller[j]->IsCoordinator();
    if (is_coordinator) {
      horovod_global.timeline.Initialize(std::string(file_name), horovod_global.controller[0]->GetSize());
      horovod_global.timeline.SetPendingTimelineFile(std::string(file_name));
    }
    horovod_global.controller[j]->SetMarkCyclesInTimelinePending(mark_cycles);
  }
  return true;
}

bool horovod_stop_timeline() {
  if (!horovod_global.initialization_done) {
    return false;
  }
  for(int j=0; j<horovod_global.num_nccl_streams;j++) {
    if(!horovod_global.controller[j]->TimelineEnabledPending()){
      LOG(INFO) << " Timeline is already stopped. Please start timeline before stopping it.";
      return true;
    }
    bool is_coordinator = horovod_global.controller[j]->IsCoordinator();
    if (is_coordinator) {
      horovod_global.timeline.SetPendingTimelineFile(std::string(""));
    }
  }
  return true;
}

int horovod_rank() {
  LOG(INFO, "Entered horovod_rank()");
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller[0]->GetRank();
}

std::vector<std::vector<int>> horovod_process_groups() {
  LOG(INFO, "Entered horovod_process_groups()");
  return horovod_global.process_groups;
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller[0]->GetLocalRank();
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller[0]->GetSize();
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller[0]->GetLocalSize();
}

bool horovod_is_homogeneous() {
  return horovod_global.controller[0]->IsHomogeneous();
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }

#if HAVE_MPI
  auto mpiController =
      std::dynamic_pointer_cast<MPIController>(horovod_global.controller[0]);
  return mpiController->IsMpiThreadsSupported() ? 1 : 0;
#endif

  return -1;
}

bool horovod_mpi_enabled() {
#if HAVE_MPI
  return mpi_context.IsEnabled();
#else
  return false;
#endif
}

bool horovod_mpi_built() {
#if HAVE_MPI
  return true;
#else
  return false;
#endif
}

bool horovod_gloo_enabled() {
#if HAVE_GLOO
  return gloo_context.IsEnabled();
#else
  return false;
#endif
}

bool horovod_gloo_built() {
#if HAVE_GLOO
  return true;
#else
  return false;
#endif
}

int horovod_nccl_built() {
#if HAVE_NCCL
  return NCCL_VERSION_CODE;
#else
  return 0;
#endif
}

bool horovod_ddl_built() {
#if HAVE_DDL
  return true;
#else
  return false;
#endif
}

bool horovod_ccl_built() {
#if HAVE_CCL
  return true;
#else
  return false;
#endif
}

bool horovod_cuda_built() {
#if HAVE_CUDA
  return true;
#else
  return false;
#endif
}

bool horovod_rocm_built() {
#if HAVE_ROCM
  return true;
#else
  return false;
#endif
}

int horovod_reduce_op_average() {
  return ReduceOp::AVERAGE;
}

int horovod_reduce_op_sum() {
  return ReduceOp::SUM;
}

int horovod_reduce_op_adasum() {
  return ReduceOp::ADASUM;
}

void horovod_wrapper_dummy_fn(){}

// C interface to initialize the process groups, i.e. NCCL sub-communicators
int horovod_nccl_create_process_groups(std::vector<std::vector<int32_t>> process_groups) {
  LOG(TRACE) << "horovod_nccl_create_process_groups() entered. Nb of process_groups = " << process_groups.size();

  horovod_global.num_nccl_streams = 1;

  for (unsigned int i=0, ii=0; i<process_groups.size(); i++) {
    std::vector<int> v = process_groups[i];
    if (std::find(v.begin(), v.end(), horovod_global.controller[0]->GetRank()) != v.end())
      LOG(TRACE, "horovod_nccl_create_process_groups() ,rank_ " << horovod_global.controller[0]->GetRank() << " found in process_group " << i);
    else {
      LOG(TRACE, "horovod_nccl_create_process_groups() ,rank_ " << horovod_global.controller[0]->GetRank() << " NOT found in process_group " << i << " => DOES NOT CREATE the MPIController nor NCCL_COMM");
      continue;
    }
    horovod_global.num_nccl_streams++;

    std::unordered_map<std::vector<int32_t>, ncclComm_t> um = { 
      {process_groups[i], (ncclComm_t) nullptr}
    };
    nccl_context.nccl_comms.push_back(um);
    LOG(TRACE) << " horovod_nccl_create_process_groups(), Pushed process_group " << i << " into nccl_context.nccl_comms.";
  
    bool* bj = new bool(); 
    horovod_global.joined.push_back(*bj);
    int* ijs = new int(); 
    horovod_global.joined_size.push_back(*ijs);
    ResponseCache* rc = new ResponseCache();
    horovod_global.response_cache.push_back(*rc);
    std::shared_ptr<Controller> ctlr ;
    horovod_global.controller.push_back(ctlr);
    LOG(TRACE, "horovod_nccl_create_process_groups(), before new MPIController()");

    MPIContext* mpictx = new MPIContext;
    horovod_global.controller[ii+1].reset(new MPIController(
          horovod_global.response_cache[ii+1],
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, horovod_global.group_table,
          *mpictx));

    LOG(TRACE, "horovod_nccl_create_process_groups(), after new MPIController()");
    for (unsigned int j=0; j<process_groups[i].size(); j++) 
      horovod_global.controller[ii+1]->GetRanks().push_back(process_groups[i][j]);

    v = horovod_global.controller[ii+1]->GetRanks();
    LOG(INFO, "horovod_nccl_create_process_groups(), controller[" << ii << "] ");
    for (std::vector<int>::const_iterator iv = v.begin(); iv != v.end(); ++iv)
      LOG(INFO, "horovod_nccl_create_process_groups(), controller[ii] = " << *iv << ". ");
    horovod_global.controller[ii+1]->SetSize(process_groups[i].size());
    horovod_global.controller[ii+1]->SetIndex(ii+1);
    horovod_global.process_groups.push_back(v);
    LOG(TRACE, "horovod_nccl_create_process_groups() , index = " << ii+1 << ", size = " << process_groups[i].size());
    ii++;
  }
  for(int j=0; j<horovod_global.process_groups.size(); j++)
    for(int k=0; k<horovod_global.process_groups[j].size(); k++)
      LOG(TRACE, "horovod_nccl_create_process_groups() , after creation loop, before creating thread, j=" << j << ", k=" << k <<
		      ", horovod_global.process_groups[j][k]=" << horovod_global.process_groups[j][k]);

  // Reset initialization flag
  horovod_global.initialization_done = false;
  horovod_global.background_thread = std::thread(
        BackgroundThreadLoop, std::ref(horovod_global));
  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG(TRACE, "horovod_nccl_create_process_groups() ,BackgroundThreadLoop created");
  return 0;
}

// // C interface to reset the process groups, i.e. NCCL sub-communicators
int horovod_nccl_shutdown() {
  return 0;
}

} // extern C

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              std::string name, const int device,
                              StatusCallback callback,
                              ReduceOp reduce_op,
                              double prescale_factor,
                              double postscale_factor, int proc_group) {
  // Wrap inputs in std::vector and pass onto multi tensor implementation
  std::vector<std::shared_ptr<OpContext>> contexts;
  std::vector<std::shared_ptr<Tensor>> tensors;
  std::vector<std::shared_ptr<Tensor>> outputs;
  std::vector<std::shared_ptr<ReadyEvent>> ready_events;
  std::vector<std::string> names;
  std::vector<StatusCallback> callbacks;

  contexts.emplace_back(std::move(context));
  tensors.emplace_back(std::move(tensor));
  outputs.emplace_back(std::move(output));
  ready_events.emplace_back(std::move(ready_event));
  names.emplace_back(std::move(name));
  callbacks.emplace_back(std::move(callback));

  return EnqueueTensorAllreduces(contexts, tensors, outputs, ready_events,
                                 names, device, callbacks, reduce_op,
                                 prescale_factor, postscale_factor, proc_group);
}

Status EnqueueTensorAllreduces(std::vector<std::shared_ptr<OpContext>>& contexts,
                               std::vector<std::shared_ptr<Tensor>>& tensors,
                               std::vector<std::shared_ptr<Tensor>>& outputs,
                               std::vector<std::shared_ptr<ReadyEvent>>& ready_events,
                               std::vector<std::string>& names,
                               const int device,
                               std::vector<StatusCallback>& callbacks,
                               ReduceOp reduce_op,
                               double prescale_factor,
                               double postscale_factor, int proc_group) {
  Status status;

  if (reduce_op == ReduceOp::AVERAGE) {
#if !HAVE_ROCM
    // Averaging happens via postscale_factor
    postscale_factor /= horovod_global.controller[proc_group]->GetSize();
#else
    LOG(ERROR, horovod_global.controller[0]->GetRank()) << "Enqueuing AVERAGE allreduce is not allowed.";
    return status.Aborted("AVERAGE not allowed.");
#endif
  } else if (reduce_op == ReduceOp::ADASUM) {
#if HAVE_NCCL && !HAVE_ROCM
    if (device != CPU_DEVICE_ID) {
      // Averaging by local size happens via postscale_factor
      postscale_factor /= horovod_global.controller[proc_group]->GetLocalSize();
    }
#endif
  }

  std::vector<Request> messages;
  std::vector<TensorTableEntry> entries;
  messages.reserve(tensors.size());
  entries.reserve(tensors.size());

  for (unsigned int n = 0; n < tensors.size(); ++n) {
    Request message;
    message.set_request_rank(horovod_global.controller[0]->GetRank());
    message.set_tensor_name(names[n]);
    message.set_tensor_type(tensors[n]->dtype());
    message.set_device(device);
    message.set_prescale_factor(prescale_factor);
    message.set_postscale_factor(postscale_factor);

    if (reduce_op == ReduceOp::ADASUM) {
      message.set_request_type(Request::ADASUM);
    } else {
      message.set_request_type(Request::ALLREDUCE);
    }

    message.set_tensor_shape(tensors[n]->shape().to_vector());
    messages.push_back(std::move(message));

    TensorTableEntry e;
    e.tensor_name = names[n];
    e.context = std::move(contexts[n]);
    // input and output can be the same, only move when safe
    if (tensors[n] != outputs[n]) {
      e.tensor = std::move(tensors[n]);
      e.output = std::move(outputs[n]);
    } else {
      e.tensor = tensors[n];
      e.output = outputs[n];
    }
    e.ready_event = std::move(ready_events[n]);
    e.device = device;
    e.callback = std::move(callbacks[n]);

    entries.push_back(std::move(e));

  }

  std::string tensors_enqueued;
  for (const auto& n : names) {
    tensors_enqueued += n + "; ";
  }
  LOG(TRACE) << "Enqueing " << tensors_enqueued;

  // Only create groups larger than 1 tensor, unless disable_group_fusion is requested.
  // In that case, even single tensor groups are created to enforce disabling fusion.
  if (tensors.size() > 1 || horovod_global.disable_group_fusion) {
    auto group_id = horovod_global.group_table.RegisterGroup(std::move(names));
    for (auto& message : messages) {
      message.set_group_id(group_id);
    }
  }

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  status = horovod_global.tensor_queue.AddToTensorQueueMulti(entries, messages);

  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback, int proc_group) {
  Request message;
  message.set_request_rank(horovod_global.controller[proc_group]->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller[proc_group]->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback, int proc_group) {
  Request message;
  message.set_request_rank(horovod_global.controller[proc_group]->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(Request::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller[proc_group]->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAlltoall(std::shared_ptr<OpContext> context,
                             std::shared_ptr<Tensor> tensor,
                             std::shared_ptr<Tensor> splits,
                             std::shared_ptr<ReadyEvent> ready_event,
                             const std::string name, const int device,
                             StatusCallback callback, int proc_group) {
  LOG(TRACE, "EnqueueTensorAlltoall() start, device = " << device);
  // Check arguments
  if (splits->shape().dims() > 1) {
    return Status::InvalidArgument("alltoall expects a 1D splits tensor");
  }
  if (splits->dtype() != HOROVOD_INT32) {
    return Status::InvalidArgument("alltoall expects splits to contain 32-bit integer elements.");
  }

  Request message;
  message.set_request_rank(horovod_global.controller[proc_group]->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLTOALL);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  int64_t splits_first_dim = splits->shape().dim_size(0);
  int64_t tensor_first_dim = tensor->shape().dim_size(0);
  int world_size = horovod_global.controller[proc_group]->GetSize();
  if (splits_first_dim == world_size) {
    auto splits_data = static_cast<const int32_t*>(splits->data());
    auto sum = std::accumulate(splits_data, splits_data + splits_first_dim, 0);
    if (sum > tensor_first_dim) {
      return Status::InvalidArgument("Sum of splits entries is greater than the first dimension of tensor.");
    }
    e.splits.assign(splits_data,
                    splits_data + splits->shape().num_elements());
  } else if (splits_first_dim == 0) {
    if (tensor_first_dim % world_size != 0) {
      return Status::InvalidArgument("splits not provided, but first dimension of tensor is not an even "
                                     "multiple of the number of workers.");
    }
    e.splits.resize(world_size, tensor_first_dim / world_size);
  } else {
      return Status::InvalidArgument("Number of entries in splits does not equal number of workers.");
  }

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, "Rank " << horovod_global.controller[proc_group]->GetRank() << ", Enqueued tensor : " << name);
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueJoin(std::shared_ptr<OpContext> context,
                   std::shared_ptr<ReadyEvent> ready_event,
                   const std::string name, const int device,
                   StatusCallback callback, int proc_group) {
  Request message;
  message.set_request_rank(horovod_global.controller[proc_group]->GetRank());
  message.set_device(device);
  message.set_request_type(Request::JOIN);

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller[proc_group]->GetRank()) << "Enqueued " << name;
  }
  return status;
}

} // namespace common
} // namespace horovod


