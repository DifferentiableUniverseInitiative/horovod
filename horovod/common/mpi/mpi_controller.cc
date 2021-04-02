// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mpi_controller.h"

#include "../common.h"
#include "../logging.h"

#include <iostream>
#include <string>

namespace horovod {
namespace common {

MPI_Comm& GetMpiWorldComm() ;
void SetMpiWorldComm(MPI_Comm comm) ;

// MPIController
void MPIController::DoInitialization() {
  // Check if multi-thread is supported.
  int provided;
  MPI_Query_thread(&provided);
  mpi_threads_supported_ = (provided == MPI_THREAD_MULTIPLE);

  LOG(DEBUG, "MPIController::DoInitialization(), entered.");
  // Get MPI rank to determine if we are rank zero.
  MPI_Comm_rank(mpi_ctx_.mpi_comm, &rank_);
  is_coordinator_ = rank_ == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  MPI_Comm_size(mpi_ctx_.mpi_comm, &size_);

  if (is_coordinator_) 
    LOG(DEBUG) << "MPIController::DoInitialization(), Coordinator : Starting Horovod with " << size_ << " processes";
  else
    LOG(DEBUG) << "MPIController::DoInitialization(), Not a Coordinator : Starting Horovod with " << size_ << " processes";

  ///////////// TODO : Initialize ransk_ and local_ranks_ and cross_ranks_//////////////////////


  // Determine local rank by querying the local communicator.
  MPI_Comm_rank(mpi_ctx_.local_comm, &local_rank_);
  MPI_Comm_size(mpi_ctx_.local_comm, &local_size_);
  local_comm_ranks_ = std::vector<int>((size_t)local_size_);
  local_comm_ranks_[local_rank_] = rank_;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), 1,
                MPI_INT, mpi_ctx_.local_comm);

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  auto local_sizes = std::vector<int>(size_);
  MPI_Allgather(&local_size_, 1, MPI_INT, local_sizes.data(), 1, MPI_INT,
                mpi_ctx_.mpi_comm);

  is_homogeneous_ = true;
  for (int i = 0; i < size_; ++i) {
    if (local_sizes[i] != local_size_) {
      is_homogeneous_ = false;
      break;
    }
  }

  // Get cross-node rank and size in case of hierarchical allreduce.
  MPI_Comm_rank(mpi_ctx_.cross_comm, &cross_rank_);
  MPI_Comm_size(mpi_ctx_.cross_comm, &cross_size_);

  // Construct a shorter local sizes vector with length cross size.
  // e.g. For local_sizes = {4, 4, 4, 4, 3, 3, 3},
  //      we want to construct a local_sizes_for_cross_rank_ = {4, 3}
  local_sizes_for_cross_rank_ = std::vector<int>(cross_size_);
  int displacement = 0;
  // For each cross rank iter, set corresponding local size and move
  // displacement advance by the local size
  for (int cross_rank = 0; cross_rank < cross_size_; ++cross_rank) {
    local_sizes_for_cross_rank_[cross_rank] = local_sizes[displacement];
    displacement += local_sizes[displacement];
  }

  LOG(DEBUG, "MPIController::DoInitialization() ended.");
}

int MPIController::GetTypeSize(DataType dtype) {
  return mpi_ctx_.GetMPITypeSize(dtype);
}

void MPIController::CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                        int count) {
  LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() start, count = " << count << ", bitvector.size = " << bitvector.size());
  for(int i=0; i<bitvector.size(); i++)
    LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() for i = " << i << ", bitvector.data = " << bitvector[i]);
  if(mpi_ctx_.mpi_comm == MPI_COMM_NULL)
    LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() mpi_comm = MPI_COMM_NULL");
  else if ( mpi_ctx_.mpi_comm == horovod::common::GetMpiWorldComm())
    LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() mpi_comm = horovod_global.mpi_world_comm");
  else
    LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() mpi_comm ! MPI_COMM_NULL and  mpi_comm ! MPI_COMM_WORLD ");

  int ret_code = MPI_Allreduce(MPI_IN_PLACE, bitvector.data(), count,
                               MPI_LONG_LONG_INT, MPI_BAND, MPI_COMM_WORLD);
                               //MPI_LONG_LONG_INT, MPI_BAND, mpi_ctx_.mpi_comm);
  LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() after MPI_Allreduce.");
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_AllReduce failed, see MPI output for details.");
  }
  LOG(DEBUG, "MPIController::CrossRankBitwiseAnd() end.");
}

void MPIController::CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                       int count) {
  int ret_code = MPI_Allreduce(MPI_IN_PLACE, bitvector.data(), count,
                               MPI_LONG_LONG_INT, MPI_BOR, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_AllReduce failed, see MPI output for details.");
  }
}

void MPIController::RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                     std::vector<RequestList>& ready_list) {
  // Rank zero has put all its own tensors in the tensor count table.
  // Now, it should count all the tensors that are coming from other
  // ranks at this tick.

  // 1. Get message lengths from every rank.
  LOG(DEBUG, "MPIController::RecvReadyTensors(), started. ready_to_reduce.size = " << ready_to_reduce.size() << 
	  ", ready_list.size = " << ready_list.size() << ", size_ = " << size_);
  //auto recvcounts = new int[size_];
  auto recvcounts = new int[4];
  recvcounts[0] = 0;
  auto sendcounts = new int[1];
  sendcounts[0] = 0;
  if(mpi_ctx_.mpi_comm == MPI_COMM_NULL)
	 LOG(INFO, "MPIController::RecvReadyTensors(), mpi_comm = MPI_COMM_NULL");
  else if(mpi_ctx_.mpi_comm == horovod::common::GetMpiWorldComm())
	 LOG(INFO, "MPIController::RecvReadyTensors(), mpi_comm = horovod_global.mpi_world_comm");
  else {
    LOG(INFO, "MPIController::RecvReadyTensors(),  before MPI_Gather, STRANGE : mpi_comm != MPI_COMM_NULL and mpi_comm != mpi_comm_world");
    LOG(DEBUG, "MPIController::RecvReadyTensors(), before MPI_Gather, STRANGE : (mpi_ctx_.mpi_comm != horovod::common::GetMpiWorldComm() = " << (&mpi_ctx_.mpi_comm == &horovod::common::GetMpiWorldComm()) );
  }

  for (int i = 0; i < size_; ++i) {
	 recvcounts[i] = 0;
  }
  //int retcode = MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm) ;
  LOG(INFO, "MPIController::RecvReadyTensors(), mpi_cookie = " << mpi_ctx_.mpi_cookie);
  int retcode = MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm) ;
  if(retcode != MPI_SUCCESS)
	 LOG(INFO, "MPIController::RecvReadyTensors(), MPI_Gather Error !!! retoode = " << retcode);
  else
	 LOG(INFO, "MPIController::RecvReadyTensors(), MPI_Gather returned OK ");
  for (int i = 0; i < size_; ++i) {
	 LOG(INFO, "MPIController::RecvReadyTensors(), MPI_Gather returned OK, recvcounts[i] = " << recvcounts[i]);
  }


  // 2. Compute displacements.
  auto displcmnts = new int[size_];
  size_t total_size = 0;
  for (int i = 0; i < size_; ++i) {
    if (i == 0) {
      displcmnts[i] = 0;
    } else {
      displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
    }
    total_size += recvcounts[i];
  }

  LOG(DEBUG, "MPIController::RecvReadyTensors(), before MPI_Gatherv.");
  // 3. Collect messages from every rank.
  auto buffer = new uint8_t[total_size];
  retcode = MPI_Gatherv(nullptr, 0, MPI_BYTE, buffer, recvcounts, displcmnts, MPI_BYTE,
              RANK_ZERO, mpi_ctx_.mpi_comm) ;
  if(retcode != MPI_SUCCESS)
    LOG(INFO, "MPIController::RecvReadyTensors(), MPI_Gatherv Error !!! retoode = " << retcode);
  else
    LOG(INFO, "MPIController::RecvReadyTensors(), MPI_Gatherv returned OK !!!");

  // 4. Process messages.
  // create a dummy list for rank 0
  ready_list.emplace_back();
  LOG(DEBUG) << "MPIController::RecvReadyTensors(), before loop on size_ = " << size_;
  for (int i = 1; i < size_; ++i) {
    LOG(DEBUG) << "MPIController::RecvReadyTensors(), in loop, start for i = " << i; 
    auto rank_buffer_ptr = buffer + displcmnts[i];
    LOG(DEBUG) << "MPIController::RecvReadyTensors(), in loop, after displcmnts, displcmnts[i] = " << displcmnts[i];
    RequestList received_message_list;
    RequestList::ParseFromBytes(received_message_list, rank_buffer_ptr);
    LOG(DEBUG) << "MPIController::RecvReadyTensors(), in loop, after ParseFromBytes.";
    ready_list.push_back(std::move(received_message_list));
    LOG(DEBUG) << "MPIController::RecvReadyTensors(), in loop, end for i = " << i; 
  }

  // 5. Free buffers.
  LOG(DEBUG) << "MPIController::RecvReadyTensors(), ended.";
  delete[] recvcounts;
  delete[] displcmnts;
  delete[] buffer;
}

void MPIController::SendFinalTensors(ResponseList& response_list) {
  // Notify all nodes which tensors we'd like to reduce at this step.
  std::string encoded_response;
  LOG(DEBUG, "MPIController::SendFinalTensors(), started, index = " << GetIndex()) ;
  ResponseList::SerializeToString(response_list, encoded_response);
  int encoded_response_length = (int)encoded_response.length() + 1;
  LOG(DEBUG) << "MPIController::SendFinalTensors() before MPI_Bcast 1.";
  MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm);

  LOG(DEBUG) << "MPIController::SendFinalTensors() before MPI_Bcast 2.";
  MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length, MPI_BYTE,
            RANK_ZERO, mpi_ctx_.mpi_comm);
  LOG(DEBUG) << "MPIController::SendFinalTensors(), ended.";
}

void MPIController::SendReadyTensors(RequestList& message_list) {
  std::string encoded_message;
  LOG(DEBUG, "MPIController::SendReadyTensors(), started, controller_index = " << GetIndex());
  RequestList::SerializeToString(message_list, encoded_message);
  LOG(DEBUG, "MPIController::SendReadyTensors(), after SerializeToString, message_list.size = " << 
	  message_list.requests().size());;
  int encoded_message_length = (int)encoded_message.length() + 1;
  LOG(DEBUG, "MPIController::SendReadyTensors(), before MPI_Gather, encoded_message_length = " << encoded_message_length);
  if(mpi_ctx_.mpi_comm == MPI_COMM_NULL)
    LOG(DEBUG, "MPIController::SendReadyTensors(), before MPI_Gather, mpi_comm is NULL");
  else if(mpi_ctx_.mpi_comm == horovod::common::GetMpiWorldComm()) {
    LOG(DEBUG, "MPIController::SendReadyTensors(), before MPI_Gather, mpi_comm = horovod_global.mpi_world_comm");
  }
  else {
    LOG(DEBUG, "MPIController::SendReadyTensors(), before MPI_Gather, STRANGE : mpi_comm is not NULL and is NOT mpi_comm_world");
    LOG(DEBUG, "MPIController::SendReadyTensors(), before MPI_Gather, STRANGE : (mpi_ctx_.mpi_comm == horovod::common::GetMpiWorldComm() = " << (&mpi_ctx_.mpi_comm == &horovod::common::GetMpiWorldComm()) );
  }
  //Replaced in the line below nullptr by by &retbuf, as Segmentation violation 
  //int ret_code = MPI_Gather(&encoded_message_length, 1, MPI_INT, nullptr, 1,
  auto retbuf = new int[4];
  // int retbuf;
  //int ret_code = MPI_Gather(&encoded_message_length, 1, MPI_INT, &retbuf, 1,
  int ret_code = MPI_Gather(&encoded_message_length, 1, MPI_INT, retbuf, 1,
                            MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    LOG(DEBUG, "MPIController::SendReadyTensors(), after MPI_Gather, ret_code != MPI_SUCCESS");
    throw std::runtime_error("MPI_Gather failed, see MPI output for details.");
  }

  ret_code = MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length,
                         MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE,
                         RANK_ZERO, mpi_ctx_.mpi_comm);
  LOG(DEBUG, "MPIController::SendReadyTensors(), after MPI_Gatherv.");
  if (ret_code != MPI_SUCCESS) {
    LOG(DEBUG, "MPIController::SendReadyTensors(), MPI_Gatherv error = " << ret_code << " !!!!!!!!");
    throw std::runtime_error("MPI_Gather failed, see MPI output for details.");
  }
  LOG(DEBUG, "MPIController::SendReadyTensors(), ended.");
}

void MPIController::RecvFinalTensors(ResponseList& response_list) {
  LOG(DEBUG) << "MPIController::RecvFinalTensors(), started .";
  int msg_length;
  int ret_code =
      MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm);
  LOG(DEBUG) << "MPIController::RecvFinalTensors(), after MPI_Bcast 1.";
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }

  auto buffer = new uint8_t[msg_length];
  LOG(DEBUG) << "MPIController::RecvFinalTensors(), before MPI_Bcast 2.";
  ret_code =
      MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, mpi_ctx_.mpi_comm);
  LOG(DEBUG) << "MPIController::RecvFinalTensors(), after MPI_Bcast 2.";
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }
  LOG(DEBUG, "MPIController::RecvFinalTensors(), before ParseFromBytes.");
  ResponseList::ParseFromBytes(response_list, buffer);
  LOG(DEBUG, "MPIController::RecvFinalTensors(), after ParseFromBytes.");
  delete[] buffer;
  LOG(DEBUG, "MPIController::RecvFinalTensors(), ended.");
}

void MPIController::Bcast(void* buffer, size_t size, int root_rank,
                          Communicator communicator) {
  LOG(DEBUG) << "MPIController::Bcast(), started.";
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(communicator);
  LOG(DEBUG) << "MPIController::Bcast(), after GetMPICommunicator.";
  int ret_code = MPI_Bcast(buffer, size, MPI_BYTE, root_rank, comm);
  LOG(DEBUG) << "MPIController::Bcast(), after MPI_cast.";
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }
  LOG(DEBUG) << "MPIController::Bcast(), ended .";
}

void MPIController::AlltoallGetRecvSplits(const std::vector<int32_t>& splits,
                                          std::vector<int32_t>& recvsplits) {
  recvsplits.resize(size_);
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL);
  int ret_code = MPI_Alltoall(splits.data(), 1, MPI_INT,
                              recvsplits.data(), 1, MPI_INT,
                              comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Alltoall failed, see MPI output for details.");
  }
};

void MPIController::Barrier(Communicator communicator) {
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(communicator);
  int ret_code = MPI_Barrier(comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed, see MPI output for details.");
  }
}

} // namespace common
} // namespace horovod
