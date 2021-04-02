// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

#include "controller.h"
#include "mpi/mpi_controller.h"

#include <atomic>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>

#include "global_state.h"
#include "logging.h"
#include "operations.h"

#if HAVE_CUDA
#include "ops/cuda/cuda_kernels.h"
#endif


namespace horovod {
namespace common {


void Controller::SynchronizeParameters() {
  ParameterManager::Params param;
  if (is_coordinator_) {
    param = parameter_manager_.GetParams();
  }

  void* buffer = (void*)(&param);
  size_t param_size = sizeof(param);
  Bcast(buffer, param_size, 0, Communicator::GLOBAL);

  if (!is_coordinator_) {
    parameter_manager_.SetParams(param);
  }
  parameter_manager_.Reset();
}

Controller::Controller(ResponseCache& response_cache, TensorQueue& tensor_queue,
                       Timeline& timeline, ParameterManager& parameter_manager,
                       GroupTable& group_table)
    : stall_inspector_(response_cache), tensor_queue_(tensor_queue),
      timeline_(timeline), 
      response_cache_(response_cache),
      parameter_manager_(parameter_manager), group_table_(group_table) { 
	uint32_t old_capacity = response_cache.capacity();
        LOG(DEBUG, "Controller::Controller, response_cache.old_capacity = " << old_capacity );
	response_cache.set_capacity(256);
        LOG(DEBUG, "Controller::Controller, response_cache.capacity = " << response_cache.capacity() );
        response_cache_ = response_cache;
        LOG(DEBUG, "Controller::Controller, response_cache_.capacity = " << response_cache_.capacity() );
	response_cache.set_capacity(old_capacity);
        LOG(DEBUG, "Controller::Controller, after reset, response_cache.capacity = " << response_cache.capacity() );
      }

void Controller::Initialize() {
  response_cache_.clear();

  // Initialize concrete implementations.
  DoInitialization();
}

ResponseList Controller::ComputeResponseList(std::atomic_bool& shut_down,
                                             HorovodGlobalState& state) {
  LOG(DEBUG, "ComputeResponseList() entered, size_ = " << GetSize());

  //std::shared_ptr<MPIController> pMPIController = std::static_pointer_cast<MPIController>(this);
  MPIController* pMPIController = dynamic_cast<MPIController*>(this);

  int index_controller = GetIndex();
  LOG(DEBUG, "ComputeResponseList(), index_controller = " << index_controller << ", GetRank() = " << GetRank());

  std::vector<int> ranks = pMPIController->GetRanks() ;
  for (int i=0; i<ranks.size(); i++)
    LOG(DEBUG, "ComputeResponseList(), GetRanks()[" << i << "] = " << ranks[i]);

  // TEST MPI_Gather 1
  auto recvcounts1 = new int[4];
  recvcounts1[0] = 0;
  auto sendcounts1 = new int[1];
  sendcounts1[0] = GetRank()+1;
  //int retcode = MPI_Gather(sendcounts1, 1, MPI_INT, recvcounts1, 1, MPI_INT, RANK_ZERO, pMPIController->GetMpiContext().mpi_comm) ;
  int retcode = MPI_Gather(sendcounts1, 1, MPI_INT, recvcounts1, 1, MPI_INT, 0, pMPIController->GetMpiContext().mpi_comm) ;
  if(retcode != MPI_SUCCESS)
    LOG(INFO, "MPIController::ComputeResponseList(), MPI_Gather TEST 1 Error !!! retoode = " << retcode);
  else
    LOG(INFO, "MPIController::ComputeResponseList(), MPI_Gather TEST 1 OK");
  for (int i = 0; i < 4; ++i) 
    LOG(INFO, "MPIController::ComputeResponseList() TEST 1  recvcounts1[i] = " << recvcounts1[i]);

  // Update cache capacity if autotuning is active.
  LOG(DEBUG, "ComputeResponseList(),  before IsAutoTuning, response_cache.capacity_ = " << response_cache_.capacity());
  LOG(DEBUG, "ComputeResponseList(),  before IsAutoTuning, parameter_manager.CacheEnabled = " << (int)parameter_manager_.CacheEnabled());
  if (parameter_manager_.IsAutoTuning()) {
    LOG(DEBUG, "ComputeResponseList(),  IsAutoTuning() = TRUE");
    response_cache_.set_capacity((int)parameter_manager_.CacheEnabled() *
                                 cache_capacity_);
  }
  else
    LOG(DEBUG, "ComputeResponseList(),  IsAutoTuning() = FALSE");
  LOG(DEBUG, "ComputeResponseList(),  after IsAutoTuning, response_cache.capacity_ = " << response_cache_.capacity());

  // Copy the data structures out from parameters.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.

  LOG(DEBUG, "ComputeResponseList(),  before CacheCoordinator, num_active_bits = " << response_cache_.num_active_bits());
  CacheCoordinator cache_coordinator(response_cache_.num_active_bits());

  // message queue used only in this cycle
  std::deque<Request> message_queue_tmp;
  tensor_queue_.PopMessagesFromQueue(message_queue_tmp);
  for (auto& message : message_queue_tmp) {
    LOG(DEBUG, "ComputeResponseList(), message_queue_tmp loop");
    if (message.request_type() == Request::JOIN) {
      LOG(DEBUG, "ComputeResponseList(), message_queue_tmp loop, Request::JOIN");
      state.joined[index_controller] = true;
      cache_coordinator.set_uncached_in_queue(true);
      continue;
    }

    // Keep track of cache hits
    if (response_cache_.capacity() > 0) {
      LOG(DEBUG, "ComputeResponseList(), message_queue_tmp loop, response_cache.capacity() > 0");
      auto cache_ = response_cache_.cached(message);
      if (cache_ == ResponseCache::CacheState::HIT) {
        uint32_t cache_bit = response_cache_.peek_cache_bit(message);
        cache_coordinator.record_hit(cache_bit);

        // Record initial time cached tensor is encountered in queue.
        stall_inspector_.RecordCachedTensorStart(message.tensor_name());

      } else {
        if (cache_ == ResponseCache::CacheState::INVALID) {
          uint32_t cache_bit = response_cache_.peek_cache_bit(message);
          cache_coordinator.record_invalid_bit(cache_bit);
        }
        cache_coordinator.set_uncached_in_queue(true);

        // Remove timing entry if uncached or marked invalid.
        stall_inspector_.RemoveCachedTensor(message.tensor_name());
      }
    }
  }

  LOG(DEBUG, "ComputeResponseList() after analyzing message_queue_tmp , GetRank() = " << GetRank());
  LOG(DEBUG, "ComputeResponseList() after analyzing message_queue_tmp , capacity = " <<  response_cache_.capacity());
  if (state.joined[index_controller] && response_cache_.capacity() > 0) {
    LOG(DEBUG, "state.joined[index_controller] && response_cache_.capacity() > 0");
    for (uint32_t bit : response_cache_.list_all_bits()) {
      LOG(DEBUG, "before record_hit");
      cache_coordinator.record_hit(bit);
      LOG(DEBUG, "after record_hit");
    }
  }

  // Flag indicating that the background thread should shut down.
  bool should_shut_down = shut_down;
  LOG(DEBUG, "ComputeResponseList() should_shut_down = " << should_shut_down);

  // Check for stalled tensors.
  LOG(DEBUG, "ComputeResponseList() before ShouldPerformCheck");
  if (stall_inspector_.ShouldPerformCheck()) {
    if (is_coordinator_) {
      should_shut_down |= stall_inspector_.CheckForStalledTensors(GetSize());
      LOG(DEBUG, "ComputeResponseList() ShouldPerformCheck, is_coordinator, should_shut_down = " << should_shut_down);
    }
    else
      LOG(DEBUG, "ComputeResponseList() ShouldPerformCheck, NOT is_coordinator");

    if (response_cache_.capacity() > 0) {
      LOG(DEBUG, "ComputeResponseList() ShoudPerformCheck, response_cache.capacity > 0");
      stall_inspector_.InvalidateStalledCachedTensors(cache_coordinator);
    }
    stall_inspector_.UpdateCheckTime();
  }

  cache_coordinator.set_should_shut_down(should_shut_down);
  LOG(DEBUG, "ComputeResponseList() after UpdateCheckTime, should_shut_down = " << should_shut_down);

  if (response_cache_.capacity() > 0) {
    LOG(DEBUG, "ComputeResponseList() response_cache_.capacity > 0");
    // Obtain common cache hits and cache invalidations across workers. Also,
    // determine if any worker has uncached messages in queue or requests
    // a shutdown. This function removes any invalid cache entries, if they
    // exist.
    CoordinateCacheAndState(cache_coordinator);
    LOG(DEBUG, "ComputeResponseList() after CoordinateCacheAndState()");
    // Remove uncommon cached tensors from queue and replace to state
    // queue for next cycle. Skip adding common cached tensors to
    // queue as they are handled separately.
    std::deque<Request> messages_to_replace;
    size_t num_messages = message_queue_tmp.size();
    LOG(DEBUG, "ComputeResponseList() num_messages = " << num_messages);
    for (size_t i = 0; i < num_messages; ++i) {
      LOG(DEBUG, "ComputeResponseList() start loop for message = " << i);
      auto& message = message_queue_tmp.front();
      if (response_cache_.cached(message) == ResponseCache::CacheState::HIT) {
        LOG(DEBUG, "ComputeResponseList() cached = ::HIT for message " << i);
        uint32_t cache_bit = response_cache_.peek_cache_bit(message);
        if (cache_coordinator.cache_hits().find(cache_bit) ==
            cache_coordinator.cache_hits().end()) {
          LOG(DEBUG, "ComputeResponseList() to push_back in messages_to_replace for message " << i);
          // Try to process again in next cycle.
          messages_to_replace.push_back(std::move(message));
        } else {
          LOG(DEBUG, "ComputeResponseList() to RemoveCachdTensor for messagge " << i);
          // Remove timing entry for messages being handled this cycle.
          stall_inspector_.RemoveCachedTensor(message.tensor_name());
        }
        LOG(DEBUG, "ComputeResponseList() cached = ::HIT end for message " << i);
      } else {
        LOG(DEBUG, "ComputeResponseList() cached != ::HIT for message " << i);
        // Remove timing entry for messages being handled this cycle.
        stall_inspector_.RemoveCachedTensor(message.tensor_name());
        message_queue_tmp.push_back(std::move(message));
        LOG(DEBUG, "ComputeResponseList() cached != ::HIT end for message " << i);
      }
      message_queue_tmp.pop_front();
      LOG(DEBUG, "ComputeResponseList() end loop for message = " << i);
    }
    tensor_queue_.PushMessagesToQueue(messages_to_replace);
  }
  else
    LOG(DEBUG, "ComputeResponseList() response_cache_.capacity = 0 ");
  LOG(DEBUG, "ComputeResponseList() after test response_cache_.capacity ");

  if (!message_queue_tmp.empty()) {
    LOG(TRACE, rank_) << "ComputeResponseList() Sent " << message_queue_tmp.size()
                      << " messages to coordinator.";
  }
  else
    LOG(TRACE, rank_) << "ComputeResponseList() message_queue_empty, No message sent to coordinator.";

  LOG(DEBUG, "ComputeResponseList() before set_shutdownn should_shut_down = " << cache_coordinator.should_shut_down());
  ResponseList response_list;
  response_list.set_shutdown(cache_coordinator.should_shut_down());

  bool need_communication = true;
  LOG(DEBUG, "ComputeResponseList() before computing need_communication, uncached_in_queue = " << cache_coordinator.uncached_in_queue() );
  if (response_cache_.capacity() > 0 &&
      !cache_coordinator.uncached_in_queue()) {
    // if cache is enabled and no uncached new message coming in, no need for
    // additional communications
    need_communication = false;

    LOG(DEBUG, "ComputeResponseList() need_communication = FALSE");
    // If no messages to send, we can simply return an empty response list;
    if (cache_coordinator.cache_hits().empty()) {
      LOG(DEBUG, "ComputeResponseList() cache_coordinator.cache_hits().empty(), return");
      return response_list;
    }
    // otherwise we need to add cached messages to response list.
  }

  if (!need_communication) {
    LOG(DEBUG, "ComputeResponseList() !need_communication");
    // If all messages in queue have responses in cache, use fast path with
    // no additional coordination.

    // If group fusion is disabled, fuse tensors in groups separately
    if (state.disable_group_fusion && !group_table_.empty()) {
      // Note: need group order to be based on position in cache for global consistency
      std::vector<int> common_ready_groups;
      std::unordered_set<int> processed;
      for (auto bit : cache_coordinator.cache_hits()) {
        const auto& tensor_name = response_cache_.peek_response(bit).tensor_names()[0];
        int group_id = group_table_.GetGroupIDFromTensorName(tensor_name);
        if (group_id != NULL_GROUP_ID && processed.find(group_id) == processed.end()) {
          common_ready_groups.push_back(group_id);
          processed.insert(group_id);
        }
      }

      for (auto id : common_ready_groups) {
        std::deque<Response> responses;
        for (const auto &tensor_name : group_table_.GetGroupTensorNames(id)) {
          auto bit = response_cache_.peek_cache_bit(tensor_name);
          responses.push_back(response_cache_.get_response(bit));
          // Erase cache hit to avoid processing a second time.
          cache_coordinator.erase_hit(bit);
        }

        FuseResponses(responses, state, response_list);
      }
    }


    std::deque<Response> responses;
    // Convert cache hits to responses. Populate so that least
    // recently used responses get priority. All workers call the code
    // here so we use the get method here to consistently update the cache
    // order.
    for (auto bit : cache_coordinator.cache_hits()) {
      responses.push_back(response_cache_.get_response(bit));
    }

    // Fuse responses as normal.
    FuseResponses(responses, state, response_list);
    LOG(DEBUG, "ComputeResponseList() in !need_communication, after FuseResponses , should_shut_down = " << cache_coordinator.should_shut_down());
    response_list.set_shutdown(cache_coordinator.should_shut_down());
  } else {
    LOG(DEBUG, "ComputeResponseList() NOT !need_communication");
    // There are uncached messages coming in, need communication to figure out
    // whether those are ready to be reduced.

    // Collect all tensors that are ready to be reduced. Record them in the
    // tensor count table (rank zero) or send them to rank zero to be
    // recorded (everyone else).
    std::vector<std::string> ready_to_reduce;

    if (is_coordinator_) {
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Before loop on message_queueu_tmp.";

    // TEST MPI_Gather 1
    auto recvcounts2 = new int[4];
    recvcounts2[0] = 0;
    auto sendcounts2 = new int[1];
    sendcounts2[0] = index_controller;
    int retcode = MPI_Gather(sendcounts2, 1, MPI_INT, recvcounts2, 1, MPI_INT, RANK_ZERO, pMPIController->GetMpiContext().mpi_comm) ;
    if(retcode != MPI_SUCCESS)
	LOG(INFO, "MPIController::ComputeResponseList(), MPI_Gather TEST 2 Error !!! retoode = " << retcode);
    else
        LOG(INFO, "MPIController::ComputeResponseList(), MPI_Gather TEST 2 OK");
    for (int i = 0; i < 4; ++i)
        LOG(INFO, "MPIController::ComputeResponseList() TEST 2  recvcounts2[i] = " << recvcounts2[i]);
	
      while (!message_queue_tmp.empty()) {
        // Pop the first available message
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Loop before message_queue_tmp.front()";
        Request message = message_queue_tmp.front();
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Loop before message_queue_tmp.pop_front()";
        message_queue_tmp.pop_front();
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Loop after message_queue_tmp.pop_front()";

        if (message.request_type() == Request::JOIN) {
          LOG(TRACE, "ComputeResponseList() is_coordinator_ = TRUE, request_type = JOIN");
          state.joined_size[index_controller]++;
          continue;
        }
	else
          LOG(TRACE, "ComputeResponseList() is_coordinator_ = TRUE, request_type = " << message.request_type());

        bool reduce = IncrementTensorCount(message, state.joined_size[index_controller]);
        LOG(TRACE, "ComputeResponseList() is_coordinator_ = TRUE, reduce = " << reduce <<
			", joined_size[index_controller] = " << state.joined_size[index_controller] <<
			", tensor_name = " <<  message.tensor_name() << ", request_rank = " <<  message.request_rank() <<
			", GetSize() = " << GetSize() );
        stall_inspector_.RecordUncachedTensorStart(
            message.tensor_name(), message.request_rank(), GetSize());
        if (reduce) {
          ready_to_reduce.push_back(message.tensor_name());
          LOG(TRACE, "ComputeResponseList() is_coordinator_ = TRUE, ready_to_reduce.push_back() done.");
        }
      }
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, after Loop message_queue_tmp.";

      // Receive ready tensors from other ranks
      std::vector<RequestList> ready_list;
      RecvReadyTensors(ready_to_reduce, ready_list);
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, After RecvReadyTensors." ;

      // Process messages.
      for (int i = 0; i < GetSize(); ++i) {
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Adding messages from rank " << i;
        auto received_message_list = ready_list[i];
        for (auto& received_message : received_message_list.requests()) {
          auto& received_name = received_message.tensor_name();

          if (received_message.request_type() == Request::JOIN) {
            state.joined_size[index_controller]++;
            continue;
          }

          LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Before IncrementTensorCount.";
          bool reduce = IncrementTensorCount(received_message, state.joined_size[index_controller]);
          stall_inspector_.RecordUncachedTensorStart(
              received_message.tensor_name(), received_message.request_rank(),
              GetSize());
          if (reduce) {
            ready_to_reduce.push_back(received_name);
          }
        }
        if (received_message_list.shutdown()) {
          // Received SHUTDOWN request from one of the workers.
          should_shut_down = true;
          LOG(TRACE, "ComputeResponseList() in received_message_list.shutdown(), setting should_shut_down = true");
        }
      }
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Before IncrementTensorCount.";

      // Check if tensors from previous ticks are ready to reduce after Joins.
      if (state.joined_size[index_controller] > 0) {
        for (auto& table_iter : message_table_) {
          int count = (int)table_iter.second.size();
          if (count == (GetSize() - state.joined_size[index_controller]) &&
              std::find(ready_to_reduce.begin(), ready_to_reduce.end(),
                        table_iter.first) == ready_to_reduce.end()) {
            state.timeline.NegotiateEnd(table_iter.first);
            ready_to_reduce.push_back(table_iter.first);
          }
        }
      }
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Before Fuse.";

      // Fuse tensors in groups before processing others.
      if (state.disable_group_fusion && !group_table_.empty()) {

        // Extract set of common groups from coordinator tensor list and cache hits.
        std::vector<int> common_ready_groups;
        std::unordered_set<int> processed;

        for (const auto& tensor_name : ready_to_reduce) {
          int group_id = group_table_.GetGroupIDFromTensorName(tensor_name);
          if (group_id != NULL_GROUP_ID && processed.find(group_id) == processed.end()) {
            common_ready_groups.push_back(group_id);
            processed.insert(group_id);
            // Leaving name in list, to be skipped later.
          }
        }

        if (response_cache_.capacity() > 0) {
          for (auto bit : cache_coordinator.cache_hits()) {
            const auto& tensor_name = response_cache_.peek_response(bit).tensor_names()[0];
            int group_id = group_table_.GetGroupIDFromTensorName(tensor_name);
            if (group_id != NULL_GROUP_ID && processed.find(group_id) == processed.end()) {
              common_ready_groups.push_back(group_id);
              processed.insert(group_id);
            }
          }
        }

        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, Before Loop common_ready_groups.";
        // For each ready group, form and fuse response lists independently
        for (auto id : common_ready_groups) {
          std::deque<Response> responses;
          for (const auto &tensor_name : group_table_.GetGroupTensorNames(id)) {
            if (message_table_.find(tensor_name) != message_table_.end()) {
              // Uncached message
              Response response = ConstructResponse(tensor_name, state.joined_size[index_controller]);
              responses.push_back(std::move(response));

            } else {
              // Cached message
              auto bit = response_cache_.peek_cache_bit(tensor_name);
              responses.push_back(response_cache_.get_response(bit));
              // Erase cache hit to avoid processing a second time.
              cache_coordinator.erase_hit(bit);
            }
          }

          FuseResponses(responses, state, response_list);
        }
      }


      // At this point, rank zero should have a fully updated tensor count
      // table and should know all the tensors that need to be reduced or
      // gathered, and everyone else should have sent all their information
      // to rank zero. We can now do reductions and gathers; rank zero will
      // choose which ones and in what order, and will notify the other ranks
      /// before doing each reduction.
      //
      LOG(DEBUG, "ComputeResponseList() is_coordinator_ = TRUE, Constructing Reponse");
      std::deque<Response> responses;

      if (response_cache_.capacity() > 0) {
        // Prepopulate response list with cached responses. Populate so that
        // least recently used responses get priority. Since only the
        // coordinator rank calls this code, use peek instead of get here to
        // preserve cache order across workers.
        // No need to do this when all ranks did Join.
        if (state.joined_size[index_controller] < GetSize()) {
          for (auto bit : cache_coordinator.cache_hits()) {
            responses.push_back(response_cache_.peek_response(bit));
          }
        }
      }

      for (auto& tensor_name : ready_to_reduce) {
        // Skip tensors in group that were handled earlier.
        if (state.disable_group_fusion &&
            !group_table_.empty() &&
            group_table_.GetGroupIDFromTensorName(tensor_name) != NULL_GROUP_ID) {
          continue;
        }

        Response response = ConstructResponse(tensor_name, state.joined_size[index_controller]);
        responses.push_back(std::move(response));
      }
      if (state.joined_size[index_controller] == GetSize()) {
        // All ranks did Join(). Send the response, reset joined size.
        Response join_response;
        join_response.set_response_type(Response::JOIN);
        join_response.add_tensor_name(JOIN_TENSOR_NAME);
        responses.push_back(std::move(join_response));
        state.joined_size[index_controller] = 0;
      }
      FuseResponses(responses, state, response_list);
      response_list.set_shutdown(should_shut_down);
      LOG(TRACE, "ComputeResponseList() after FuseResponses, is_coordinator, setting should_shut_down = " << should_shut_down);

      // Broadcast final results to other ranks.
      SendFinalTensors(response_list);
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = TRUE, after SendFinalTensors";

    } else {
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, before SendReadyTensors and RecvFinalTensors";

      // TEST MPI_Gather 1
      auto recvcounts2 = new int[4];
      recvcounts2[0] = 0;
      auto sendcounts2 = new int[1];
      sendcounts2[0] = GetRank()*2;
      int retcode = MPI_Gather(sendcounts2, 1, MPI_INT, recvcounts2, 1, MPI_INT, RANK_ZERO, pMPIController->GetMpiContext().mpi_comm) ;
      if(retcode != MPI_SUCCESS)
	LOG(INFO, "MPIController::ComputeResponseList(), MPI_Gather TEST 3 Error !!! retoode = " << retcode);
      else
        LOG(INFO, "MPIController::ComputeResponseList(), MPI_Gather TEST 3 OK");
      for (int i = 0; i < 4; ++i)
        LOG(INFO, "MPIController::ComputeResponseList() TEST 3  recvcounts2[i] = " << recvcounts2[i]);
	
      RequestList message_list;
      message_list.set_shutdown(should_shut_down);
      LOG(TRACE, "ComputeResponseList() is_coordinator = FALSE , before message_queuue.empty() , setting should_shut_down = " << should_shut_down);
      while (!message_queue_tmp.empty()) {
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, Loop before add_request";
        message_list.add_request(message_queue_tmp.front());
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, Loop before pop_front";
        message_queue_tmp.pop_front();
        LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, Loop after pop_front";
      }

      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, before SendReadyTensors";
      // Send ready tensors to rank zero
      SendReadyTensors(message_list);

      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, before RecvFinalTensors";
      // Receive final tensors to be processed from rank zero
      RecvFinalTensors(response_list);
      LOG(TRACE) << "ComputeResponseList() is_coordinator_ = FALSE, after SendReadyTensors and RecvFinalTensors0";
    }
  }

  if (!response_list.responses().empty()) {
    LOG(TRACE) << "ComputeResponseList() !response_list.responses().empty()";
    std::string tensors_ready;
    for (const auto& r : response_list.responses()) {
      tensors_ready += r.tensor_names_string() + "; ";
    }
    LOG(TRACE) << "ComputeResponseList() Sending ready responses as " << tensors_ready;
  }

  // If need_communication is false, meaning no uncached message coming in,
  // thus no need to update cache.
  if (need_communication && response_cache_.capacity() > 0) {
    // All workers add supported responses to cache. This updates the cache
    // order consistently across workers.
    for (auto& response : response_list.responses()) {
      if ((response.response_type() == Response::ResponseType::ALLREDUCE ||
           response.response_type() == Response::ResponseType::ADASUM ||
           response.response_type() == Response::ResponseType::ALLTOALL) &&
          (int)response.devices().size() == GetSize()) {
        response_cache_.put(response, tensor_queue_, state.joined[index_controller]);
      }
    }
  }

  // Reassign cache bits based on current cache order.
  response_cache_.update_cache_bits();

  LOG(TRACE) << "ComputeResponseList() ending.";

  return response_list;
}

int64_t Controller::TensorFusionThresholdBytes() {
  int64_t proposed_fusion_threshold =
      parameter_manager_.TensorFusionThresholdBytes();

  // If the cluster is homogeneous,
  // adjust buffer size to make sure it is divisible by local_size to improve
  // performance for operations that perform local reductions by default such as Adasum.
  if (is_homogeneous_) {
    // Assume the worst-case data type float64, since if it is divisible with
    // float64, it will be divisible for other types too.

    // Ensuring that fusion buffer can hold a number of elements divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for performance
    int double_size = GetTypeSize(HOROVOD_FLOAT64);
    int64_t div = local_size_ * double_size * FUSION_BUFFER_ATOMIC_UNIT;
    return ((proposed_fusion_threshold + div - 1) / div) * div;
  }
  return proposed_fusion_threshold;
}

Response Controller::ConstructResponse(const std::string& name, int joined_size) {
  bool error = false;
  auto it = message_table_.find(name);
  assert(it != message_table_.end());

  std::vector<Request>& requests = it->second;
  assert(!requests.empty());

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being processed
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << DataType_Name(data_type)
                           << ", but another rank had type "
                           << DataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched operations: One rank did an "
                           << Request::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << Request::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::ADASUM ||
      message_type == Request::BROADCAST) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing an allreduce, check that prescaling and postscaling factors
  // are identical across ranks.
  double prescale_factor;
  double postscale_factor;
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::ADASUM) {
    prescale_factor = requests[0].prescale_factor();
    postscale_factor = requests[0].postscale_factor();

    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }
      double request_prescale_factor = requests[i].prescale_factor();
      double request_postscale_factor = requests[i].postscale_factor();

      if (prescale_factor != request_prescale_factor ||
          postscale_factor != request_postscale_factor) {
        error = true;
        error_message_stream
            << "Mismatched prescale and/or postscale factors: "
            << "One rank sent factors (" << prescale_factor
            << ", " << postscale_factor << "), but another rank "
            << "sent factors (" << request_prescale_factor
            << ", " << request_postscale_factor << ").";
        break;
      }
    }
  }

  std::vector<int64_t> tensor_sizes;
  if (message_type == Request::ALLGATHER ||
      message_type == Request::ALLTOALL) {
    if (joined_size > 0) {
      error = true;
      if (message_type == Request::ALLGATHER) {
        error_message_stream << "Allgather is not supported with Join at this time. "
                             << "Specify sparse_to_dense=True if using DistributedOptimizer";
      } else if (message_type == Request::ALLTOALL) {
        error_message_stream << "Alltoall is not supported with Join at this time.";
      }
    }

    // If we are doing an allgather/alltoall, make sure all but the first dimension are
    // the same. The first dimension may be different and the output tensor is
    // the sum of the first dimension. Collect the sizes by rank for allgather only.
    tensor_sizes.resize(requests.size());
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << Request::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
      tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
    }

    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of rank "
            << tensor_shape.dims()
            << ", but another rank sent a tensor of rank "
            << request_shape.dims() << ".";
        break;
      }

      bool dim_mismatch = false;
      for (int dim = 1; dim < tensor_shape.dims(); ++dim) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_message_stream
              << "Mismatched " << Request::RequestType_Name(message_type)
              << " tensor shapes: One rank sent a tensor with dimension " << dim
              << " equal to " << tensor_shape.dim_size(dim)
              << ", but another rank sent a tensor with dimension " << dim
              << " equal to " << request_shape.dim_size(dim) << ".";
          dim_mismatch = true;
          break;
        }
      }
      if (dim_mismatch) {
        break;
      }

      // Collect first dimension sizes for allgather to use for fusion and allgather op.
      if (message_type == Request::ALLGATHER) {
        tensor_sizes[requests[i].request_rank()] = request_shape.dim_size(0);
      }
    }
  }

  if (message_type == Request::ALLREDUCE || message_type == Request::ADASUM) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    tensor_sizes.push_back(tensor_shape.num_elements());
  }

  if (message_type == Request::BROADCAST) {
    if (joined_size > 0) {
      error = true;
      error_message_stream << "Broadcast is not supported with Join at this time.";
    }

    // If we are doing a broadcast, check that all root ranks are identical.
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << Request::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }
  std::vector<int32_t> devices(requests.size());
  for (auto& request : requests) {
    devices[request.request_rank()] = request.device();
  }

  Response response;
  response.add_tensor_name(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(Response::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == Request::ALLGATHER) {
    response.set_response_type(Response::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
  } else if (message_type == Request::ALLREDUCE) {
    response.set_response_type(Response::ALLREDUCE);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
    response.set_tensor_type(data_type);
    response.set_prescale_factor(prescale_factor);
    response.set_postscale_factor(postscale_factor);
  } else if (message_type == Request::BROADCAST) {
    response.set_response_type(Response::BROADCAST);
  } else if (message_type == Request::ALLTOALL) {
    response.set_response_type(Response::ALLTOALL);
  } else if (message_type == Request::ADASUM) {
    response.set_response_type(Response::ADASUM);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
    response.set_tensor_type(data_type);
    response.set_prescale_factor(prescale_factor);
    response.set_postscale_factor(postscale_factor);
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed response.
  message_table_.erase(it);
  stall_inspector_.RemoveUncachedTensor(name);

  return response;
}

void Controller::CoordinateCacheAndState(CacheCoordinator& cache_coordinator) {
  LOG(DEBUG,  "Controller::CoordinateCacheAndState entered.");
  // Sync cache and state information across workers.
  cache_coordinator.sync(shared_from_this(), timeline_enabled_);
  LOG(DEBUG,  "Controller::CoordinateCacheAndState, after cache_coordinator.sync.");

  // If invalid cache entries exist, erase associated entries.
  if (!cache_coordinator.invalid_bits().empty()) {
    for (auto bit : cache_coordinator.invalid_bits()) {
      response_cache_.erase_response(bit);
    }
  }
  LOG(DEBUG,  "Controller::CoordinateCacheAndState, after invalid_bits.empty.");

  if (timeline_enabled_) {
    // Start/continue negotiation phase on timeline bit entries.
    LOG(DEBUG,  "Controller::CoordinateCacheAndState, timeline_enabled");
    for (auto bit : cache_coordinator.timeline_bits()) {
      LOG(DEBUG,  "Controller::CoordinateCacheAndState, timeline_bits(), start loop");
      auto& response = response_cache_.peek_response(bit);
      LOG(DEBUG,  "Controller::CoordinateCacheAndState, timeline_bits(), after peek_response loop");
      timeline_.NegotiateStart(response.tensor_names()[0],
                               (Request::RequestType)response.response_type());
      LOG(DEBUG,  "Controller::CoordinateCacheAndState, timeline_bits(), after NegotiateStart loop");
    }

    // End negotiation phase for synced cache hit set entries.
    for (auto bit : cache_coordinator.cache_hits()) {
      LOG(DEBUG,  "Controller::CoordinateCacheAndState, cache_hits(), start loop");
      auto& response = response_cache_.peek_response(bit);
      LOG(DEBUG,  "Controller::CoordinateCacheAndState, cache_hits(), after peek_response loop");
      timeline_.NegotiateEnd(response.tensor_names()[0]);
      LOG(DEBUG,  "Controller::CoordinateCacheAndState, cache_hits(), after NegotiateEnd loop");
    }
  }
  LOG(DEBUG,  "Controller::CoordinateCacheAndState, ended");
}

void Controller::FuseResponses(std::deque<Response>& responses,
                               HorovodGlobalState& state,
                               ResponseList& response_list) {
  while (!responses.empty()) {

    auto response = responses.front();
    assert(response.tensor_names().size() == 1);
    responses.pop_front();
    int64_t tensor_size = 0;
    if (response.response_type() == Response::ResponseType::ALLREDUCE ||
        response.response_type() == Response::ResponseType::ADASUM) {
      // Attempt to add more responses to this fused response.

      tensor_size = response.tensor_sizes()[0] * GetTypeSize(response.tensor_type());
#if HAVE_CUDA
      if (state.batch_d2d_memcopies) {
        // Add 16 byte pad for batched memcpy op
        tensor_size = BATCHED_D2D_PADDING * ((tensor_size + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      }
#endif
      std::deque<Response> skipped_responses;
      int64_t skipped_size = 0;
      while (!responses.empty()) {
        auto& new_response = responses.front();
        assert(new_response.tensor_names().size() == 1);

        int64_t new_tensor_size = new_response.tensor_sizes().empty()
                                      ? 0
                                      : new_response.tensor_sizes()[0] *
                                        GetTypeSize(new_response.tensor_type());

#if HAVE_CUDA
        if (state.batch_d2d_memcopies) {
          // Add 16 byte pad for batched memcpy op
          new_tensor_size = BATCHED_D2D_PADDING * ((new_tensor_size + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
        }
#endif

        if (response.response_type() == new_response.response_type() &&
            response.devices() == new_response.devices() &&
            response.tensor_type() == new_response.tensor_type() &&
            tensor_size + new_tensor_size <= TensorFusionThresholdBytes() &&
            response.prescale_factor() == new_response.prescale_factor() &&
            response.postscale_factor() == new_response.postscale_factor()) {
          // These tensors will fuse together well.
          tensor_size += new_tensor_size;
          response.add_tensor_name(std::move(new_response.tensor_names()[0]));
          response.add_tensor_size(new_response.tensor_sizes()[0]);
          responses.pop_front();
        } else {
          // In general, don't try to fuse additional tensors since they are
          // usually computed in order of requests and skipping tensors may
          // mean that the batch will have to wait longer while skipped
          // tensors could be reduced at that time. However, mixed-precision
          // training may yield requests of various dtype in a mixed-up
          // sequence causing breakups in fusion. To counter this some look
          // ahead is allowed.

          skipped_size += new_tensor_size;
          if (tensor_size + skipped_size <= TensorFusionThresholdBytes()) {
            // Skip response and look ahead for more to fuse.
            skipped_responses.push_back(std::move(new_response));
            responses.pop_front();
          } else {
            break;
          }
        }
      }
      // Replace any skipped responses.
      while (!skipped_responses.empty()) {
        responses.push_front(std::move(skipped_responses.back()));
        skipped_responses.pop_back();
      }

    } else if (response.response_type() == Response::ResponseType::ALLGATHER) {
      // Attempt to add more responses to this fused response.
      const auto& entry =
          tensor_queue_.GetTensorEntry(response.tensor_names()[0]);

      // This is size of first dimension.
      int64_t total_byte_size_of_output =
          TotalByteSizeOfAllgatherOutput(response.tensor_sizes(), entry);

      std::deque<Response> skipped_responses;
      int64_t skipped_size = 0;
      while (!responses.empty()) {

        auto& new_response = responses.front();
        assert(new_response.tensor_names().size() == 1);
        const auto& new_entry =
            tensor_queue_.GetTensorEntry(new_response.tensor_names()[0]);

        int64_t new_total_byte_size_of_output = TotalByteSizeOfAllgatherOutput(
            new_response.tensor_sizes(), new_entry);

        if (response.response_type() == new_response.response_type() &&
            response.devices() == new_response.devices() &&
            entry.tensor->dtype() == new_entry.tensor->dtype() &&
            total_byte_size_of_output + new_total_byte_size_of_output <=
                TensorFusionThresholdBytes()) {

          // These tensors will fuse together well.
          total_byte_size_of_output += new_total_byte_size_of_output;
          response.add_allgather_response(new_response);
          responses.pop_front();

        } else {
          // In general, don't try to fuse additional tensors since they are
          // usually computed in order of requests and skipping tensors may
          // mean that the batch will have to wait longer while skipped
          // tensors could be reduced at that time. However, mixed-precision
          // training may yield requests of various dtype in a mixed-up
          // sequence causing breakups in fusion. To counter this some look
          // ahead is allowed.

          skipped_size += new_total_byte_size_of_output;
          if (total_byte_size_of_output + skipped_size <=
              TensorFusionThresholdBytes()) {
            // Skip response and look ahead for more to fuse.
            skipped_responses.push_back(std::move(new_response));
            responses.pop_front();
          } else {
            break;
          }
        }
      }

      // Replace any skipped responses.
      while (!skipped_responses.empty()) {
        responses.push_front(std::move(skipped_responses.back()));
        skipped_responses.pop_back();
      }
    }

    response_list.add_response(std::move(response));
    LOG(TRACE, "ComputeResponseList() Created response of size " << tensor_size);
  }
}

int64_t Controller::TotalByteSizeOfAllgatherOutput(
    const std::vector<int64_t>& tensor_sizes, const TensorTableEntry& entry) {
  int64_t total_dimension_size = 0;
  for (auto sz : tensor_sizes) {
    total_dimension_size += sz;
  }
  // Every tensor participating in Allgather operation may have
  // different first dimension size, but the rest of dimensions are same
  // for all tensors.  Here we get shape of tensor sliced by first
  // dimension. Allgather output will have shape of: (sum of first
  // dimension of every tensor) x (tensor slice shape).
  int64_t total_count_of_output_entries = total_dimension_size;
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    total_count_of_output_entries *= entry.tensor->shape().dim_size(i);
  }
  int element_size = GetTypeSize(entry.tensor->dtype());
  int64_t total_byte_size_of_output =
      total_count_of_output_entries * element_size;

  return total_byte_size_of_output;
}

int Controller::GetLocalSizeAtCrossRank(int i) {
  return local_sizes_for_cross_rank_[i];
}

bool Controller::IncrementTensorCount(const Request& msg, int joined_size) {
  auto& name = msg.tensor_name();
  auto table_iter = message_table_.find(name);
  if (table_iter == message_table_.end()) {
    LOG(DEBUG, "Controller::IncrementTensorCount() msg NOT found in message_tabla, creating messages"); 
    std::vector<Request> messages = {msg};
    messages.reserve(static_cast<unsigned long>(GetSize()));
    message_table_.emplace(name, std::move(messages));
    table_iter = message_table_.find(name);
    timeline_.NegotiateStart(name, msg.request_type());
  } else {
    LOG(DEBUG, "Controller::IncrementTensorCount() msg found in message_table"); 
    std::vector<Request>& messages = table_iter->second;
    messages.push_back(msg);
  }

  timeline_.NegotiateRankReady(name, msg.request_rank());

  std::vector<Request>& messages = table_iter->second;
  int count = (int)messages.size();
  LOG(DEBUG, "Controller::IncrementTensorCount() count = " << count);
  bool ready_to_reduce = count == (GetSize() - joined_size);
  LOG(DEBUG, "Controller::IncrementTensorCount() ready_to_reduce = " << ready_to_reduce);
  if (ready_to_reduce) {
    timeline_.NegotiateEnd(name);
  }
  return ready_to_reduce;
}

void Controller::SetTimelineEnabled(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_pending_ = value;
  timeline_enabled_ = value;
}

void Controller::SetTimelineEnabledPending(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_pending_ = value;
}

void Controller::SetMarkCyclesInTimelinePending(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  mark_cycles_in_timeline_pending_ = value;
}

void Controller::SynchronizeTimelineEnabled() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_ = timeline_enabled_pending_;
}

bool Controller::TimeLineEnabled() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return timeline_enabled_;
}

bool Controller::TimelineEnabledPending() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return timeline_enabled_pending_;
}

bool Controller::MarkCyclesInTimelinePending() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return mark_cycles_in_timeline_pending_;
}
} // namespace common
} // namespace horovod
