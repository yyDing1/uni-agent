#!/usr/bin/env bash
set -xeuo pipefail

project_name=${PROJECT_NAME:-"Uni-Agent-SWE-Agent"}
exp_name=${EXP_NAME:-"GRPO-Qwen3-30B-R2E-Sync"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Instruct-xml-template"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/data/swe_agent/runtime_env.yaml"}
# Must be launched from the repository root so Ray packages both `verl/` and `uni_agent/`.
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"examples/agent_interaction/agent_config.yaml"}

RAY_RUNTIME_ENV_ARGS=(--runtime-env "${RUNTIME_ENV}")
if [[ -n "${RUNTIME_ENV_JSON:-}" ]]; then
    RAY_RUNTIME_ENV_ARGS=(--runtime-env-json "${RUNTIME_ENV_JSON}")
fi

rollout_name=${ROLLOUT_NAME:-"vllm"} # sglang or vllm

# Algorithm parameters
adv_estimator=${ADV_ESTIMATOR:-grpo}

use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-3e-4}
clip_ratio_high=${CLIP_RATIO_HIGH:-4e-4}

# Response length parameters
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 4))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 64))}
max_model_len=$((max_prompt_length + max_response_length))
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}  # unused
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
loss_mode=${LOSS_MODE:-gspo}

# Algorithm
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# Performance Related Parameter
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-4}
train_tp=${TRAIN_TP:-4}
train_pp=${TRAIN_PP:-1}
train_cp=${TRAIN_CP:-4}
train_ep=${TRAIN_EP:-8}
train_etp=${TRAIN_ETP:-1}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$((max_model_len / train_cp))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$((max_model_len / train_cp))}

optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
overlap_cpu_optimizer_d2h_h2d=${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-True}
use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER:-True}
optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD:-True}
recompute_method=${RECOMPUTE_METHOD:-}
recompute_granularity=${RECOMPUTE_GRANULARITY:-}
recompute_num_layers=${RECOMPUTE_NUM_LAYERS:-}

RECOMPUTE_ARGS=()
if [[ -n "${recompute_method}" ]]; then
    RECOMPUTE_ARGS+=(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method="${recompute_method}")
fi
if [[ -n "${recompute_granularity}" ]]; then
    RECOMPUTE_ARGS+=(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity="${recompute_granularity}")
fi
if [[ -n "${recompute_num_layers}" ]]; then
    RECOMPUTE_ARGS+=(+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers="${recompute_num_layers}")
fi

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

# Sync training parameters
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-64}
agent_num_workers=${AGENT_NUM_WORKERS:-8}
reward_num_workers=${REWARD_NUM_WORKERS:-8}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-8}
test_freq=${TEST_FREQ:-10}
total_epochs=${TOTAL_EPOCHS:-20}
save_freq=${SAVE_FREQ:--1}
resume_mode=${RESUME_MODE:-auto}
log_val_generations=${LOG_VAL_GENERATIONS:-10}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}
logger=${LOGGER:-"['console','tensorboard']"}
actor_checkpoint_save_contents=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-}
actor_checkpoint_load_contents=${ACTOR_CHECKPOINT_LOAD_CONTENTS:-}

CHECKPOINT_ARGS=()
if [[ -n "${actor_checkpoint_save_contents}" ]]; then
    CHECKPOINT_ARGS+=(actor_rollout_ref.actor.checkpoint.save_contents="${actor_checkpoint_save_contents}")
fi
if [[ -n "${actor_checkpoint_load_contents}" ]]; then
    CHECKPOINT_ARGS+=(actor_rollout_ref.actor.checkpoint.load_contents="${actor_checkpoint_load_contents}")
fi

ray job submit --no-wait "${RAY_RUNTIME_ENV_ARGS[@]}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.dataloader_num_workers=${dataloader_num_workers} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${max_model_len} \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=${overlap_cpu_optimizer_d2h_h2d} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${use_precision_aware_optimizer} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${optimizer_cpu_offload} \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    "${RECOMPUTE_ARGS[@]}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    "${CHECKPOINT_ARGS[@]}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=${agent_num_workers} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
    reward.reward_manager.name=dapo \
    reward.num_workers=${reward_num_workers} \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=${logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=${resume_mode} \
    trainer.log_val_generations=${log_val_generations} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}"
