"""torchtitan config: a Qwen3-30B-A3B-style MoE trained from scratch with expert parallelism.

Uses the Qwen3-30B-A3B architecture (128 experts, 8 active per token) with fewer layers so it
fits one 8xH100 node. Experts are split across GPUs (expert parallelism, EP); the rest of the
model is sharded with FSDP. Settings come from environment variables set in workload.yaml.
"""

import os

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def qwen3_moe_ep() -> Trainer.Config:
    seq_len = int(os.environ["SEQ_LEN"])
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,  # FSDP over all GPUs
        expert_parallel_degree=int(os.environ["EXPERT_PARALLEL_DEGREE"]),
        tensor_parallel_degree=1,
    )
    # FlexAttention is built into PyTorch; the default "varlen" backend needs FlashAttention 3.
    model = build_model_config(
        "30B-A3B", seq_len=seq_len, parallelism=parallelism, attn_backend="flex"
    )
    # Keep the 30B-A3B layer shape (attention, 128 experts, top-8 routing) but fewer layers.
    model.layers = model.layers[: int(os.environ["NUM_LAYERS"])]
    tokens_per_microbatch = int(os.environ["MICRO_BATCH_SIZE"]) * seq_len
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        ),
        hf_assets_path=os.environ["HF_ASSETS_PATH"],
        metrics=MetricsProcessor.Config(log_freq=10),
        model=model,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=float(os.environ["LEARNING_RATE"]))]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens_per_microbatch,
            max_context_length=seq_len,
            steps=int(os.environ["STEPS"]),
            # The default all-to-all expert dispatcher syncs with the CPU.
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )
