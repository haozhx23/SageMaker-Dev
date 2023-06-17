import glob
import logging
import os
import sys
from typing import Dict, Union

# import deepspeed
import hydra
import torch
import wandb
# from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import DictConfig, OmegaConf
# from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)

from general_util.evaluator import evaluate_fn as evaluate
from general_util.logger import setting_logger
from general_util.sagemaker_training_utils import batch_to_device, unwrap_model, note_best_checkpoint, \
    load_and_cache_examples, _load_and_cache_examples, set_seed_int, \
    initialize_optimizer, initialize_lr_scheduler

from transformers import set_seed

from smdistributed.modelparallel.torch.nn import FusedLayerNorm as LayerNorm
import smdistributed.modelparallel.torch as smp
from torch import optim
from general_util.tokenization_utils import expand_special_tokenizer
# from learning_rates import AnnealingLR
from transformers import get_linear_schedule_with_warmup

# torch.backends.cuda.matmul.allow_tf32 = True

@smp.step
def forward_step(model, inputs: Dict[str, torch.Tensor]):
    # print(f'''--DD-- In @smp.step forward_step()''')
    # print(f'''--DD-- In @smp.step inputs {inputs}''')
    outputs = model(**inputs)

    if isinstance(outputs, tuple):
        loss = outputs[0]
    else:
        loss = outputs["loss"]
        # print(f'''--DD-- loss = outputs["loss"] {loss}''')

    model.backward(loss)
    # model.step()

    print(f'''--DD-- In @smp.step loss is {loss}''')
    return loss, outputs

def train(cfg, model, tokenizer, optimizer, lr_scheduler, device, continue_from_global_step=0):
    model.train()
    dp_rank = smp.dp_rank() if not cfg.smp_init_params.prescaled_batch else smp.rdp_rank()
    dp_size = smp.dp_size() if not cfg.smp_init_params.prescaled_batch else smp.rdp_size()

    if "_target_" in cfg.train_file:
        files = hydra.utils.instantiate(cfg.train_file)
    elif os.path.exists(cfg.train_file):
        files = [cfg.train_file]
    else:
        files = list(glob.glob(cfg.train_file))

    # m = model.get_module()

    if continue_from_global_step > 0:
        # logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
        model.load_checkpoint(cfg.resume)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=smp.local_rank() not in [-1, 0])
    set_seed(cfg.seed)

    def _grad_accumulation_boundary(batch_idx):
        return batch_idx % cfg.gradient_accumulation_steps == cfg.gradient_accumulation_steps - 1

    for epoch in train_iterator:
        print(f'''-DD- epoch...{epoch}.''')
        for _file in files:
            print(f'''-DD- _file...{_file}.''')
            sub_train_dataset = _load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            print(f'''-DD- sub_train_dataset return rank : {smp.local_rank()}...''')

            _train_sampler = DistributedSampler(sub_train_dataset,
                                                shuffle=True,
                                                seed=cfg.seed,
                                                rank=dp_rank,
                                                num_replicas=dp_size,
                                                drop_last=True,
                                                )

            # _train_sampler = DistributedSampler(sub_train_dataset)
            _train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None

            sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                            sampler=_train_sampler,
                                            collate_fn=_train_collator,
                                            batch_size=cfg.per_gpu_train_batch_size,
                                            num_workers=0,
                                            pin_memory=True,
                                            drop_last=True,
                                            )

            dataloader_iterator = tqdm(sub_train_dataloader, desc="Iteration", disable=smp.local_rank() not in [-1, 0],
                                  dynamic_ncols=True)

            # print(f'''-DD- dataloader_iterator...{dataloader_iterator}.''')
            sub_train_dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(dataloader_iterator):
                # model.train()
                print(f'''--DD-- before batch_to_device - {device}''')
                batch = batch_to_device(batch, device)
                # optimizer.zero_grad()  ########

                if _grad_accumulation_boundary(step - 1):
                    optimizer.zero_grad(set_to_none=True)

                loss, outputs = forward_step(model, batch)
                print(f'''-DD- return forward_step loss {loss}''')
                # loss /= cfg.gradient_accumulation_steps
                tr_loss += loss.reduce_mean()

                # optimizer.step()  ########
                # tr_loss += loss
                print(f'''-DD- tr_loss {tr_loss}''')

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    global_step += 1
                    print(f'''-DD- global_step {global_step}''')

                if _grad_accumulation_boundary(step):
                    if cfg.smp_init_params.fp16:
                        optimizer.clip_master_grads(cfg.smp_params.grad_clip)

                    # # optimizer.step()
                    # if not (cfg.smp_init_params.fp16 and optimizer.overflow):
                    #     lr_scheduler.step()

                # Save model checkpoint
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    print(f'''-DD- before save ckpt mkdir''')
                    output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                    if smp.local_rank() in [-1, 0] and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    # TODO
                    print(f'''-DD- TODO save model''')
                    # save_model(model, cfg, output_dir, tokenizer)

                del batch

                if 0 < cfg.smp_params.max_steps < global_step:
                    dataloader_iterator.close()
                    break
            if 0 < cfg.smp_params.max_steps < global_step:
                dataloader_iterator.close()
                break
        if 0 < cfg.smp_params.max_steps < global_step:
            dataloader_iterator.close()
            break

        return global_step, tr_loss / (global_step + 1)



def main():
    cfg = OmegaConf.load('smp_llama_7b_zh_instruct_coig.yaml')

    # smp.init(smp_config)
    smp.init()

    # Set seed
    set_seed(cfg.seed)

    if smp.local_rank() == 0:
        os.system('chmod +x ./s5cmd')
        os.system('./s5cmd sync s3://llm-artifacts-us-east-1/decapoda-research-llama-7b-hf/* /tmp/llama_pretrain/')

    smp.barrier()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    expand_special_tokenizer(tokenizer)

    if cfg.smp_init_params.fp16 and cfg.smp_init_params.bf16:
        raise ValueError("FP16 and BF16 cannot be simultaneously enabled.")
    elif cfg.smp_init_params.fp16:
        dtype = torch.float16
    elif cfg.smp_init_params.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.get_default_dtype()

    with smp.model_creation(
            tensor_parallelism=smp.tp_size() > 1 or cfg.smp_params.use_distributed_transformer,
            dtype=dtype,  # dtype=torch.float16 if args.fp16 else torch.get_default_dtype()
            attention_in_fp32=cfg.smp_params.attention_in_fp32,
            query_key_layer_scaling=cfg.smp_params.query_key_layer_scaling and cfg.smp_init_params.bf16==False,
            fused_softmax=cfg.smp_params.fused_softmax,
            fused_dropout=cfg.smp_params.fused_dropout,
            fused_bias_gelu=cfg.smp_params.fused_bias_gelu,
    ):
        print('--DD-- smp.model_creation')
        model = hydra.utils.call(cfg.model, cfg.model_name_or_path)

        # TODO
        # tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        # expand_special_tokenizer(tokenizer)

    # print('--DD-- smp.model_creation')
    # model = hydra.utils.call(cfg.model, cfg.model_name_or_path)

    torch.cuda.set_device(smp.local_rank())
    # device = torch.device("cuda")
    device = str(torch.device("cuda", smp.local_rank()))

    if not cfg.smp_params.same_seed:
        # Set seed by tp_rank to prevent weights from being the same on different tp_ranks
        set_seed(cfg.seed + smp.tp_rank())

    model = smp.DistributedModel(model, trace_device="gpu", backward_passes_per_step=cfg.gradient_accumulation_steps)
    # TODO
    # tokenizer = smp.DistributedModel()

    m = model.get_module()
    # # m = model.model
    # if cfg.smp_params.use_distributed_transformer:
    #     transformer_layers = m.transformer.seq_layers
    # else:
    #     transformer_layers = m.transformer.h
    #
    # if cfg.smp_params.manual_partition:
    #     # evenly distribute layers across all partitions
    #     div, rem = divmod(cfg.smp_params.num_layers, smp.pp_size())
    #     get_num_layers = lambda x: (div + 1 if x >= smp.pp_size() - rem else div)
    #
    #     assignments = []
    #     for pp_rank in range(smp.pp_size()):
    #         nl = get_num_layers(pp_rank)
    #         print(f"-DD- {nl} layers assigned to partition {pp_rank}")
    #         assignments += [pp_rank for _ in range(nl)]
    #
    #     for i, c in enumerate(transformer_layers.children()):
    #         smp.set_partition(c, assignments[i])

    def _get_param_groups_by_weight_decay(module):
        weight_decay_params = {"params": []}
        no_weight_decay_params = {"params": [], "weight_decay": 0.0}
        param_ids = set()
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                for p in list(module_._parameters.values()):
                    if p is not None and id(p) not in param_ids:
                        no_weight_decay_params["params"].append(p)
                        param_ids.add(id(p))
            else:
                for n, p in list(module_._parameters.items()):
                    if p is not None and n != "bias" and id(p) not in param_ids:
                        weight_decay_params["params"].append(p)
                        param_ids.add(id(p))
                for n, p in list(module_._parameters.items()):
                    if p is not None and n == "bias" and id(p) not in param_ids:
                        no_weight_decay_params["params"].append(p)
                        param_ids.add(id(p))
        return weight_decay_params, no_weight_decay_params

    param_groups = _get_param_groups_by_weight_decay(m)

    optimizer = optim.AdamW(
        param_groups, betas=(cfg.smp_params.beta1, cfg.smp_params.beta2), lr=cfg.smp_params.lr,
        weight_decay=cfg.smp_params.weight_decay
    )

    # if cfg.smp_params.activation_checkpointing:
    #     if cfg.smp_params.use_distributed_transformer or smp.tp_size() > 1:
    #         if cfg.smp_params.checkpoint_sublayers:
    #             for c in transformer_layers.children():
    #                 smp.set_activation_checkpointing(c.attention)
    #                 smp.set_activation_checkpointing(c.output)
    #         else:
    #             smp.set_activation_checkpointing(transformer_layers, strategy=cfg.smp_params.activation_strategy)
    #     else:
    #         for c in transformer_layers.children():
    #             if cfg.smp_params.checkpoint_sublayers:
    #                 smp.set_activation_checkpointing(c.attn)
    #                 smp.set_activation_checkpointing(c.mlp)
    #             else:
    #                 smp.set_activation_checkpointing(c)

    optimizer = smp.DistributedOptimizer(
        optimizer,
        static_loss_scale=None,
        dynamic_loss_scale=True,
        dynamic_loss_args={"scale_window": 1000, "min_scale": 1, "delayed_shift": 2},
    )

    # def _get_learning_rate_scheduler(optimizer):
    #     # Add linear learning rate scheduler.
    #     if cfg.smp_params.lr_decay_iters is not None:
    #         num_iters = cfg.smp_params.lr_decay_iters
    #     else:
    #         num_iters = cfg.smp_params.max_steps
    #     num_iters = max(1, num_iters)
    #     init_step = 0
    #     warmup_iter = cfg.smp_params.warmup * num_iters
    #     plateau_iter = warmup_iter + cfg.smp_params.plateau * num_iters
    #     lr_scheduler = AnnealingLR(
    #         optimizer,
    #         start_lr=cfg.smp_params.lr,
    #         warmup_iter=warmup_iter,
    #         plateau_iter=plateau_iter,
    #         total_iters=num_iters,
    #         decay_style=cfg.smp_params.lr_decay_style,
    #         last_iter=init_step,
    #         min_lr=cfg.smp_params.min_lr,
    #         use_checkpoint_lr_scheduler=cfg.smp_params.load_partial or cfg.smp_params.load_full,
    #         override_lr_scheduler=False,
    #     )
    #
    #     return lr_scheduler
    # lr_scheduler = _get_learning_rate_scheduler(optimizer)

    lr_scheduler = None

    # warmup_iter = cfg.smp_params.warmup * cfg.smp_params.max_steps
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_iter, cfg.smp_params.max_steps)

    # # load after wrapping model and optimizer with smp Distributed...
    # if cfg.smp_params.load_full or cfg.smp_params.load_partial:
    #     if cfg.smp_params.load_partial and cfg.smp_params.load_full:
    #         print(
    #             "Since both --load_partial and --load_full set, will try to load from full checkpoint."
    #             "If the intention is to load from partial checkpoint, please don't set --load_full"
    #         )
    #     partial = not cfg.smp_params.load_full
    #     path = cfg.checkpoint_path if partial else cfg.model_name_or_path ##########################
    #     tag = None if partial else "fullmodel.pt"
    #     user_content = smp.resume_from_checkpoint(path, tag=tag, partial=partial)
    #     total_steps = user_content["total_steps"] if partial else 0
    #     start_train_path_index = user_content.get("start_train_path_index", 0)
    #     start_batch_index = user_content.get("start_batch_index", 0)
    #     if "lr_scheduler" in user_content:
    #         lr_scheduler.load_state_dict(user_content["lr_scheduler"])
    # else:
    #     total_steps = 0
    #     start_train_path_index = 0
    #     start_batch_index = 0


    if smp.rank() in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))
        print('--DD-- after OmegaConf.save() to training_config.yaml')

    if cfg.do_train:
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        if os.path.exists(cfg.output_dir) and getattr(cfg, "resume", None):
            checkpoint = cfg.resume
            # logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
            continue_from_global_step = int(checkpoint.split('-')[-1])

        #######
        print('--DD-- before main.train() ..')
        global_step, tr_loss = train(cfg, model, tokenizer, optimizer, lr_scheduler, device, continue_from_global_step)
        print(f'''--DD-- global_step = {global_step}, average loss = {tr_loss}''')
        #######

    smp.barrier()
    if smp.rank() == 0:
        print("--DD-- SMP training finished successfully")


if __name__ == "__main__":

    # os.system('chmod +x ./s5cmd')
    # os.system('./s5cmd sync s3://llm-artifacts-us-east-1/decapoda-research-llama-7b-hf/* /tmp/llama_pretrain/')

    main()