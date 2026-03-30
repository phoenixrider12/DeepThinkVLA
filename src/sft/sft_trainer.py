import os
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, PreTrainedModel, GenerationConfig
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    logging,
    PREFIX_CHECKPOINT_DIR,
    is_peft_available,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)
from termcolor import colored
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import Dataset
from typing import Optional, List
logger = logging.get_logger(__name__)
from torch.utils.data import DataLoader, Dataset
from transformers.integrations.deepspeed import deepspeed_init
import time
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import safetensors

from sft.utils import get_peft_state_non_lora_maybe_zero_3
from sft.modeling_deepthinkvla import get_actions_mask_cot

def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy

def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss

class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


def _get_cosine_schedule_with_warmup_lr_lambda_align_jax(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0, init_value_rate: float = 0.0
):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        factor = 1- init_value_rate
        return factor * progress + init_value_rate
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup_align_jax(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
):
    init_value = optimizer.defaults["lr"] / (num_warmup_steps + 1)
    init_value_rate = init_value/optimizer.defaults["lr"]
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda_align_jax,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        init_value_rate=init_value_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class DeepThinkVLATrainer(Trainer):
    def __init__(self, processor, action_tokenizer, *args, **kwargs):
        super(DeepThinkVLATrainer, self).__init__(*args, **kwargs)
        self.processor = processor
        self.action_tokenizer = action_tokenizer

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["vision_tower"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name and "multi_modal_projector" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "multi_modal_projector" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        num_learnable_params = sum(p.numel() for p in opt_model.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in opt_model.parameters())
        print(colored(f"Number of learnable parameters: {num_learnable_params}", "yellow"))
        print(colored(f"Number of total parameters: {num_total_params}", "yellow"))

        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup_align_jax(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
            
        enable_action_grads = getattr(self.args, "log_action_gradients", False)
        embeds_list = []
        handle = None
        if enable_action_grads:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "get_input_embeddings"):
                embed_layer = unwrapped_model.get_input_embeddings()
                def fw_hook(module, inputs_orig, output):
                    embeds_list.append(output)
                handle = embed_layer.register_forward_hook(fw_hook)

        outputs = model(**inputs, output_attentions=True)
        
        if handle is not None:
            handle.remove()
            
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        ##############################################################################################################
        # Add deepthinkvla
        ground_truth_token_ids = inputs["labels"][:, 1:]
        all_actions_mask = get_actions_mask_cot(
            ground_truth_token_ids,
            action_token_begin_idx=model.module.config.action_token_begin_idx,
            action_token_end_idx=model.module.config.action_token_end_idx,
            ignore_index=model.module.config.ignore_index,
        )
        predict_token_mask = ground_truth_token_ids != model.module.config.ignore_index

        predicted_token_ids = outputs.logits[:, :-1].argmax(dim=-1)
        action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=all_actions_mask
        )
        action_l1_loss = compute_actions_l1_loss(
            self.action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=all_actions_mask
        )
        predict_token_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=predict_token_mask
        )
        metrics ={
                "action_accuracy": action_accuracy.item(),
                "action_l1_loss": action_l1_loss.item(),
                "predict_token_accuracy": predict_token_accuracy.item(),
                "ce_loss": loss.item(),
            }

        # Calculate attention from action tokens to various modalities
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            layer_attns = outputs.attentions[-1]
            avg_layer_attn = layer_attns.mean(dim=1)
            
            _labels = labels if labels is not None else inputs.get("labels")
            input_ids = inputs.get("input_ids")
            if _labels is not None and input_ids is not None:
                config = model.module.config if hasattr(model, "module") else model.config
                target_model = model.module if hasattr(model, "module") else model
                pad_token_id = target_model.pad_token_id
                image_token_index = getattr(config, "image_token_index", None)

                ignore_idx = config.ignore_index
                begin_idx = config.action_token_begin_idx
                end_idx = config.action_token_end_idx
                eos_id = config.eos_token_id
                
                # 1. CoT Mask
                is_not_ignore = _labels != ignore_idx
                is_not_action = (_labels < begin_idx) | (_labels > end_idx)
                is_not_special = (_labels != 257156) & (_labels != eos_id)
                cot_mask = is_not_ignore & is_not_action & is_not_special
                
                # 2. Vision and Text Input Masks (Prompt)
                is_prompt = _labels == ignore_idx
                is_pad = input_ids == pad_token_id if pad_token_id is not None else torch.zeros_like(is_prompt)
                is_prompt_true = is_prompt & (~is_pad)
                
                if image_token_index is not None:
                    image_mask = is_prompt_true & (input_ids == image_token_index)
                else:
                    image_mask = torch.zeros_like(is_prompt)
                text_mask = is_prompt_true & (~image_mask)
                
                # 3. Action Mask
                action_token_mask = (input_ids >= begin_idx) & (input_ids <= end_idx)
                
                cot_scores, image_scores, text_scores, action_scores = [], [], [], []
                
                for b in range(avg_layer_attn.shape[0]):
                    queries = torch.nonzero(all_actions_mask[b]).squeeze(-1)
                    cot_keys = torch.nonzero(cot_mask[b]).squeeze(-1)
                    img_keys = torch.nonzero(image_mask[b]).squeeze(-1)
                    txt_keys = torch.nonzero(text_mask[b]).squeeze(-1)
                    act_keys = torch.nonzero(action_token_mask[b]).squeeze(-1)
                    
                    if len(queries) > 0:
                        if len(cot_keys) > 0:
                            cot_scores.append(avg_layer_attn[b][queries][:, cot_keys].sum(dim=-1).mean().item())
                        if len(img_keys) > 0:
                            image_scores.append(avg_layer_attn[b][queries][:, img_keys].sum(dim=-1).mean().item())
                        if len(txt_keys) > 0:
                            text_scores.append(avg_layer_attn[b][queries][:, txt_keys].sum(dim=-1).mean().item())
                        if len(act_keys) > 0:
                            action_scores.append(avg_layer_attn[b][queries][:, act_keys].sum(dim=-1).mean().item())
                        
                if cot_scores:
                    metrics["action_to_cot_attention"] = sum(cot_scores) / len(cot_scores)
                if image_scores:
                    metrics["action_to_vision_attention"] = sum(image_scores) / len(image_scores)
                if text_scores:
                    metrics["action_to_text_attention"] = sum(text_scores) / len(text_scores)
                if action_scores:
                    metrics["action_to_action_attention"] = sum(action_scores) / len(action_scores)

        self.log(metrics)
        ##############################################################################################################
        # Counterfactual Regularization Loss (Divergence & Entropy)
        enable_div = getattr(self.args, "enable_divergence_loss", False)
        enable_ent = getattr(self.args, "enable_entropy_loss", False)

        if enable_div or enable_ent:
            with torch.enable_grad():
                perturbed_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                _labels = inputs.get("labels")
                if _labels is not None and "input_ids" in perturbed_inputs:
                    perturbed_input_ids = perturbed_inputs["input_ids"]
                    
                    config = model.module.config if hasattr(model, "module") else model.config
                    ignore_idx = config.ignore_index
                    begin_idx = config.action_token_begin_idx
                    end_idx = config.action_token_end_idx
                    eos_id = config.eos_token_id
                    
                    # Define CoT: not ignored, not actions, not special endings
                    is_not_ignore = _labels != ignore_idx
                    is_not_action = (_labels < begin_idx) | (_labels > end_idx)
                    is_not_special = (_labels != 257156) & (_labels != eos_id)
                    cot_mask = is_not_ignore & is_not_action & is_not_special
                    
                    perturb_type = getattr(self.args, "cot_perturbation_type", "shuffle")
                    
                    for b in range(perturbed_input_ids.shape[0]):
                        row_mask = cot_mask[b]
                        if row_mask.sum() > 0:
                            if perturb_type == "shuffle":
                                cot_tokens = perturbed_input_ids[b, row_mask]
                                shuffled_indices = torch.randperm(cot_tokens.shape[0])
                                perturbed_input_ids[b, row_mask] = cot_tokens[shuffled_indices]
                            elif perturb_type == "random":
                                vocab_size = config.text_config.vocab_size
                                random_tokens = torch.randint(0, vocab_size, (row_mask.sum(),), device=perturbed_input_ids.device)
                                perturbed_input_ids[b, row_mask] = random_tokens
                    
                    # Forward pass with perturbed inputs
                    perturbed_outputs = model(**perturbed_inputs)
                    
                    true_logits = outputs.logits[:, :-1, :]
                    pert_logits = perturbed_outputs.logits[:, :-1, :]
                    
                    # We use all_actions_mask from above (already computed)
                    true_action_logits = true_logits[all_actions_mask] 
                    pert_action_logits = pert_logits[all_actions_mask] 
                    
                    if true_action_logits.shape[0] > 0:
                        cf_loss = 0.0
                        
                        if enable_div:
                            true_probs = F.softmax(true_action_logits, dim=-1).detach()
                            pert_log_probs = F.log_softmax(pert_action_logits, dim=-1)
                            
                            kl_div = F.kl_div(pert_log_probs, true_probs, reduction='batchmean')
                            div_weight = getattr(self.args, "divergence_loss_weight", 0.1)
                            margin = 5.0
                            div_penalty = torch.clamp(margin - kl_div, min=0.0)
                            div_loss = div_weight * div_penalty
                            cf_loss += div_loss
                            metrics["div_loss"] = div_loss.item()
                            metrics["kl_div"] = kl_div.item()
                            
                        if enable_ent:
                            vocab_size_act = pert_action_logits.shape[-1]
                            pert_probs = F.softmax(pert_action_logits, dim=-1)
                            pert_log_probs = F.log_softmax(pert_action_logits, dim=-1)
                            
                            entropy = -torch.sum(pert_probs * pert_log_probs, dim=-1).mean()
                            ent_weight = getattr(self.args, "entropy_loss_weight", 0.1)
                            
                            max_ent = math.log(vocab_size_act)
                            ent_penalty = max_ent - entropy
                            ent_loss = ent_weight * ent_penalty
                            cf_loss += ent_loss
                            metrics["ent_loss"] = ent_loss.item()
                            metrics["entropy"] = entropy.item()
                            
                        loss = loss + cf_loss
                        metrics["cf_loss"] = cf_loss.item() if isinstance(cf_loss, torch.Tensor) else cf_loss
                        
        if enable_action_grads and len(embeds_list) > 0:
            embeds = embeds_list[0]
            true_action_logits = outputs.logits[:, :-1, :][all_actions_mask]
            
            if true_action_logits.shape[0] > 0:
                action_logits_sum = true_action_logits.sum()
                try:
                    grads = torch.autograd.grad(
                        outputs=action_logits_sum,
                        inputs=embeds,
                        retain_graph=True,
                        create_graph=False,
                        only_inputs=True,
                        allow_unused=True
                    )[0]
                    
                    if grads is not None:
                        _labels = inputs.get("labels")
                        config = model.module.config if hasattr(model, "module") else model.config
                        ignore_idx = config.ignore_index
                        begin_idx = config.action_token_begin_idx
                        end_idx = config.action_token_end_idx
                        eos_id = config.eos_token_id
                        
                        prompt_mask = (_labels == ignore_idx)
                        is_not_ignore = _labels != ignore_idx
                        is_not_action = (_labels < begin_idx) | (_labels > end_idx)
                        is_not_special = (_labels != 257156) & (_labels != eos_id)
                        cot_mask = is_not_ignore & is_not_action & is_not_special
                        
                        grad_norms = torch.norm(grads, p=2, dim=-1)
                        
                        prompt_grad_norm = grad_norms[prompt_mask].mean()
                        cot_grad_norm = grad_norms[cot_mask].mean()
                        
                        metrics["prompt_grad_norm"] = prompt_grad_norm.item() if not torch.isnan(prompt_grad_norm) else 0.0
                        metrics["cot_grad_norm"] = cot_grad_norm.item() if not torch.isnan(cot_grad_norm) else 0.0
                except RuntimeError as e:
                    logger.warning(f"Failed to compute action gradients: {e}")

        metrics["total_loss"] = loss.item()
        self.log(metrics)
        ##############################################################################################################

        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(DeepThinkVLATrainer, self)._save_checkpoint(model, trial)

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

