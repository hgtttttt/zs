

import re
import json
import logging
import random
import os

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Tuple

import wandb
from accelerate.utils import save_fsdp_model
from peft import LoraConfig, TaskType, get_peft_model, AdaLoraConfig, LoHaConfig, LoKrConfig, PeftModel
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils import is_peft_available, add_start_docstrings_to_model_forward, replace_return_docstrings


from preprocess import get_dataset, BaseProcessor

from tqdm import tqdm

import datasets
# import evaluate
# import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from datasets import Dataset
from filelock import FileLock

import torch

import transformers
from transformers import (
    AutoConfig,
    LlamaConfig, GenerationConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    LlamaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed, LlamaForCausalLM, MistralForCausalLM,
    AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForSeq2Seq, Trainer, TrainingArguments, TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer_callback import TrainerCallback

# from myloralib.prompt_pattern import PROMPT, STOP_WORD

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    method: Optional[str] = field(
        default="lora",
        metadata={
            "help": "you can choose follow method: ft, lora, adapter, adalora, bslora"
        }
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA r"},
    )
    lora_num: Optional[int] = field(
        default=3
    )
    res_flag: Optional[int] = field(
        default=0
    )
    mul_flag: Optional[int] = field(
        default=0
    )
    merge_flag: Optional[int] = field(
        default=0
    )
    pre_num: Optional[int] = field(
        default=0
    )
    target_module: Optional[str] = field(
        default="q_proj.v_proj"
    )
    prompt_loss: Optional[float] = field(
        default=0,
    )
    flag_reset: Optional[int] = field(
        default=0
    )
    flag_reset_lr: Optional[int] = field(
        default=0
    )



@dataclass
class DataTrainingArguments:
    debug_flag: Optional[int] = field(default=0, metadata={"help": "whether or not use wandb"})
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization for source text."},
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization for target text."},
    )
    result_path: Optional[str] = field(
        default="temp_result",
        metadata={"help": "The path to save result."},
    )
    task_name: Optional[str] = field(
        default="gsm8k",
        metadata={"help": "The name of the task to train"},
    )
    sample_flag: Optional[int] = field(
        default=1,
        metadata={"help": "whether or not sample the dataset"}
    )

# class CustomCallback(TrainerCallback):
#     def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

# class BsTrainer(Seq2SeqTrainer):
#     def prediction_step(
#         self,
#         model: nn.Module,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#         **gen_kwargs,
#     ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         inputs = self._prepare_inputs(inputs)
#         print(inputs)
#         exit(-1)



class EvalEpochIntervalCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):
        global total_epoch
        total_epoch += 1
        epoch = round(state.epoch)

        if (epoch % 5 == 0):
            control.should_save = True
        else:
            control.should_save = False

        if (args.logging_strategy == IntervalStrategy.EPOCH):
            control.should_log = True

        control.should_evaluate = True

        return control


class BsTrainer(Seq2SeqTrainer):
    def __init__(self, *args, bsmodel=None, global_rank=-1, prompt_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.bsmodel = bsmodel
        self.global_rank = global_rank
        # self.prompt_loss = prompt_loss
        # self.prompt_loss_weight = 0.1

    def save_model(self, *args, **kwargs):
        if self.bsmodel.bsconfig.res_flag > 0:
            if (self.bsmodel.bsconfig.res_flag == 1 or self.bsmodel.bsconfig.res_flag == 3) and self.bsmodel.bsconfig.merge_flag > 0:
                self.bsmodel.calculate_froc()
                assert self.bsmodel.bsconfig.lora_merge_alpha is not None
            self.bsmodel.unconcat_loras()
            super().save_model(*args, **kwargs)
            if self.global_rank == 0:
                if len(args) > 0:
                    temp_save_path = args[0]
                else:
                    temp_save_path = self.args.output_dir
                with open(f"{temp_save_path}/bslora_config.json", 'w') as f:
                    json.dump(self.bsmodel.bsconfig.to_json_string(), f)
            self.bsmodel.concat_loras()
        else:
            super().save_model(*args, **kwargs)
            if self.global_rank == 0:
                if len(args) > 0:
                    temp_save_path = args[0]
                else:
                    temp_save_path = self.args.output_dir
                with open(f"{temp_save_path}/bslora_config.json", 'w') as f:
                    json.dump(self.bsmodel.bsconfig.to_json_string(), f)

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        if (self.bsmodel.bsconfig.res_flag == 1 or self.bsmodel.bsconfig.res_flag == 3) and self.bsmodel.bsconfig.merge_flag > 0:
            self.bsmodel.calculate_froc()
        return super().evaluate(*args, **kwargs)
        # else:
        #     return super().evaluate(*args, **kwargs)


class MistralForCausalLMWithPromptLoss(MistralForCausalLM):
    def __init__(self, config, prompt_loss=0):
        super().__init__(config)
        self.prompt_loss_weight = prompt_loss

    def get_loss(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # loss = loss_fct(shift_logits, shift_labels)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if self.prompt_loss_weight > 0:
                raw_labels = labels
                shift_raw_labels = raw_labels[..., 1:].contiguous()
                shift_raw_labels = shift_raw_labels.view(-1)
                # Enable model parallelism
                shift_raw_labels = shift_raw_labels.to(logits.device)
                labels = input_ids
                loss_fct = CrossEntropyLoss(reduction="none")
            else:
                loss_fct = CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if self.prompt_loss_weight > 0:
                # print(f"shift_raw_labels: {shift_raw_labels.shape}, shift_labels: {shift_labels.shape}, loss: {loss.shape}")
                prompt_loss = loss[(shift_raw_labels == -100) & (shift_labels != -100)].mean()
                target_loss = loss[(shift_raw_labels != -100) & (shift_labels != -100)].mean()
                # loss = self.prompt_loss_weight * prompt_loss + (1 - self.prompt_loss_weight) * target_loss
                loss = self.prompt_loss_weight * prompt_loss + target_loss
                # print(f"prompt_loss: {prompt_loss}, target_loss: {target_loss}, loss: {loss}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


my_max_input_length = 0
total_epoch = 0

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = local_rank
    world_size = int(os.environ["WORLD_SIZE"])

    if global_rank == 0:
        if data_args.debug_flag:
            wandb.init(mode="disabled")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    random.seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_args.max_source_length+data_args.max_target_length,
        # add_eos_token=True,
    )
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    logger.info(config)
    logger.info(len(tokenizer))
    logger.info(tokenizer.pad_token)
    logger.info(tokenizer.pad_token_id)
    logger.info(tokenizer.bos_token)
    logger.info(tokenizer.eos_token)
    logger.info(tokenizer.padding_side)
    logger.info(tokenizer.truncation_side)

    if "eem" in data_args.task_name:
        assert "mistral" in model_args.model_name_or_path.lower(), "Error: eem task must use mistral model."
        model = MistralForCausalLMWithPromptLoss.from_pretrained(
            model_args.model_name_or_path,
            prompt_loss=model_args.prompt_loss,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

    logger.info(model)



    # def process(data):
    #     global my_max_input_length
    #     prompt = demo + data['question']
    #     inputs = tokenizer(prompt)
    #     ans = data['answer']
    #     ans = ans.replace('####', 'Therefore, the answer is')
    #     labels = tokenizer(ans)
    #     labels['input_ids'] = labels['input_ids'][1:]
    #
    #     data['input_ids'] = inputs['input_ids'] + labels['input_ids']
    #     data['labels'] = [-100] * len(inputs['input_ids']) + labels['input_ids']
    #     data['attention_mask'] = [1] * len(data['input_ids'])
    #     assert (len(data['input_ids']) == len(data['labels'])), "Error: input_ids and labels have different length!"
    #     my_max_input_length = max(my_max_input_length, len(data['input_ids']))
    #
    #     return data
    # my_dataset = load_dataset("gsm8k", name='main')
    # train_dataset = my_dataset['train']
    # test_dataset = my_dataset['test']
    #
    # ids = random.sample([i for i in range(len(train_dataset))], 3)
    # demo = ''
    # # pattern = PROMPT + STOP_WORD
    # for idx in ids:
    #     data = train_dataset[idx]
    #     problem = data['question']
    #     solution = data['answer']
    #     answer = solution.split('####')[-1].strip()
    #     solution = solution.split('####')[0].strip()
    #     demo = demo + f'{problem}\n{solution} Therefore, the answer is {answer}'

    if global_rank == 0 and not os.path.exists(data_args.result_path):
            os.makedirs(data_args.result_path, exist_ok=True)

    train_dataset, test_dataset, processor = get_dataset(data_args.task_name, tokenizer, data_args.result_path, data_args.max_source_length+data_args.max_target_length)
    if training_args.do_train:
        # add_train_dataset2, add_test_dataset2, add_processor2 = get_dataset('svamp', tokenizer, data_args.result_path, data_args.max_source_length+data_args.max_target_length)
        #
        # add_processor.max_input_length = 0
        # add_train_dataset = add_train_dataset.map(add_processor.process_train, load_from_cache_file=False)
        # add_train_dataset = add_processor.post_process(add_train_dataset)

        # info: only use in mmqa
        if data_args.task_name == "mmqa":
            add_train_dataset, add_test_dataset, add_processor = get_dataset('gsm8k', tokenizer, data_args.result_path, data_args.max_source_length+data_args.max_target_length)
            add_test_dataset = add_test_dataset.map(add_processor.process_test, load_from_cache_file=False)
            test_dataset = add_test_dataset
        elif test_dataset:
            test_dataset = test_dataset.map(processor.process_test, load_from_cache_file=False)

        # add_processor2.max_input_length = 0
        # add_train_dataset2 = add_train_dataset2.map(add_processor2.process_train, load_from_cache_file=False)
        # add_train_dataset2 = add_processor2.post_process(add_train_dataset2)

        # add_train_dataset = BaseProcessor.sort_by_length(add_train_dataset)
        # logger.info(f"Max Add Train Input Length: {add_processor.max_input_length}")
        # add_processor.max_input_length = 0
        # add_test_dataset = add_test_dataset.map(add_processor.process_test, load_from_cache_file=False)
        # logger.info(f"Max Add Test Input Length: {add_processor.max_input_length}")

        # logger.info('--------------- Raw Dataset ---------------')
        # logger.info(train_dataset)
        # logger.info(test_dataset)

        if data_args.debug_flag:
            # add_train_dataset = add_train_dataset.select(range(100))
            # add_train_dataset2 = add_train_dataset2.select(range(100))
            train_dataset = train_dataset.select(range(100))
            if test_dataset:
                test_dataset = test_dataset.select(range(100))
        processor.max_input_length = 0
        train_dataset = train_dataset.map(processor.process_train, load_from_cache_file=False)
        if data_args.sample_flag and len(train_dataset) > 40_000:
            final_train_num = 40_000
            print(f"Warning: Auto set final train num to 40_000, because of the length {len(train_dataset)} of train dataset.")
        else:
            final_train_num = -1
        train_dataset = processor.post_process(train_dataset, final_train_num)
        logger.info(f"Max Train Input Length: {processor.max_input_length}")

        logger.warning(f"before add: {len(train_dataset)}")
        # final_train_dataset = concatenate_datasets([train_dataset, add_train_dataset])
        # logger.warning(f"after add: {len(final_train_dataset)}")
        # exit(-1)

        processor.max_input_length = 0

        logger.info(f"Max Test Input Length: {processor.max_input_length}")
        logger.info('--------------- Processed Dataset ---------------')
        logger.info(train_dataset)
        logger.info(test_dataset)
        # test_dataset = processor.sort_by_length(test_dataset)
        processor.test_dataset = test_dataset

        demo = "Sample:\n"
        for i in range(3):
            demo += f"{train_dataset[i]['input_text']}\n{train_dataset[i]['label_text']}\n\n"
        logger.info(f'\n{demo}')
        demo = ''

        # data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

        logger.info(f"training dataset: {train_dataset}")
        logger.info(f"test dataset: {test_dataset}")
        # logger.info(f"add training dataset: {add_train_dataset}")
        # logger.info(f"add training dataset2: {add_train_dataset2}")
        # logger.info(f"final training dataset: {final_train_dataset}")

    # best_em = 0.0
    # def compute_metrics(eval_preds):
    #     nonlocal best_em
    #     preds, labels = eval_preds
    #     num_correct, total_problem = 0, len(preds)
    #     assert len(preds) == len(labels)
    #
    #     for p, l in zip(preds, labels):
    #         p = np.where(p != -100, p, tokenizer.pad_token_id)
    #         p = tokenizer.decode(p, skip_special_tokens=True)
    #         l = np.where(l != -100, l, tokenizer.pad_token_id)
    #         l = tokenizer.decode(l, skip_special_tokens=True)
    #         print(f"p: {p}\n l: {l}")
    #         if 'Therefore, the answer is' in p:
    #             p = p.split('Therefore, the answer is')[1].strip()
    #         else:
    #             p = ""
    #         assert 'Therefore, the answer is' in l
    #         l = l.split('Therefore, the answer is')[1].strip()
    #         if p == l:
    #             num_correct += 1
    #
    #     result = round(num_correct / total_problem * 100, 2)
    #     best_em = max(best_em, result)
    #     logger.info(f'Best Exactly Match: {best_em}')
    #     return {'EM': result}

    bsmodel = None
    if training_args.num_train_epochs == 0:
        if len(train_dataset) <= 2_000:
            training_args.num_train_epochs = 40
        elif len(train_dataset) <= 10_000:
            training_args.num_train_epochs = 20
        else:
            training_args.num_train_epochs = 8
        logger.info(f"Auto set training epochs to {training_args.num_train_epochs}, because of the length of {len(train_dataset)}.")

    if model_args.method == "ft":
        print("Info: This is full finetune method.")
    elif model_args.method == "lora":
        pconfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_module.split('.'),
            lora_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        # for name, p in model.named_parameters():
        #     if "lora" in name:
        #         print(name, p)
        # exit(-1)
        model.print_trainable_parameters()
    elif model_args.method == "adalora":
        pconfig = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    # elif model_args.method == "ptuning":
    #     pconfig = PromptEncoderConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         encoder_reparameterization_type="MLP",
    #         encoder_hidden_size = 768,
    #     )
    #     model = get_peft_model(model, pconfig)
    #     model.print_trainable_parameters()
    elif model_args.method == "loha":
        pconfig = LoHaConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            rank_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    elif model_args.method == "lokr":
        pconfig = LoKrConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            rank_dropout=0.1,
        )
        model = get_peft_model(model, pconfig)
        model.print_trainable_parameters()
    elif model_args.method == "bslora":
        if model_args.merge_flag:
            assert model_args.res_flag, "Error: merge flag must be used with res flag 1 or 3."
        if model_args.res_flag == 2 or model_args.res_flag == 3:
            assert model_args.pre_num != 0, "Error: pre num must be used when res flag 2 or 3."
        bsconfig = BsLoraConfig(rank=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_num=model_args.lora_num,
                                target_modules=model_args.target_module, lora_dropout=0.1, res_flag=model_args.res_flag,
                                mul_flag=model_args.mul_flag, merge_flag=model_args.merge_flag, pre_num=model_args.pre_num, flag_reset=model_args.flag_reset)
        lora_num = bsconfig.lora_num
        # new_epochs = (training_args.num_train_epochs // lora_num) * bsconfig.lora_num
        # print(f"Info: BsLora has change training epochs from {training_args.num_train_epochs} to {new_epochs}, to adapt for lora num.")
        # training_args.num_train_epochs = int(new_epochs)
        bsmodel = BsLoraModel(model, bsconfig, epochs=training_args.num_train_epochs)
        # for name, p in model.named_parameters():
        #     if "lora" in name:
        #         print(name, p)
        # exit(-1)
        # data_scheduler = DataScheduler(world_size, global_rank, tokenizer.pad_token_id, 100, 40)
        logger.info(
            f"bslora config: {bsconfig.to_json_string()}\n"
        )
        logger.info(
            f"Trainable params: {sum(p.numel() for p in bsmodel.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    else:
        print("Error: Wrong method!")
        raise NotImplementedError

    if training_args.resume_from_checkpoint:
        if bsmodel:
            bsmodel.unconcat_loras()
        model_weights = {}
        from safetensors.torch import load_file

        for root, folders, files in os.walk(training_args.resume_from_checkpoint):
            for file in files:
                if file.endswith(".safetensors"):
                    model_weights.update(load_file(os.path.join(root, file)))
        # print(model_weights.keys())
        load_result = model.load_state_dict(model_weights, strict=True)
        print(load_result)

        if bsmodel and bsmodel.bsconfig.res_flag > 0:
            with open(f"{training_args.resume_from_checkpoint}/bslora_config.json", 'r') as f:
                bsconfig = BsLoraConfig(**json.load(f))
                assert bsconfig.rank == bsmodel.bsconfig.rank and bsconfig.res_flag == bsmodel.bsconfig.res_flag and bsconfig.pre_num == bsmodel.bsconfig.pre_num and ((bsconfig.merge_flag == bsmodel.bsconfig.merge_flag) or (bsmodel.bsconfig.merge_flag == 0))
                if bsmodel and bsmodel.bsconfig.res_flag == 1 or bsmodel.bsconfig.res_flag == 3:
                    bsmodel.bsconfig.lora_merge_alpha = bsconfig.lora_merge_alpha
                    bsmodel.load_merge_alpha()
            bsmodel.concat_loras()
        print(f"Loaded model successfully from checkpoint {training_args.resume_from_checkpoint}")


    generation_config = GenerationConfig(temperate=0.95, max_length=data_args.max_target_length,
                                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                         min_new_tokens=10, remove_invalid_values=True)
    training_args.generation_config = generation_config
    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=processor.compute_metrics,
    #     callbacks=[],
    # )
    if global_rank == 0 and not os.path.exists(data_args.result_path):
        os.makedirs(data_args.result_path)

    if global_rank == 0:
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0


        for param in model.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量
        logger.info(f"trainable params: {Trainable_params / 1_000_000:.2f}M, non-trainable params: {NonTrainable_params / 1_000_000:.2f}M, total params: {Total_params / 1_000_000:.2f}M")

    if model_args.method == "bslora":
        # per_epoch = int(training_args.num_train_epochs // lora_num)
        raw_output_dir = training_args.output_dir
        logger.info(training_args)

        # train_num = len(train_dataset)
        # train_set1 = train_dataset.select(range(0, train_num // 3))
        # train_set2 = train_dataset.select(range(train_num // 3, train_num // 3 * 2))
        # train_set3 = train_dataset.select(range(train_num // 3 * 2, train_num))
        # train_set1 = train_dataset
        # train_set2 = train_dataset.select(range(0, train_num // 2))
        # train_set3 = train_dataset.select(range(train_num // 2, train_num))
        # train_sets = [train_set2, train_set3, train_set1, ]
        # assert lora_num == 3

        # train_list = [add_train_dataset, train_dataset, final_train_dataset]
        # train_list = [add_train_dataset, add_train_dataset2, train_dataset]
        # epoch_list = [10, 10, 20]
        # train_list = [final_train_dataset, add_train_dataset, train_dataset]

        lora_num = 1
        for i in range(lora_num):
            if global_rank == 0:
                if data_args.debug_flag:
                    wandb.init(mode="disabled")
                else:
                    names = training_args.output_dir.split('/')
                    tags = [names[-2], names[-3], model_args.method, f"lora_{i}"]
                    wandb.init(project=f"bslora_gsm8k", tags=tags, name="_".join(tags))
            # training_args.num_train_epochs = per_epoch
            training_args.output_dir = f"{raw_output_dir}_{i}"
            # training_args.num_train_epochs = epoch_list[i]
            # training_args.num_train_epochs = 40
            logger.warning(f"Training epoch: {training_args.num_train_epochs}")
            print(f"now model is in lora {bsmodel.new_epoch()}")
            # print(f"now model is in lora {bsmodel.new_epoch(data_scheduler)}")
            logger.info(training_args)
            logger.info(test_dataset)
            # if model_args.flag_reset_lr > 0:
            #     base_lr = training_args.
            #     optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
            #     linear_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_args.num_train_epochs)
            # else:
            #     pass
            trainer = BsTrainer(
                model=bsmodel.wrapped_model,
                global_rank=global_rank,
                # prompt_loss=model_args.prompt_loss,
                args=training_args,
                train_dataset=train_dataset,
                # train_dataset=train_sets[i] if training_args.do_train else None,
                # train_dataset=train_list[i] if training_args.do_train else None,
                # train_dataset=final_train_dataset if training_args.do_train else None,
                eval_dataset=test_dataset if training_args.do_eval else None,
                compute_metrics=processor.compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                bsmodel=bsmodel,
            )

            # trainer.add_callback(callback_step)

            # interval = training_args.num_train_epochs // bsconfig.lora_num

            # Training
            if training_args.do_train:
                checkpoint = None
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
                metrics = train_result.metrics
                metrics["train_samples"] = len(train_dataset)

                trainer.save_model()  # Saves the tokenizer too for easy upload

                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()

            if training_args.do_predict:
                test_dataset_list = []
                if 'eem2' in data_args.task_name:
                    test_dataset_list = [
                        '/data/local/User/minghuisong/NLG/QKG/EEM/test_apac_2k.txt',
                        '/data/local/User/minghuisong/NLG/QKG/EEM/test_eu3_2k.txt',
                        '/data/local/User/minghuisong/NLG/QKG/EEM/test_latam_2k.txt',
                        '/data/local/User/minghuisong/NLG/QKG/EEM/test_na10k.txt',
                        '/data/local/User/minghuisong/NLG/QKG/EEM/test_roe_2k.txt',
                    ]

                for test_name in test_dataset_list:
                    test_dataset, _, test_processor = get_dataset(data_args.task_name, tokenizer, data_args.result_path, data_args.max_source_length, test_name)
                    test_processor.max_input_length = 0
                    test_dataset = test_dataset.map(test_processor.process_test, load_from_cache_file=False)
                    test_dataset = test_dataset.remove_columns(["labels", "label_text"])
                    if data_args.debug_flag:
                        test_dataset = test_dataset.select(range(1000))

                    logger.info(test_dataset)

                    predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")

                    assert len(predict_results.predictions) == len(test_dataset)
                    result = []
                    predict_results.predictions[predict_results.predictions == -100] = tokenizer.pad_token_id
                    pred = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True)
                    for p in pred:
                        result.append({'predict': p})
                    # for t, p in zip(test_dataset, pred):
                    #     print(t)
                    #     print(p)
                    #     result.append({'input_text': t['input_text'], 'predict_text': p})
                    test_name = test_name.split('/')[-1].split('.')[0]
                    with open(f"{data_args.result_path}/infer_{test_name}_result.json", 'w') as f:
                        json.dump(result, f)
                    logger.info(f"Predict result has been saved to {test_name.split('/')[-1].split('.')[0]}_result.json")

            # # Evaluation
            # if training_args.do_eval:
            #     logger.info("*** Evaluate ***")
            #
            #     # Loop to handle MNLI double evaluation (matched, mis-matched)
            #     tasks = [data_args.task_name]
            #     eval_datasets = [eval_dataset]
            #     if data_args.task_name == "mnli":
            #         tasks.append("mnli-mm")
            #         eval_datasets.append(datasets["validation_mismatched"])
            #
            #     for eval_dataset, task in zip(eval_datasets, tasks):
            #         # trainer.model.set_rank(rank=rank) # set the test rank
            #         metrics = trainer.evaluate(eval_dataset=eval_dataset)
            #         max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            #             eval_dataset)
            #         metrics[f"eval_samples"] = min(max_val_samples, len(eval_dataset))
            #
            #         trainer.log_metrics(f"eval", metrics)
            #         trainer.save_metrics(f"eval", metrics)
            wandb.finish()
    else:
        class TrainerAdapterCallback(TrainerCallback):

            def __init__(self):
                self.global_step = 0

            # offload original_modules to cpu, to save memory
            def on_train_begin(self, _args, state, control, **kwargs):
                # if hasattr(model, 'set_active_adapters'):
                #     model.set_active_adapters(model.adapters.keys(), offload='cpu')
                if model_args.method == 'adalora':
                    # model.peft_config['default'].total_step = state.max_steps

                    def zero_grad(_self, *args, **kwargs):
                        _self.update_and_allocate(self.global_step + 1)
                        _self._zero_grad(*args, **kwargs)

                    model._zero_grad = model.zero_grad
                    model.zero_grad = zero_grad
                    # model.zero_grad = types.MethodType(zero_grad, model)

            def on_step_end(self, _args, state, control, **kwargs):
                if model_args.method == 'adalora':
                    self.global_step = state.global_step

        if global_rank == 0:
            if data_args.debug_flag:
                wandb.init(mode="disabled")
            else:
                names = training_args.output_dir.split('/')
                tags = [names[-2], names[-3], model_args.method, f"lora_{0}"]
                wandb.init(project=f"bslora_gsm8k", tags=tags, name="_".join(tags))

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=processor.compute_metrics,
            callbacks=[],
        )

        if model_args.method == "adalora":
            trainer.add_callback(TrainerAdapterCallback())


        # Training
        if training_args.do_train:
            checkpoint = None
            # if last_checkpoint is not None:
            #     checkpoint = last_checkpoint
            # elif os.path.isdir(model_args.model_name_or_path):
            #     # Check the config from that potential checkpoint has the right number of labels before using it as a
            #     # checkpoint.
            #     if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
            #         checkpoint = model_args.model_name_or_path

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    # if training_args.do_eval:
    #     trainer.evaluate()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         eval_datasets.append(datasets["validation_mismatched"])
    #
    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         for rank in range(0, model_args.lora_r):
    #             print(f'--> eval rank={rank}')
    #             # trainer.model.set_rank(rank=rank) # set the test rank
    #             metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #
    #             max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
    #                 eval_dataset)
    #             metrics[f"eval_samples_r{rank}"] = min(max_val_samples, len(eval_dataset))
    #
    #             trainer.log_metrics(f"eval_r{rank}", metrics)
    #             trainer.save_metrics(f"eval_r{rank}", metrics)

    # # Predict
    # logger.info(predict_results)


if __name__ == "__main__":
    main()