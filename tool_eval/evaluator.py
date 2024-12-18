import argparse
import ast
import json
import re
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset
from hermes_utils import *
from prompter import PromptManager
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from utils import (calculate_pass_rate, eval_logger, get_assistant_message,
                   get_chat_template, validate_and_extract_tool_calls_regex,
                   validate_tool_calls)
from validator import validate_function_call_schema


class ModelEvaluator:
    def __init__(self, model_path, chat_template, load_in_4bit, flash_attn, dpo):
        self.prompter = PromptManager()
        self.bnb_config = None

        if load_in_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Prepare model loading arguments
        model_args = {
            "trust_remote_code": True,
            "return_dict": True,
            "quantization_config": self.bnb_config,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        # Conditionally add attn_implementation if flash_attn is True
        if flash_attn:
            model_args["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_args
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = get_chat_template(chat_template)

        self.eval_results = []
        if dpo:
            self.dpo_results = []
        
        eval_logger.info(self.model.config)
        eval_logger.info(self.model.generation_config)
        eval_logger.info(self.model.parameters)
        eval_logger.info(self.tokenizer.chat_template)
        eval_logger.info(self.tokenizer.special_tokens_map)


    def evaluate_hermes(self, eval_dataset, chat_template, scratch_pad, num_fewshot):
        for sample in tqdm(eval_dataset, desc="processing samples", unit="sample"):  
            prompts = generate_prompt(sample, enable_system=False)
            prompt, eval_completion = prompts[:-1], prompts[-1]
            
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors='pt'
            )

            tokens = self.model.generate(
                inputs.to(self.model.device),
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True
            )[:, inputs.shape[-1]:]

            completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False)
            eval_logger.info(f"model completion with eval prompt:\n{completion}")

            assistant_message = completion.replace(self.tokenizer.eos_token, "")
            validation, tool_calls = validate_and_extract_tool_calls_regex(assistant_message)

            sample['model_completion'] = []
            sample['result'] = "failed"

            eval_completion = parse_completion(eval_completion["content"])
            if eval_completion is None:
                eval_logger.info("No function calls found in completion")
                continue
            
            eval_completion = validate_hermes_tool_calls(eval_completion)
            if eval_completion is None:
                eval_logger.info("No valid function calls found in completion")
                continue

            if validation:
                if isinstance(eval_completion, list):        
                    eval_tool_calls = eval_completion
                else:
                    eval_tool_calls = [eval_completion]
                
                all_valid = True
                if len(tool_calls) != len(eval_tool_calls):
                    all_valid = False
                    eval_logger.info("Number of tool calls doesn't match")
                    eval_logger.info(f"Expected: {len(eval_tool_calls)} tool calls; Got: {len(tool_calls)}")

                for eval_tool_call in eval_tool_calls:
                    function_found = False

                    for tool_call in tool_calls:
                        schema_validation = validate_function_call_schema(tool_call, json.loads(sample['tools']))
                        if not schema_validation:
                            all_valid = False
                            break

                        if tool_call['name'] == eval_tool_call['name']:
                            function_found = True
                            result = validate_tool_calls(tool_call['arguments'], eval_tool_call['arguments'])
                            sample['model_completion'].append(tool_call)
                            eval_logger.info(f"{tool_call['name']} validation: {result}")
                            if result == "failed":
                                all_valid = False
                            break
                    if not function_found:
                        eval_logger.info(f"Function '{eval_tool_call['name']}' not found") 
                        all_valid = False         
            else:
                eval_logger.info("Function call validation failed")
                all_valid = False
            
            if all_valid:
                sample['result'] = "passed"
                eval_logger.info(f"all validations: {sample['result']}")
                eval_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
            else:
                sample['model_completion'] = assistant_message
                eval_logger.info(f"all validations: {sample['result']}")
                eval_logger.info(f"failed tool calls:\n{assistant_message}")

            self.eval_results.append(sample)

    def evaluate_model(self, eval_dataset, chat_template, scratch_pad, num_fewshot):

        for sample in tqdm(eval_dataset, desc="processing samples", unit="sample"):  
            prompt, sys_prompt = self.prompter.generate_prompt(sample, scratch_pad, num_fewshot)
            
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors='pt'
            )

            tokens = self.model.generate(
                inputs.to(self.model.device),
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True
            )

            completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False)
            eval_logger.info(f"model completion with eval prompt:\n{completion}")

            assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)
            validation, tool_calls = validate_and_extract_tool_calls_regex(assistant_message)

            sample['model_completion'] = []
            sample['result'] = "failed"

            eval_completion = json.loads(sample['completion'])
            if validation:
                if isinstance(eval_completion, list):        
                    eval_tool_calls = eval_completion
                else:
                    eval_tool_calls = [eval_completion]
                
                all_valid = True
                if len(tool_calls) != len(eval_tool_calls):
                    all_valid = False
                    eval_logger.info("Number of tool calls doesn't match")
                    eval_logger.info(f"Expected: {len(eval_tool_calls)} tool calls; Got: {len(tool_calls)}")

                for eval_tool_call in eval_tool_calls:
                    function_found = False

                    for tool_call in tool_calls:
                        schema_validation = validate_function_call_schema(tool_call, json.loads(sample['tools']))
                        if not schema_validation:
                            all_valid = False
                            break

                        if tool_call['name'] == eval_tool_call['name']:
                            function_found = True
                            result = validate_tool_calls(tool_call['arguments'], eval_tool_call['arguments'])
                            sample['model_completion'].append(tool_call)
                            eval_logger.info(f"{tool_call['name']} validation: {result}")
                            if result == "failed":
                                all_valid = False
                            break
                    if not function_found:
                        eval_logger.info(f"Function '{eval_tool_call['name']}' not found") 
                        all_valid = False         
            else:
                eval_logger.info("Function call validation failed")
                all_valid = False
            
            if all_valid:
                sample['result'] = "passed"
                eval_logger.info(f"all validations: {sample['result']}")
                eval_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
            else:
                sample['model_completion'] = assistant_message
                eval_logger.info(f"all validations: {sample['result']}")
                eval_logger.info(f"failed tool calls:\n{assistant_message}")

                if hasattr(self, 'dpo_results'):
                    chosen_completion = ""
                    for tool_call in eval_completion:
                        chosen_completion += f"<tool_call>\n{tool_call}\n<tool_call>\n"
                    self.dpo_results.append({
                        "system": sys_prompt,
                        "question": sample['prompt'][0]['content'],
                        "chosen": chosen_completion,
                        "rejected": assistant_message
                    })

            self.eval_results.append(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance on fireworks-ai dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to subset eval dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="Huggingface dataset path")
    parser.add_argument("--load_in_4bit", type=bool, default=False, help="Option to load in 4bit with bitsandbytes")
    parser.add_argument("--flash_attn", type=bool, default=False, help="weather to use flash attention; requires installing flash-attn")
    parser.add_argument("--scratch_pad", type=bool, default=False, help="whether to use scratchpad reasoning")
    parser.add_argument("--dpo", type=bool, default=False, help="Option to save dpo dataset")
    args = parser.parse_args()

    # load eval dataset
    if args.dataset_path:
        eval_dataset = load_dataset(args.dataset_path)["train"]
    else:
        eval_dataset = load_dataset("riczhou/hermes-function-calling-v1-glaive-split", split="test")

    # Create model evaluator instance
    model_evaluator = ModelEvaluator(args.model_path, args.chat_template, args.load_in_4bit, args.flash_attn, args.dpo)

    # Run the model evaluator
    model_evaluator.evaluate_hermes(eval_dataset, args.chat_template, args.scratch_pad, args.num_fewshot)

    # Calculate and print pass rate
    pass_rate = calculate_pass_rate(model_evaluator.eval_results)
    eval_logger.info(f"function-calling eval (pass@1): {pass_rate}")

    results_path = './function_calling_eval_results.json'
    with open(results_path, 'w') as file:
        json.dump(model_evaluator.eval_results, file)

    if args.dpo == "True":
        dpo_path = './function_calling_dpo_pairs.json'
        with open(dpo_path, 'w') as file:
            json.dump(model_evaluator.dpo_results, file)


