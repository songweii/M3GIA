import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import requests
import time
import base64
from random import choice
import random
import numpy as np
from PIL import Image
import math
from PIL import Image
import requests
from io import BytesIO
import math
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pyarrow.parquet as pq
from utils.data_utils import ABILITY_DICT, CLUSTER_CAT2QUESTION_CAT, QUESTION_CAT_DICT, QUESTION_TYPE_NAME
from utils.eval_utils import split_list, is_none, get_options, parse_multi_choice_response, eval_multi_choice
from utils.prompt_config import PROMPT_DICT


all_options = ['A', 'B', 'C', 'D', 'E', 'F']
image_tags = ['<image>\n', '\n<image>']
text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
# text_only_template = "你是一个人工智能助手，我有一些关于能力测试的选择题，针对我给的问题和选项希望能够严谨和如实的回答，问题："


def eval_model(args):
    # Evaluate closed sourced model:
    if args.api_key is not None:
        api_dict = {
            "api_base": args.api_base,
            "api_key": args.api_key,
        }
    # Evaluate open sourced model. Load model and processor.
    if args.model_path is not None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
        ).to(DEVICE).eval()

    # load from huggingface
    # dataset = load_dataset('Songweii/M3GIA', name=args.language, split='test')
    # load from parquet file
    dataset = pq.read_table(args.question_file).to_pandas()
    print(dataset.head())

    answers_path = args.answers_file + '/' + args.model_name + '/' + args.language + '.jsonl'
    answers_file = os.path.expanduser(answers_path)
    # answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    cnt_all_valid = 0
    cnt_type_numbers = {}
    for question_type in list(QUESTION_CAT_DICT.keys()):
        cnt_type_numbers[question_type] = 0

    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        options, index2ans = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]
        if len(options) < 1 or (QUESTION_TYPE_NAME in row and row[QUESTION_TYPE_NAME] not in QUESTION_CAT_DICT):
            continue

        idx = str(row['index']).strip()
        question = str(row['question']).strip()
        hint = str(row['hint']).strip()

        if is_none(hint):
            question = question
        if is_none(question):
            question = hint
        if not is_none(hint) and not is_none(question):
            question = hint + '\n' + str(question)

        choices = ['(' + option_char + ') ' + option for option_char, option in zip(all_options[:len(options)], options)]
        prompt = "Question: " + question + "\n" + "Choices: " + "  ".join(choices) + "\n" + "Answer: "
        qs = cur_prompt = prompt

        assert args.prompt_type in ["prefix", "suffix"], "Prompt Type must be Prefix or Suffix."
        if args.prompt_type == "prefix":
            qs = PROMPT_DICT[args.prompt_type] + "\n" + qs
        else:
            qs = qs + '\n' + PROMPT_DICT[args.prompt_type]

        image_data = row['image']
        history = []
        if isinstance(image_data, dict):
            print("processing image data")
            img_data = BytesIO(image_data['bytes'])
            image = Image.open(img_data).convert("RGB")
            query = qs

            # Customized for evaluating CogVLM2, which we use as an example in this template.
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )  # You can change these lines flexibly according to the model you evaluate.
        else:
            print("no input image")
            image = None
            query = text_only_template.format(qs)
            text_only_first_query = False

            # Customized for evaluating CogVLM2, which we use as an example in this template.
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                template_version='chat'
            )  # You can change these lines flexibly according to the model you evaluate.

        # ------------------Customized for evaluating CogVLM2, which we use as an example in this template.------------------
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": 128002,
            "temperature":args.temperature,
            "do_sample":False
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
        # ------------------You can change these lines flexibly according to the model you evaluate.------------------

        # print(generation_text)
        print("---------------------------------- prompt --------------------------------------")
        print(qs)
        print("--------------------------------------------------------------------------------\n")

        print(f"{args.model_name}:", response)

        # result_char = parse_multi_choice_response(question, choices, response, cur_option_char, index2ans, api_dict=api_dict)
        # print("choice: " + result_char)

        qs_type = str(row[QUESTION_TYPE_NAME]).strip()
        cnt_type_numbers[qs_type] += 1
        cnt_all_valid += 1

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx, 
                                   "question_type": qs_type,
                                   "question": question,
                                   "choices": choices,
                                   "prompt": qs,
                                   "index2ans": index2ans,
                                   "response": response,
                                   "prediction": None,
                                   "options": options,
                                   "option_char": cur_option_char,
                                   "answer_id": ans_id,
                                   "model_id": args.model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()

    print("===="*10 + f" {args.model_name} {args.language}" + "===="*10)
    print(f"total questions: {cnt_all_valid}\n")

    from prettytable import PrettyTable
    # Create PrettyTable object
    table = PrettyTable()
    table.add_column("Question Type", list(QUESTION_CAT_DICT.keys()))
    table.add_column("Question Number", list(QUESTION_CAT_DICT.values()))
    table.add_column("Complete Number", list(cnt_type_numbers.values()))
    print(table)

    ans_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="en.parquet")
    parser.add_argument("--answers-file", type=str, default="./data/answers", help="Path to save the output")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--language", type=str, default="english", choices=['english', 'chinese', 'spanish', 'french', 'portuguese', 'korean'])
    parser.add_argument("--prompt-type", type=str, default="prefix")

    parser.add_argument('--model-name', type=str, default=None, help="Name the model evaluated, for example: cogvlm2-19B.")

    # Enter the path to the model's weight if you want to evaluate an open source model. 
    parser.add_argument('--model-path', type=str, default=None, help="The path to the model's weight.")

    # Enter your api-key if you want to evaluate a closed source model.
    parser.add_argument('--api-key', type=str, default=None, help="API Key.")
    parser.add_argument('--api-base', type=str, default=None, help="API Base.")

    args = parser.parse_args()
    eval_model(args)

    # "/mnt/data/pretrain/pretrain_weight/cogvlm/cogvlm2-llama3-chinese-chat-19B"
