"""Parse and Evalate"""
import sys
sys.path.insert(0, './')
import os
import json
import jsonlines
import pandas as pd
from pandas import json_normalize

import pdb
from argparse import ArgumentParser

from utils.data_utils import save_json, QUESTION_CAT_DICT, CLUSTER_CAT2QUESTION_CAT, ABILITY_DICT
from utils.eval_utils import evaluate, parse_multi_choice_response, parse_open_response, calculate_ins_level_acc, eval_multi_choice


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./data/answers/", help="The path to model output file.")
    parser.add_argument('--answer_path', type=str, default="./data/ground_truth/", help="Answer file path.")
    parser.add_argument('--result_path', type=str, default="./data/eval_result/", help="Eval file path.")
    parser.add_argument('--data_version', type=str, default="v1", help="Data version.")
    parser.add_argument('--model_name_list', nargs='+', type=str, help="Model list.")
    parser.add_argument('--language_list', nargs='+', type=str, help="Language list.")
    # Openai-key for option extraction (gpt4-turbo)
    parser.add_argument('--openai-api-key', type=str, default=None, help="OpenAI API Key.")
    parser.add_argument('--openai-api-base', type=str, default=None, help="OpenAI API Base.")
    args = parser.parse_args()

    openai_api_dict = None
    if args.openai_api_base is not None and args.openai_api_key is not None:
        openai_api_dict = {
            "api_base": args.openai_api_base,
            "api_key": args.openai_api_key,
        }

    version = args.data_version
    model_name_list = args.model_name_list
    language_list = args.language_list

    for model_name in model_name_list:
        category_data = []
        ability_data = []
        for language in language_list:
            print("Evaluating: {}".format(language))
            # output_path = f'{args.output_path}/{model_name}/{language}_{version}.jsonl'
            output_path = f'{args.output_path}/{model_name}/{language}.jsonl'
            output_dict = []
            with open(output_path, 'r') as f:
                for item in jsonlines.Reader(f):
                    output_dict.append(item)

            # answer_path = f'{args.answer_path}/answer_dict_{language}_{version}.jsonl'
            answer_path = f'{args.answer_path}/answer_dict_{language}.jsonl'
            answer_dict = json.load(open(answer_path))


            if "question_type" in output_dict[0]:
                # group by category
                output_dict_w_cat = {}
                for item in output_dict:
                    parsed_pred = parse_multi_choice_response(item["question"], item["choices"], 
                                                              item["response"], item["option_char"], 
                                                              item["index2ans"], api_dict=openai_api_dict)
                    item["prediction"] = parsed_pred
                    # print("choice: " + parsed_pred)

                    data_id, category = item["question_id"], item["question_type"]
                    if category not in output_dict_w_cat:
                        output_dict_w_cat.update({category: {}})
                    output_dict_w_cat[category].update({data_id: parsed_pred})

                answer_dict_w_cat = {}
                for item in answer_dict:
                    data_id, category, ground_truth = item["question_id"], item["question_type"], item["ground_truth"],
                    if category not in answer_dict_w_cat:
                        answer_dict_w_cat.update({category: {}})
                    answer_dict_w_cat[category].update({data_id: ground_truth})

                evaluation_result = {}
                for category in QUESTION_CAT_DICT.keys():
                    # get cat_outputs and cat_answers
                    try:
                        cat_outputs = output_dict_w_cat[category]
                        cat_answers = answer_dict_w_cat[category]
                    except KeyError:
                        print("Skipping {} for not found".format(category))
                        continue

                    exampels_to_eval = []
                    for data_id, parsed_pred in cat_outputs.items():
                        if data_id in cat_answers:
                            answer = cat_answers[data_id]
                        else:
                            answer = ""
                        exampels_to_eval.append({
                            "id": data_id,
                            "question_type": 'multiple-choice',
                            "answer": answer,
                            "parsed_pred": parsed_pred
                        })

                    judge_dict, metric_dict = evaluate(exampels_to_eval)
                    metric_dict.update({"num_example": len(exampels_to_eval)})

                    evaluation_result[category] = metric_dict

                printable_results = {}
                # pdb.set_trace()
                # add domain Subject
                for domain, in_domain_cats in CLUSTER_CAT2QUESTION_CAT.items():
                    in_domain_cat_results = {}
                    for cat_name in in_domain_cats:  # use the order in CLUSTER_CAT2QUESTION_CAT
                        if cat_name in evaluation_result.keys():
                            in_domain_cat_results[cat_name] = evaluation_result[cat_name]
                        else:
                            pass
                    in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
                    in_domain_data_num = sum([cat_results['num_example'] for cat_results in in_domain_cat_results.values()])
                    printable_results['Overall-' + domain] = {"num": int(in_domain_data_num),
                                                              "acc": round(in_domain_ins_acc, 3)
                                                            }
                    # add sub category
                    for cat_name, cat_results in in_domain_cat_results.items():
                        printable_results[cat_name] = {"num": int(cat_results['num_example']),
                                                       "acc": round(cat_results['acc'], 3)
                                                      }

                all_ins_acc = calculate_ins_level_acc(evaluation_result)
                printable_results['Overall'] = {"num": sum([cat_results['num_example'] for cat_results in evaluation_result.values()]),
                                                "acc": round(all_ins_acc, 3)
                                                }
                # print("group by catagory:")
                # print(printable_results)

                printable_results['language'] = [language]
                category_data.append(printable_results)
            else:
                print("there is no question type in output json file, thus can't group by category!")

            # group by ability
            # ABILITY2ID = json.load(open(f'{args.answer_path}/ability_label_dict_{version}.json'))
            ABILITY2ID = json.load(open(f'{args.answer_path}/ability_label_dict.json'))
            printable_results = {}
            answer_dict_w_id = {}
            for item in answer_dict:
                data_id, ground_truth = item["question_id"], item["ground_truth"],
                if data_id not in answer_dict_w_id:
                    answer_dict_w_id[data_id] = ""
                answer_dict_w_id[data_id] = ground_truth
            exampels_eval_res = []
            for item in output_dict:
                data_id, parsed_pred = item["question_id"], item["prediction"]
                if data_id not in answer_dict_w_id:
                    continue
                exampels_eval_res.append({
                    "id": data_id,
                    "correct": eval_multi_choice(answer_dict_w_id[data_id], parsed_pred)
                })
            for ability_name, ability_code in ABILITY_DICT.items():
                if language not in ABILITY2ID:
                    # print(f"there is no {language} in ability label json file, thus can't group by ability in {language}!")
                    continue
                id_list = ABILITY2ID[language][ability_code]
                if len(id_list)<1:
                    printable_results[ability_name] = {'num': len(id_list), 'correct': 0 , 'acc': 0}
                    continue
                acc = 0
                for eval_item in exampels_eval_res:
                    if eval_item["id"] in id_list and eval_item["correct"]:
                        acc += 1
                printable_results[ability_name] = {'num': len(id_list), 'correct': acc , 'acc': acc / len(id_list)}
            # print("group by ability:")
            # print(printable_results)
            printable_results['language'] = [language]
            ability_data.append(printable_results)
            
        meta = []
        for category in list(CLUSTER_CAT2QUESTION_CAT.keys()):
            meta.append([f'Overall-{category}', 'acc'])
        meta.append([f'Overall', 'acc'])
        df_1 = json_normalize(
            category_data, 
            record_path='language', 
            meta=meta,
            record_prefix='language',
            # errors='ignore'
        )

        meta = []
        for ability in list(ABILITY_DICT.keys()):
            meta.append([f'{ability}', 'acc'])
        df_2 = json_normalize(
            ability_data, 
            record_path='language', 
            meta=meta,
            record_prefix='language',
            # errors='ignore'
        )

        meta = []
        for category in list(QUESTION_CAT_DICT.keys()):
            meta.append([f'{category}', 'acc'])
        meta.append([f'Overall', 'acc'])
        df_3 = json_normalize(
            category_data, 
            record_path='language', 
            meta=meta,
            record_prefix='language',
            # errors='ignore'
        )

        # eval_result_path = f'{args.result_path}/{model_name}_{version}.xlsx'
        eval_result_path = f'{args.result_path}/{model_name}.xlsx'
        with pd.ExcelWriter(eval_result_path, engine='openpyxl') as writer:
            df_1.T.to_excel(writer, sheet_name='question', index=True)
            df_2.T.to_excel(writer, sheet_name='ability', index=True)
            df_3.T.to_excel(writer, sheet_name='sub_question', index=True)
            df_1.to_excel(writer, sheet_name='question_T', index=True)
            df_2.to_excel(writer, sheet_name='ability_T', index=True)
            df_3.to_excel(writer, sheet_name='sub_question_T', index=True)
