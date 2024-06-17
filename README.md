# M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark
M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark



## Evaluation
To evaluate a model on our M3GIA, there are the following steps:
1. **Prepare Datasets**: Download M3GIA [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/) and choose the evalution file;
2. **Inference**: Choose the model and the evalution file, you can refer to the following command to get the model predictions of the questions:
```Shell
python cogvlm2_inference.py --question-file ./data/huggingface_noanswer/en_final.parquet --answers-file ./data/answers/cogvlm2-19B/english_v1.jsonl --language english
```
The arguments are as follows:
  - `question-file`: The file you download from [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/).
  - `answers-file`: The output file of the model predictions.
  - `language`: The language chosen from english, chinese, french, spanish, portuguese and korean.
3. **Evaluation**: 
After inference all the questons in 6 languages, you can run the following command to get the final results (Please check the folder: `data/eval_result`):
```bash
python main_eval_only.py --model_name_list cogvlm2-19B --language_list chinese english spanish french portuguese korean
```
