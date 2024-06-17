# M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark
M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark



## Evaluation
To evaluate a model on our M3GIA, there are the following steps:
- **Prepare Datasets**: Download M3GIA [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/) and choose the evalution file;
- **Inference**: Choose the Model and the evalution file, you can refer to the following command to get the model predictions of the questions:
```Shell
python cogvlm2_inference.py --question-file /mnt/data/sw/Marco_VL_Bench/data/huggingface_noanswer/en_final.parquet --answers-file /mnt/data/sw/M3GIA_Bench/data/answers/cogvlm2-19B/english_v1.jsonl --language english
```
The arguments are as follows:
  - `question-file`: You can download the file from [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/).
  - `answers-file`: The output file of the model predicitons.
  - `language`: The language chosen from english, chinese, french, spanish, portuguese and korean.
- **Evaluation**: 
After inference all the 6 languages, you can run the following command to get the final results (Please check the folder: `data/eval_result`):
```bash
python main_eval_only.py --model_name_list cogvlm2-19B --language_list chinese english spanish french portuguese korean
```
