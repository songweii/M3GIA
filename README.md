# M3GIA: A Cognition Inspired Multilingual and Multimodal General Intelligence Ability Benchmark

[**🌐 Homepage**] | [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/) | [**📖 arXiv**](https://arxiv.org/abs/2406.05343) | [**GitHub**]

[**Abstract**]
As recent multi-modality large language models (MLLMs) have shown formidable proficiency on various complex tasks, there has been increasing attention on debating whether these models could eventually mirror human intelligence.
However, existing benchmarks mainly focus on evaluating solely on task performance, such as the accuracy of identifying the attribute of an object. Combining well-developed cognitive science to understand the intelligence of MLLMs beyond superficial achievements remains largely unexplored. To this end, we introduce the first cognitive-driven multi-lingual and multi-modal benchmark to evaluate the general intelligence ability of MLLMs, dubbed M3GIA. Specifically, we identify five key cognitive factors based on the well-recognized Cattell-Horn-Carrol (CHC) model of intelligence and propose a novel evaluation metric. In addition, since most MLLMs are trained to perform in different languages, a natural question arises: is language a key factor influencing the cognitive ability of MLLMs? As such, we go beyond English to encompass other languages based on their popularity, including Chinese, French, Spanish, Portuguese and Korean, to construct our M3GIA. We make sure all the data relevant to the cultural backgrounds are collected from their native context to avoid English-centric bias. 
We collected a significant corpus of data from human participants, revealing that the most advanced MLLM reaches the lower boundary of human intelligence in English. Yet, there remains a pronounced disparity in the other five languages assessed. We also reveals an interesting winner takes all phenomenon that are aligned with the discovery in cognitive studies.

## Evaluation
To evaluate a model on our M3GIA, there are the following steps:
1. **Prepare Datasets**: Download M3GIA [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/) and choose the evalution file;
2. **Inference**: Choose the model and the evalution file, you can refer to the following command to get the model predictions of the questions. In this demo, we take CogVLM-2 as an example:
```Shell
python inference_only.py --question-file ./data/huggingface/en_final.parquet --answers-file ./data/answers --language english --model-name cogvlm2-19B --model-path ./weight/cogvlm/cogvlm2-llama3-chinese-chat-19B
```
The arguments are as follows:
  - `question-file`: The file you download from [**🤗 Dataset**](https://huggingface.co/datasets/Songweii/M3GIA/).
  - `answers-file`: The output file of the model predictions.
  - `model-name`: The name of the model you evaluate.
  - `model-path`: The path to the model's weight if you want to evaluate an open source model.
  - `language`: The language chosen from english, chinese, french, spanish, portuguese and korean.

If you want to evaluate a closed-source model:
```Shell
python inference_only.py --question-file ./data/huggingface/en_final.parquet --answers-file ./data/answers --language english --model-name gpt-4v --api-key *** --api-base ***
```
The arguments are as follows:
  - `api-key`: The API key of the closed-source model evaluated.
  - `api-base`: The API base of the closed-source model evaluated.

3. **Evaluation**: 
After inference all the questons in 6 languages, you can run the following command to get the final results (Please check the folder: `data/eval_result`):
```bash
python main_parse_and_eval.py --model_name_list cogvlm2-19B --language_list chinese english spanish french portuguese korean
```

If you want to use LLM for answer extraction:
```bash
python main_parse_and_eval.py --model_name_list cogvlm2-19B --language_list chinese english spanish french portuguese korean --openai-api-key *** --openai-api-base ***
```
