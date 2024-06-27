PROMPT_DICT = {
    "prefix": "Assuming you are an expert in multimodality. The following are multiple-choice questions (with answers) about causal commonsense reasoning. That may include picture information. Please only output options such as A, B, C, D, E, etc. without any explanation.",

    "suffix": "Answer with the option's letter from the given choices directly.", 

    "choice_extraction": "You are an AI assistant who will help me to match an answer with several options of a single-choice question. You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. If the meaning of all options are significantly different from the answer, output Z. You should only do the matching based exactly on the literal meaning of the options and answer. You should not perform any external inference based on your knowledge during the matching. Your should output a single uppercase character in A, B, C, D, E, F (if they are valid options), and Z. \nExample 1: \nQuestion: What is the main object in image? \nOptions: (A) teddy bear (B) rabbit (C) cat (D) dog \nAnswer: a cute teddy bear \nYour output: A \nExample 2: \nQuestion: \nWhat is the main object in image? \nOptions: (A) teddy bear (B) rabbit (C) cat (D) dog \nAnswer: Spider \nYour output: Z \nNow it's your turn: \nQuestion: {question} \nOptions: {options} \nAnswer: {answer} \nYour output:"
}