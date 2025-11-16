# DL4BSP
Replication Package for the Paper Entitled: Deep Learning for Bug Severity Prediction: How Far Are We?


#### Data and Necessary Files Preparation for PTMs and LLMs
1. Extract the data used to train and evaluate DL models and LLMs via ```tar -zxvf data.tar.gz```.

    **Note:**

    1.1 The **.csv** and **.json** files, under the **data/DL** and **data/LLM** folder, are used to evaluate DL models and LLMs, respectively.
    
    1.2 These **.json** files are generated from **.csv** files using the pre-defined instruction template (**data/LLM/data2Alphaca.py**), which is **(a) a concise classification-oriented instruction** template mentioned in our paper.


2. Manually download the **config.json, dict.txt, merges.txt, pytorch_model.bin, tokenizer.json, tokenizer_config.json, and vocab.json** from [Hugging Face-RoBERTa](https://huggingface.co/FacebookAI/roberta-base/tree/main), upload them to the **roberta-base** folder.

    **Note:**
    
    2.1 The necessary files for other PTMs such as BERT and CodeBERT are downloaded from [Hugging Face-CodeBERT](https://huggingface.co/microsoft/codebert-base/tree/main) and [Hugging Face-BERT](https://huggingface.co/google-bert/bert-base-uncased/tree/main).
    
    2.2 The necessary files for LLMs such as Code Llama, Llama 3.1 and DeepSeek-R1 are downloaded from [ModelScope](https://www.modelscope.cn/home). Specifically, the evaluated models include [CodeLlama-7b-Instruct](https://www.modelscope.cn/models/AI-ModelScope/CodeLlama-7b-Instruct-hf/files), [Llama-3.1-8B-Instruct](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/files), [DeepSeek-R1-Distill-Qwen-1.5B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/files), [DeepSeek-R1-Distill-Qwen-8B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/files), and [DeepSeek-R1-Distill-Llama-8B
    ](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/files).


### Get Involved
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports.