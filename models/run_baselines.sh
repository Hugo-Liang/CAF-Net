#!/bin/bash
# ===============================
# 批量运行多组实验的脚本
# python BERT_cmp.py > ./results/BERT/train_BERT.log 2>&1
# echo "Finished training BERT~"
python BERT_clean.py > ./results/BERT-clean/train_BERT-clean.log 2>&1
echo "Finished training BERT_clean~"
python CodeBERT_cmp.py > ./results/CodeBERT/train_CodeBERT.log 2>&1
echo "Finished training CodeBERT~"
python CodeBERT_clean.py > ./results/CodeBERT-clean/train_CodeBERT-clean.log 2>&1
echo "Finished training CodeBERT_clean~"
python RoBERTa_cmp.py > ./results/RoBERTa/train_RoBERTa.log 2>&1
echo "Finished training RoBERTa~"
python RoBERTa_clean.py > ./results/RoBERTa-clean/train_RoBERTa-clean.log 2>&1
echo "Finished training RoBERTa_clean~"

echo "Experiments Done !"
