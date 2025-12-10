# A Multi-solution Study on GDPR AI-enabled Completeness Checking of DPAs


This is the supplementary material repository of the paper "A Multi-solution Study on GDPR AI-enabled Completeness Checking of DPAs". In this paper, we explore the performance of multiple alternative solutions to address the completeness checking of DPAs against GDPR provisions as a text classification problem. The alternative solutions are based on different technologies including traditional machine learning, deep learning, language modeling, and few-shot learning. The objective of this study is to evaluate how these different technologies perform in the legal domain.


The code is developed at SnT / University of Luxembourg with funding from Luxembourg's National Research Fund (FNR).


## What is released?

- ./Resources/pretrained_models/multi_class: is a directory that contains a trained multi-class classification model based on RoBERTa. Note that RoBERTa performed the best in addressing the completeness of DPA against GDPR provisions as a multi-class classification problem.
- ./Resources/pretained_models/binary: is a directory that contains 19 trained binary models based on BERT. Note that BERT performed the best in binary classification problem.
- ./train_models.ipynb: is a Python notebook that contains code we used for training the large language models including BERT, RoBERTa, ALBERT, and Legal-BERT.
- ./test_models.ipynb: is a Python notebook that contains code for testing our pertained models.
- ./baseline_models.ipynb: is a python notebook that contains code we used for training three machine learning models Logistic Regression, Support Vector Machines, Random Forest, two deep learning models Multi-layer Perceptron, and BiLSTM, and one few-shot learning model based on SetFit framework.
- ./Resources/utils.py: is a Python file that contains code that facilitates the three notebooks to perform model training and testing.
- ./Input/train_set.csv & test_set.csv: contain the train and testfrom the non-proprietary DPAs respectively. 
- ./requriements.txt: is a text file containing the required Python libraries needed to run the three notebooks


## How to Cite?
- Not yet published





