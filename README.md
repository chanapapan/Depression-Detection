# Comparing Selective Masking Methods for Depression Detection in Social Media

## Abstract
Identifying those at risk for depression is a crucial issue in which social media provides an excellent platform for examining the linguistic patterns of depressed individuals. A significant challenge in a depression classification problem is ensuring that the prediction model is not overly dependent on keywords, such that it fails to predict when keywords are unavailable. One promising approach is masking, i.e., by masking important words selectively and asking the model to predict the masked words, the model is forced to learn the context rather than the keywords. This study evaluates seven masking techniques, such as random masking, log-odds ratio, and the use of attention scores. In addition, whether to predict the masked words during pretraining or fine-tuning phase was also examined. Last, six class imbalance ratios were compared to determine the robustness of the masked selection methods. Key findings demonstrated that selective masking generally outperforms random masking in terms of classification accuracy.  In addition, the most accurate and robust models were identified. Our research also indicated that reconstructing the masked words during the pre-training phase is more advantageous than during the fine-tuning phase. Further discussion and implications were made.  This is the first study to comprehensively compare masking selection methods, which has broad implications for the field of depression classification and the general NLP.

Training Approaches
- BERT further pre-train + fine-tune (adapted from https://github.com/GU-DataLab/stance-detection-KE-MLM and https://github.com/thunlp/SelectiveMasking)
- BERT fine-tune with reconstruction objective (adapted from https://github.com/alinlab/MASKER)

Selective Masking Methods
- Random masking
- Depression Lexicon (lexicon.txt from https://github.com/gamallo/depression_classification/tree/master/lexicons)
- Log-odds-ratio (from https://github.com/kornosk/log-odds-ratio)
- TF-IDF (adapted from https://github.com/alinlab/MASKER)
- Sum attention (adapted from https://github.com/alinlab/MASKER)
- Top attention
- Neural Network (adapted from https://github.com/thunlp/SelectiveMasking)


