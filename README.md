# Comparing Selective Masking Methods for Depression Detection in Social Media

## Abstract
Identifying those at risk for depression is a crucial issue in which social media provides an excellent platform for examining the linguistic patterns of depressed individuals. A significant challenge in a depression classification problem is ensuring that the prediction model is not overly dependent on keywords, such that it fails to predict when keywords are unavailable. One promising approach is masking, i.e., by masking important words selectively and asking the model to predict the masked words, the model is forced to learn the context rather than the keywords. This study evaluates seven masking techniques, such as random masking, log-odds ratio, and the use of attention scores. In addition, whether to predict the masked words during pretraining or fine-tuning phase was also examined. Last, six class imbalance ratios were compared to determine the robustness of the masked selection methods. Key findings demonstrated that selective masking generally outperforms random masking in terms of classification accuracy.  In addition, the most accurate and robust models were identified. Our research also indicated that reconstructing the masked words during the pre-training phase is more advantageous than during the fine-tuning phase. Further discussion and implications were made.  This is the first study to comprehensively compare masking selection methods, which has broad implications for the field of depression classification and the general NLP.

## Dataset
- Reddit Self-reported Depression Diagnosis (RSDD) dataset and Time-RSDD dataset (https://georgetown-ir-lab.github.io/emnlp17-depression/)

The datasets should be loaded into the <code>OP_datasets</code> folder

## Training Approaches
- BERT further pre-train + fine-tune <code>FURTHER-01-MLM.py</code> and <code>FURTHER-02-classi.py</code> (adapted from https://github.com/GU-DataLab/stance-detection-KE-MLM and https://github.com/thunlp/SelectiveMasking)
- BERT fine-tune with reconstruction objective <code>MASKER.py</code> (adapted from https://github.com/alinlab/MASKER)
- Standard BERT fine-tune <code>BASE-classi.py</code>

## Selective Masking Methods
1. Random masking <code>random</code>
2. Depression Lexicon <code>deplex</code> (lexicon.txt from https://github.com/gamallo/depression_classification/tree/master/lexicons)
3. Log-odds-ratio <code>logodds</code> (from https://github.com/kornosk/log-odds-ratio)
4. TF-IDF <code>tfidf</code> (adapted from https://github.com/alinlab/MASKER)
5. Sum attention <code>sumatt</code> (adapted from https://github.com/alinlab/MASKER)
6. Top attention <code>prop</code>
7. Neural Network <code>NN</code> (adapted from https://github.com/thunlp/SelectiveMasking)

<code>get_datasets</code> contains python script and .ipynb files for extracting, preprocesing and creating the dataset objects for training

<code>keyword</code> contains .ipynb files for obtaining the keywords and the resulting keywords in .txt format

<code>src</code> contain the source code for creating a masked dataset and training & evaluation loop





