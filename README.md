# Neural Topic Model via Optimal Transport

This repo contains the Tensorflow implementation of the ICLR paper [1].

## Requirements
The code runs on Tensorflow 1.0 and should be easily adapted to Tensorflow 2.0. The requirements are in the file of ```requirements.txt```. Please use numpy==1.19.5 or lower to avoid incompatibility with Tensorflow 1.15.

## Data

We provided the preprocessed versions of the datasets used in the paper, including 20News, WS, and TMN, under the folder of ```datasets```.

Each dataset is stored in a MAT file with the following contents:
- wordsTrain: V by N<sub>train</sub> count matrix for N<sub>train</sub> training documents with V words in the vocabulary
- wordsTest: V by N<sub>test</sub> count matrix for N<sub>test</sub> testing documents
- labelsTrain: Indexes of the labels of the training documents
- labelsTest: Indexes of the labels of the testing documents 
- labelsToGroup: Names of the labels
- vocabulary: Words in the vocabulary
- embeddings: Pretrained 50-dimensional GloVe word embeddings for the words in the vocabulary

Please prepare your own documents in the above format. If you want to use this dataset, please cite the original papers, which are cited in our paper.

## Run NSTM

Simply ```python nstm.py --dataset=20News --K=100```. For other settings, please see ```nstm.py``` as well as our paper.

The files of a run of the model will be saved under the folder of ```save```. 

## Evaluation

- We provided the Matlab functions (Matlab and Java installations required) to compute topic diversity (with all the topics), top-Purity/NMI and km-Purity/MNI for document clustering.
- Run ```evaluation/evaluate.m``` will give the above results.
- To evaluate the topic coherence results, we used [Palmetto](https://github.com/dice-group/Palmetto), which is not provided in this repo. One needs to download and set up separately.

## Misc

- The computing of document clustering results is based on the Java implementation of [LFTM](https://github.com/datquocnguyen/LFTM/tree/master/src/eval).
- We provided independent Tensorflow (1.0 and 2.0) and Pytorch implementations of the Sinkhorn algorithm used in our model at [Tensorflow_Pytorch_Sinkhorn_OT](https://github.com/ethanhezhao/Tensorflow_Pytorch_Sinkhorn_OT), which can be used for computing the optimal transport distances between discrete distributions in general.
- The code comes without support.

[1] He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine: Neural Topic Model via Optimal Transport, ICLR 2021
