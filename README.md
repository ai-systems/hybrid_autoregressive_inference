# Hybrid Autoregressive Inference for Scalable Multi-hop Explanation Regeneration (AAAI 2022)

## Abstract
Regenerating natural language explanations in the scientific domain has been proposed as a benchmark to evaluate complex multi-hop and explainable inference. In this context, large language models can achieve state-of-the-art performance when employed as cross-encoder architectures and fine-tuned on human-annotated explanations. However, while much attention has been devoted to the quality of the explanations, the problem of performing inference efficiently is largely under-studied. Cross-encoders, in fact, are intrinsically not scalable, possessing limited applicability to real-world scenarios that require inference on massive facts banks. To enable complex multi-hop reasoning at scale, this paper focuses on bi-encoder architectures, investigating the problem of scientific explanation regeneration at the intersection of dense and sparse models. Specifically, we present SCAR (for Scalable Autoregressive Inference), a hybrid framework that iteratively combines a Transformer-based bi-encoder with a sparse model of explanatory power, designed to leverage explicit inference patterns in the explanations. Our experiments demonstrate that the hybrid framework significantly outperforms previous sparse models, achieving performance comparable with that of state-of-the-art cross-encoders while being approx 50 times faster and scalable to corpora of millions of facts. Further analyses on semantic drift and multi-hop question answering reveal that the proposed hybridisation boosts the quality of the most challenging explanations, contributing to improved performance on downstream inference tasks.

## Reproducibility

Welcome! :) 

Here, you can find the code (`explanation_regeneration_experiment.py`) to reproduce the results obtained by SCAR on the [Worldtree Multi-hop Explanation Regeneration Task](https://github.com/umanlp/tg2019task).

**Dense encoder:**

The Bert-based autoregressive bi-encoder used in our experiments can be downloaded [here!](https://drive.google.com/file/d/1iz38q8EIYZdO9U7mAMVz1qUprU8jmEwI/view?usp=sharing)

**Run the experiment:**

Once the dense model is downloaded and the requirements installed, run the following command to start the experiment:

`python ./explanation_regeneration_experiment.py`

This will create the [FAISS](https://faiss.ai/) index and perform multi-hop inference using SCAR.

**Compute the Mean Average Precision (MAP) score:** 

Once the experiment is completed, you can compute the Mean Average Precision (MAP) using the following command:

`./evaluate.py --gold=./data/questions/dev.tsv prediction.txt`

The experiment is performed by default on the dev-set. 

Since the explanations in the test-set are masked, if you want to reproduce our results on this split, you would need to submit `prediction.txt` to the official [leaderboard](https://competitions.codalab.org/competitions/20150#results).

### Bibtex
If you use SCAR in your work, or find this repository useful, please consider citing our paper.


For any issues or questions, feel free to contact us at marco.valentino@manchester.ac.uk
