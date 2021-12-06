"""
Code to reproduce the results on the multi-hop explanation regeneration task ( https://github.com/umanlp/tg2019task )
presented in "Hybrid Autoregressive Inference for Scalable Multi-hop Explanation Regeneration" (AAAI 2022)
"""
import msgpack
from tqdm import tqdm
import json
import csv
import os
import nltk
from nltk.corpus import stopwords

from explanation_retrieval.ranker.bm25 import BM25
from explanation_retrieval.ranker.relevance_score import RelevanceScore
from explanation_retrieval.ranker.explanatory_power import ExplanatoryPower
from explanation_retrieval.ranker.utils import Utils

#load utils
utils = Utils()
utils.init_explanation_bank_lemmatizer()

#Load facts bank
with open("data/cache/table_store.mpk", "rb") as f:
    facts_bank = msgpack.unpackb(f.read(), raw=False)

#Load train set (explanations corpus)
with open("data/cache/eb_train.mpk", "rb") as f:
    eb_dataset_train = msgpack.unpackb(f.read(), raw=False)

#load mapping between questions and hypotheses
with open('data/cache/hypotheses_train.json') as f:
    hypotheses_train = json.load(f)

######### CHAINS EXTRACTION ###########

# open output files to save the final results
chains_output = open("./data/training/chains_train.csv", "w")

# Parameters
K = len(facts_bank.items())
Negative = 5

# load and fit the sparse model
sparse_model = BM25()
facts_bank_lemmatized = []
explanations_corpus_lemmatized = []
ids = []
q_ids = []
# construct sparse index for the facts bank
for t_id, ts in tqdm(facts_bank.items()):
    # facts cleaning and lemmatization
    if "#" in ts["_sentence_explanation"][-1]:
        fact = ts["_sentence_explanation"][:-1]
    else:
        fact = ts["_sentence_explanation"]
    lemmatized_fact = []
    for chunck in fact:
        temp = []
        for word in nltk.word_tokenize(
            chunck.replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(";", " ")
            .replace("-", " ")
        ):
            temp.append(utils.explanation_bank_lemmatize(word.lower()))
        if len(temp) > 0:
            lemmatized_fact.append(" ".join(temp))
    facts_bank_lemmatized.append(lemmatized_fact)
    ids.append(t_id)
# construct sparse index for the explanations corpus
for q_id, exp in tqdm(eb_dataset_train.items()):
    # transform question and candidate answer into a hypothesis
    if exp["_answerKey"] in exp["_choices"]:
        question = hypotheses_train[q_id][exp["_answerKey"]]
    else:
      continue
    temp = []
    # question lemmatization
    for word in nltk.word_tokenize(question):
        temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)
    explanations_corpus_lemmatized.append(lemmatized_question)
    q_ids.append(q_id)
#fit the sparse model
sparse_model.fit(facts_bank_lemmatized, explanations_corpus_lemmatized, ids, q_ids)

#load relevance using the sparse model
RS = RelevanceScore(sparse_model)

# Construct inference chains for each question in the training set
for q_id, exp in tqdm(eb_dataset_train.items()):
    # initialize the partially constructed explanation as an empty list
    partial_explanation = []
    # transform question and candidate answer into a hypothesis
    if exp["_answerKey"] in exp["_choices"]:
          question = hypotheses_train[q_id][exp["_answerKey"]] 
    else:
      continue
    # lemmatization and stopwords removal
    temp = []
    for word in nltk.word_tokenize(question):
        if not word.lower() in stopwords.words("english"):
          temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)
    # for each item in the explanation
    positive_examples = []
    for step in range(len(eb_dataset_train[q_id]["_explanation"])):
        #compute the relevance score for the question 
        relevance_scores_positive = RS.compute(lemmatized_question, K)
        #retrieve the most relevant gold fact
        retrieved = 0
        for fact in sorted(relevance_scores_positive, key=relevance_scores_positive.get, reverse=True):
            if fact in eb_dataset_train[q_id]["_explanation"] and not fact in positive_examples:
                #save positive tuple
                clean_fact = utils.clean_fact(facts_bank[fact]["_explanation"])
                positive_examples.append(fact)
                partial_explanation.append(clean_fact)
                print(question+"\t"+clean_fact+"\t"+str(1), file = chains_output)
                #retrieve negative examples
                # lemmatization and stopwords removal
                temp = []
                for word in nltk.word_tokenize(clean_fact):
                    if not word.lower() in stopwords.words("english"):
                        temp.append(utils.explanation_bank_lemmatize(word.lower()))
                lemmatized_fact = " ".join(temp)
                #retrieve most similar negative facts
                relevance_scores_negative = RS.compute(lemmatized_fact, K)
                count = 0
                for fact_negative in sorted(relevance_scores_negative, key=relevance_scores_negative.get, reverse=True):
                    if not fact_negative in eb_dataset_train[q_id]["_explanation"]:
                        #save negative example
                        clean_fact_negative = utils.clean_fact(facts_bank[fact_negative]["_explanation"])
                        print(question+"\t"+clean_fact_negative+"\t"+str(relevance_scores_negative[fact_negative]), file = chains_output)
                        count += 1
                    if count >= Negative:
                        break
                break
        # update the query concatenating it with the partially constructed explanation
        question = hypotheses_train[q_id][exp["_answerKey"]]
        for fact in partial_explanation:
            question += ". " + fact
        # lemmatization and stopwords removal
        temp = []
        for word in nltk.word_tokenize(question):
            if not word.lower() in stopwords.words("english"):
                temp.append(utils.explanation_bank_lemmatize(word.lower()))
        lemmatized_question = " ".join(temp)

chains_output.close()